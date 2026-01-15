import argparse
import copy
import json
from PIL import Image

import torch
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_dataset


def convert_example(example):
    """
    correct example into "messages" 
    eg:
    {
      "system": "You are a helpful assistant.",
      "conversations": [
          {"from": "user", "value": "How many objects are included in this image?",
           "image_path": "/path/to/image.png"},
          {"from": "assistant", "value": "<think>\nI can see 10 objects\n</think>\n<answer>\n10\n</answer>"}
      ]
    }
    """
    messages = []
    
    image = example.get("image")
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": example["problem"]},
            {"type": "image", "image": image},
            ]
    })
    
    return messages


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2-VL model on validation dataset")
    parser.add_argument("--eval_data_path", type=str, required=True, help="Path to evaluation dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save evaluation results")
    return parser.parse_args()


def main():

    args = parse_args()
    model_path = args.model_path
    eval_data_path = args.eval_data_path
    save_path = args.save_path

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)


    eval_dataset = load_dataset(eval_data_path)


    tot_num, correct_num = 0, 0

    SPEED_PLAN = ['KEEP', 'ACCELERATE', 'DECELERATE', 'STOP']
    PATH_PLAN = ['RIGHT_TURN', 'RIGHT_CHANGE', 'LEFT_TURN', 'LEFT_CHANGE', 'STRAIGHT']

    metric_tot_cnt = {speed + '_' + path: 0 for speed in SPEED_PLAN for path in PATH_PLAN}
    metric_correct_cnt = copy.deepcopy(metric_tot_cnt)

    eval_record = {}

    speed_tp = {
        'KEEP': 0,
        'ACCELERATE': 0,
        'DECELERATE': 0,
        'STOP': 0,
    }

    speed_fp, speed_fn = copy.deepcopy(speed_tp), copy.deepcopy(speed_tp)

    path_tp = {
        'RIGHT_TURN': 0,
        'LEFT_TURN': 0,
        'RIGHT_CHANGE': 0,
        'LEFT_CHANGE': 0,
        'STRAIGHT': 0,
    }

    path_fp, path_fn = copy.deepcopy(path_tp), copy.deepcopy(path_tp)

    f1_score = {
        'KEEP': 0,
        'ACCELERATE': 0,
        'DECELERATE': 0,
        'STOP': 0,
        'RIGHT_TURN': 0,
        'LEFT_TURN': 0,
        'RIGHT_CHANGE': 0,
        'LEFT_CHANGE': 0,
        'STRAIGHT': 0,
    }

    for sample in tqdm(eval_dataset['validation']):

        tot_num = tot_num + 1

        gt_answer = sample['solution']

        messages = convert_example(sample)

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        answer = answer.strip()

        gt_answer = gt_answer.replace('<answer> ', '')
        gt_answer = gt_answer.replace(' </answer>', '')
        speed_plan, path_plan = gt_answer.split(', ')
        path_plan = path_plan.split('\n')[0]


        for key in speed_tp.keys():
            if key in answer:  # P
                if speed_plan in answer:
                    speed_tp[key] += 1  # TP
                else:
                    speed_fp[key] += 1  # FP
            else:  # N
                if key in speed_plan:
                    speed_fn[key] += 1  # FN

        for key in path_tp.keys():
            if key in answer:  # P
                if path_plan in answer:
                    path_tp[key] += 1  # TP
                else:
                    path_fp[key] += 1  # FP
            else:  # N
                if key in path_plan:
                    path_fn[key] += 1  # FN


        metric_tot_cnt[speed_plan+'_'+path_plan] += 1
        if speed_plan in answer and path_plan in answer:
            correct_num = correct_num + 1
            metric_correct_cnt[speed_plan+'_'+path_plan] += 1
        else:
            fail_case = {
                'gt': gt_answer,
                'pred': answer,
            }


    for key in f1_score.keys():
        if key in speed_tp.keys():
            if speed_tp[key] + speed_fp[key] != 0:
                precision = speed_tp[key] / (speed_tp[key] + speed_fp[key])
            else:
                precision = 0
            if speed_tp[key] + speed_fn[key] != 0:
                recall = speed_tp[key] / (speed_tp[key] + speed_fn[key])
            else:
                recall = 0
            if precision + recall != 0:
                f1_score[key] = 2.0 * precision * recall / (precision + recall)
            else:
                f1_score[key] = 0
        if key in path_tp.keys():
            if path_tp[key] + path_tp[key] != 0:
                precision = path_tp[key] / (path_tp[key] + path_fp[key])
            else:
                precision = 0
            if path_tp[key] + path_fn[key] != 0:
                recall = path_tp[key] / (path_tp[key] + path_fn[key])
            else:
                recall = 0
            if precision + recall != 0:
                f1_score[key] = 2.0 * precision * recall / (precision + recall)
            else:
                f1_score[key] = 0

    print("\n\n=========== F1 Score ===========\n\n")
    for k, v in f1_score.items():
        print(f"{k}: {v}")
    print("\n\n================================\n\n")

    print(f'\nTotal Number: {tot_num}\n')
    print(f'\nCorrect Number: {correct_num}\n')

    print('\n------------------------------\n\n')
    print(f"Planning Accuracy: {correct_num/tot_num * 100:.2f}%")
    print('\n\n------------------------------\n')

    for key in metric_tot_cnt.keys():
        if metric_tot_cnt[key] > 0:
            print(f"{key}: num: {metric_tot_cnt[key]}, correct num: {metric_correct_cnt[key]}, {100*metric_correct_cnt[key]/metric_tot_cnt[key]:.2f}%")

    eval_record['summary'] = f'Total Number: {tot_num}'
    eval_record['summary'] = eval_record['summary'] + '\n' + f'Correct Number: {correct_num}'
    eval_record['summary'] = eval_record['summary'] + '\n' + f"Planning Accuracy: {correct_num/tot_num * 100:.2f}%"

    for key in metric_tot_cnt.keys():
        if metric_tot_cnt[key] > 0:
            eval_record['summary'] = eval_record['summary'] + '\n' + \
                f"{key}: num: {metric_tot_cnt[key]}, correct num: {metric_correct_cnt[key]}, {100*metric_correct_cnt[key]/metric_tot_cnt[key]:.2f}%"

    eval_record['f1_score'] = {}
    for k, v in f1_score.items():
        eval_record['f1_score'][k] = v

    with open(save_path, "w") as f:
        json.dump(eval_record, f)
        print(f'\nEval results saved to {save_path}\n')


if __name__ == "__main__":
    main()
