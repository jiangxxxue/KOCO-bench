import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

ds_names = ["GSM8K", "MATH500"]
ds = [
    load_dataset(
        f"RLHFlow/Deepseek-{ds_name}-Test"
    )['test'] for ds_name in ds_names
]

def make_step_rewards(logits, token_masks):
    all_scores_res = []
    for sample, token_mask in zip(logits, token_masks):
        # sample: (seq_len, num_labels)
        probs = sample[token_mask].softmax(dim=-1)  # (num_steps, 2)
        process_reward = probs[:, 1] - probs[:, 0]  # (num_steps,)
        # weighted sum to approx. min, highly recommend when BoN eval and Fine-tuning LLM
        weight = torch.softmax(
            -process_reward / 0.1, 
            dim=-1,
        )
        process_reward = weight * process_reward
        all_scores_res.append(process_reward.cpu().tolist())
    return all_scores_res


model_name = "jinachris/PURE-PRM-7B"
device = "auto"

tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    trust_remote_code=True,
)
model = AutoModelForTokenClassification.from_pretrained(
    model_name, 
    device_map=device, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval()

step_separator = "\n"
step_separator_token = tokenizer(
    step_separator, 
    add_special_tokens=False, 
    return_tensors='pt',
)['input_ids']


for ds_item, ds_name in zip(ds, ds_names):
    # sampled_ids = np.random.choice(range(len(ds_item)), size=100, replace=False)
    correct = 0
    total = 0
    for idx in tqdm(range(len(ds_item)), desc=f"Processing questions in {ds_name}"):
        question = ds_item['prompt'][idx]
        answers = ds_item['answers'][idx]
        labels = ds_item['label'][idx]
        outcome_scores = []

        question_ids = tokenizer(
            question, 
            add_special_tokens=False, 
            return_tensors='pt',
        )['input_ids']
        for answer in tqdm(answers, desc="Processing answers"):
            steps = [i.rstrip() for i in answer.split("\n\n")]
            input_ids = question_ids.clone()

            score_ids = []
            for step in steps:
                step_ids = tokenizer(
                    step, 
                    add_special_tokens=False, 
                    return_tensors='pt',
                )['input_ids']
                input_ids = torch.cat(
                    [input_ids, step_ids, step_separator_token], 
                    dim=-1,
                )
                score_ids.append(input_ids.size(-1) - 1)

            input_ids = input_ids.to(model.device, dtype=torch.long)
            token_masks = torch.zeros_like(input_ids, dtype=torch.bool)
            token_masks[0, score_ids] = True
            assert torch.all(input_ids[token_masks].to("cpu") == step_separator_token)
            
            with torch.no_grad():
                logits = model(input_ids).logits
                step_reward = make_step_rewards(logits, token_masks)
                outcome_reward = torch.tensor(step_reward).sum(dim=-1)

            # TODO: batch input & output
            outcome_scores.append(outcome_reward.item())
        
        best_idx = np.argmax(outcome_scores)
        if labels[best_idx] == 1:
            correct += 1
        total += 1
    print(f"Accuracy on {ds_name}: {correct / total}")
