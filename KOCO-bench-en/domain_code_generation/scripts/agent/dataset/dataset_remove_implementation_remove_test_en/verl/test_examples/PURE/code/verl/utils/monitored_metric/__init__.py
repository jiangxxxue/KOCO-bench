import asyncio
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import islice, zip_longest

import torch

from verl import DataProto


def repeatness(s: str):
    def ranks(l):
        index = {v: i for i, v in enumerate(sorted(set(l)))}
        return [index[v] for v in l]

    def suffixArray(s):
        line = ranks(s)
        n, k, ans, sa = len(s), 1, line, [0] * len(s)
        while k < n - 1:
            line = ranks(list(zip_longest(line, islice(line, k, None), fillvalue=-1)))
            ans, k = line, k << 1
        for i, k in enumerate(ans):
            sa[k] = i
        return ans, sa

    def lcp(arr, suffixArr, inv_suff):
        n, ans, k = len(arr), [0] * len(arr), 0

        for i in range(n):
            if inv_suff[i] == n - 1:
                k = 0
                continue

            j = suffixArr[inv_suff[i] + 1]
            while i + k < n and j + k < n and arr[i + k] == arr[j + k]:
                k += 1

            ans[inv_suff[i]] = k
            if k > 0:
                k -= 1

        return ans

    arr = [ord(i) for i in s]
    n = len(arr)
    if n <= 1:
        return 0
    c, sa = suffixArray(arr)
    cnt = sum(lcp(arr, sa, c))

    return cnt * 2 / (n * (n + 1))


def check_reflection_pattern(response: str) -> dict[str, int]:
    # TODO: may need to add more pattern
    reflection_pattern_words = [
        r"wait,",
        r"recheck[,\s]",
        r"retry[,\s]",
        r"try again",
        r"alternatively,",
        r"however,",
        r"rethink[,\s]",
        r"let's check",
        r"let's verify",
    ]
    res = defaultdict(int)
    for word in reflection_pattern_words:
        res[word] = len(re.findall(word, response))
    return res


def get_reflection_pattern_score(s: str):
    reflection_pattern_dict = check_reflection_pattern(s)
    reflection_pattern_num = sum(reflection_pattern_dict.values())
    return reflection_pattern_num


def get_repeatness_and_reflection_score_sync(response):
    return repeatness(response), get_reflection_pattern_score(response.lower())


async def single_get_repeatness_and_reflection_score(response, executor):
    num_steps = len(response.split('\n\n'))

    loop = asyncio.get_running_loop()
    try:
        repeat, reflection = await loop.run_in_executor(
            executor,
            partial(get_repeatness_and_reflection_score_sync, response)
        )
        return (num_steps, repeat, reflection)
    except asyncio.TimeoutError:
        print(f"Timeout occurred for completion: {response}")
        return (num_steps, 0.0, 0)
    except Exception as e:
        print(f"Error processing completion: {response[:10]}, Error: {e}")
        return (num_steps, 0.0, 0)


async def parallel_get_repeatness_and_reflection_score_async(responses, num_processes=64):
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = [
            single_get_repeatness_and_reflection_score(
                response, executor,
            ) for response in responses
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)
    
    num_steps = [result[0] for result in results]
    repeatness_list = [result[1] for result in results if result[1] is not None]
    reflection_scores = [result[2] for result in results if result[2] is not None]
    return num_steps, repeatness_list, reflection_scores


def get_repeatness_and_reflection_score(data: DataProto, tokenizer, high_repeat_threshold=0.05):
    response_tokens = data.batch['responses']
    responses = tokenizer.batch_decode(response_tokens, skip_special_tokens=True)

    num_steps, repeat_score, reflection_score = asyncio.run(
        parallel_get_repeatness_and_reflection_score_async(
            responses,
            num_processes=64,
        )
    )
    repeat_score = torch.tensor(repeat_score, dtype=torch.float32)
    output = DataProto.from_dict(dict(
        num_steps=torch.tensor(num_steps, dtype=torch.float32),
        repeat_score=repeat_score,
        repeat_mask=(repeat_score > high_repeat_threshold).to(dtype=torch.float32),
        reflection_pattern_score=torch.tensor(reflection_score, dtype=torch.float32),
    ))
    return output
