# Copyright (2025) critic-rl Authors 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License.
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from tenacity import (
    retry,
    retry_any,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_random_exponential,
)


def is_empty_result(result):
    return (
        result is None
        or (isinstance(result, str) and not result.strip())
        or (isinstance(result, list) and not any(result))
    )


def build_singleturn_messages(prompt, system_prompt=None):
    messages = []
    if system_prompt is not None:
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )
    messages.append({"role": "user", "content": prompt})
    return messages


def chat_api_response(messages, model, api_key, base_url, api_version=None, **kwargs):
    if api_version is None:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    else:  # use Azure API
        client = AzureOpenAI(
            azure_endpoint=base_url, api_version=api_version, api_key=api_key
        )
    response = client.chat.completions.create(
        model=model, messages=messages, stream=False, **kwargs
    )

    # extract the response from the completion
    if response is not None:
        result = response.choices[0].message.content
    else:
        result = None
    return result


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    retry=retry_any(
        retry_if_result(is_empty_result),
        retry_if_exception_type(Exception),  # This will catch any type of error
    ),
)
async def async_chat_api_response(
    messages, model, api_key, base_url, api_version=None, **kwargs
):
    if api_version is None:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    else:  # use Azure API
        client = AsyncAzureOpenAI(
            azure_endpoint=base_url, api_version=api_version, api_key=api_key
        )

    response = await client.chat.completions.create(
        model=model, messages=messages, stream=False, **kwargs
    )

    if response is not None:
        choices = response.choices

        if len(choices) == 1:
            result = choices[0].message.content
        else:
            result = [c.message.content for c in choices]
    else:
        result = None
    return result
