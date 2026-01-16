"""
Gemini 2.5 Flash LLM Integration for RAG-Anything
Compatible with LightRAG's openai_complete_if_cache interface
"""

import os
import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional


async def gemini_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict[str, str]] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> str:
    """
    Gemini 2.5 Flash completion function compatible with LightRAG's interface

    Args:
        model: Model name (e.g., "gemini-2.5-flash")
        prompt: User prompt text
        system_prompt: System prompt (optional)
        history_messages: Chat history (optional)
        api_key: Vertex AI API key
        **kwargs: Additional arguments (ignored for Gemini)

    Returns:
        Generated text response from Gemini
    """
    if history_messages is None:
        history_messages = []

    # Get API key from environment if not provided
    if not api_key:
        api_key = os.getenv("VERTEX_AI_API_KEY")
        if not api_key:
            raise ValueError("VERTEX_AI_API_KEY not found in environment")

    # Build Gemini API endpoint
    # Use streamGenerateContent for streaming responses
    endpoint = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:streamGenerateContent?key={api_key}"

    # Build content structure for Gemini
    # Gemini expects: {"contents": [{"role": "user", "parts": [{"text": "..."}]}]}
    contents = []

    # Add system prompt as first user message (Gemini doesn't have system role)
    if system_prompt:
        contents.append({
            "role": "user",
            "parts": [{"text": f"System instructions: {system_prompt}"}]
        })

    # Add history messages
    for msg in history_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Map OpenAI roles to Gemini roles
        if role == "assistant":
            gemini_role = "model"
        else:
            gemini_role = "user"

        contents.append({
            "role": gemini_role,
            "parts": [{"text": content}]
        })

    # Add current prompt
    contents.append({
        "role": "user",
        "parts": [{"text": prompt}]
    })

    # DEBUG: Log the prompt being sent to Gemini
    print(f"\n[DEBUG] Gemini prompt length: {len(prompt)} characters")
    print(f"[DEBUG] First 500 chars: {prompt[:500]}")
    print(f"[DEBUG] Last 500 chars: {prompt[-500:]}\n")

    # Build request payload
    payload = {
        "contents": contents
    }

    # Make async request to Gemini API
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=aiohttp.ClientTimeout(total=180)  # 180s timeout like LightRAG
            ) as response:

                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Gemini API error {response.status}: {error_text}")

                # Parse response - Gemini returns JSON array
                full_text = await response.text()
                result_text = ""

                try:
                    # Parse entire response as JSON (it's an array)
                    data = json.loads(full_text)

                    # Handle array response
                    if isinstance(data, list):
                        for item in data:
                            if 'candidates' in item:
                                for candidate in item['candidates']:
                                    if 'content' in candidate and 'parts' in candidate['content']:
                                        for part in candidate['content']['parts']:
                                            if 'text' in part:
                                                result_text += part['text']
                    # Handle single object response
                    elif isinstance(data, dict):
                        if 'candidates' in data:
                            for candidate in data['candidates']:
                                if 'content' in candidate and 'parts' in candidate['content']:
                                    for part in candidate['content']['parts']:
                                        if 'text' in part:
                                            result_text += part['text']

                except json.JSONDecodeError as e:
                    print(f"[ERROR] Failed to parse Gemini response: {e}")
                    print(f"[DEBUG] Response: {full_text[:500]}")
                    return ""

                return result_text.strip() if result_text else ""

    except asyncio.TimeoutError:
        raise Exception("Gemini API request timed out after 180 seconds")
    except Exception as e:
        raise Exception(f"Gemini API call failed: {str(e)}")


async def gemini_vision_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict[str, str]] = None,
    image_data: Optional[str] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> str:
    """
    Gemini 2.5 Flash vision completion for multimodal content

    Args:
        prompt: Text prompt
        system_prompt: System prompt (optional)
        history_messages: Chat history (optional)
        image_data: Base64 encoded image data (optional)
        messages: Pre-formatted messages (optional)
        api_key: Vertex AI API key
        **kwargs: Additional arguments

    Returns:
        Generated response for multimodal content
    """
    if history_messages is None:
        history_messages = []

    # Get API key
    if not api_key:
        api_key = os.getenv("VERTEX_AI_API_KEY")
        if not api_key:
            raise ValueError("VERTEX_AI_API_KEY not found in environment")

    # Use Gemini 2.5 Flash for vision (supports multimodal)
    model = "gemini-2.5-flash"
    endpoint = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:streamGenerateContent?key={api_key}"

    # Build content structure
    contents = []

    # Add system prompt
    if system_prompt:
        contents.append({
            "role": "user",
            "parts": [{"text": f"System instructions: {system_prompt}"}]
        })

    # Handle pre-formatted messages (multimodal VLM format)
    if messages:
        for msg in messages:
            if msg.get("role") == "system":
                continue  # Already handled

            role = "model" if msg.get("role") == "assistant" else "user"
            msg_content = msg.get("content", "")

            # Handle multimodal content
            if isinstance(msg_content, list):
                parts = []
                for item in msg_content:
                    if item.get("type") == "text":
                        parts.append({"text": item.get("text", "")})
                    elif item.get("type") == "image_url":
                        # Extract base64 image data
                        image_url = item.get("image_url", {}).get("url", "")
                        if "base64," in image_url:
                            b64_data = image_url.split("base64,")[1]
                            parts.append({
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": b64_data
                                }
                            })

                if parts:
                    contents.append({"role": role, "parts": parts})
            else:
                contents.append({
                    "role": role,
                    "parts": [{"text": str(msg_content)}]
                })

    # Handle traditional single image format
    elif image_data:
        parts = [{"text": prompt}]

        # Add image data
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": image_data
            }
        })

        contents.append({
            "role": "user",
            "parts": parts
        })

    # Pure text format
    else:
        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })

    # Build request
    payload = {"contents": contents}

    # Make request
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=aiohttp.ClientTimeout(total=180)
            ) as response:

                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Gemini Vision API error {response.status}: {error_text}")

                # Parse response - Gemini returns JSON array
                full_text = await response.text()
                result_text = ""

                try:
                    # Parse entire response as JSON (it's an array)
                    data = json.loads(full_text)

                    # Handle array response
                    if isinstance(data, list):
                        for item in data:
                            if 'candidates' in item:
                                for candidate in item['candidates']:
                                    if 'content' in candidate and 'parts' in candidate['content']:
                                        for part in candidate['content']['parts']:
                                            if 'text' in part:
                                                result_text += part['text']
                    # Handle single object response
                    elif isinstance(data, dict):
                        if 'candidates' in data:
                            for candidate in data['candidates']:
                                if 'content' in candidate and 'parts' in candidate['content']:
                                    for part in candidate['content']['parts']:
                                        if 'text' in part:
                                            result_text += part['text']

                except json.JSONDecodeError as e:
                    print(f"[ERROR] Failed to parse Gemini vision response: {e}")
                    return ""

                return result_text.strip() if result_text else ""

    except asyncio.TimeoutError:
        raise Exception("Gemini Vision API request timed out")
    except Exception as e:
        raise Exception(f"Gemini Vision API call failed: {str(e)}")
