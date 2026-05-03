import os
import json
import httpx
from fastapi import HTTPException

async def generate_with_cloud(prompt: str, system_prompt: str = None, is_json: bool = False):
    """
    Unified cloud inference helper using Groq as primary and NVIDIA as fallback.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found in environment")
        return await _try_nvidia_fallback(prompt, system_prompt, is_json)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "temperature": 0.5,
    }
    
    if is_json:
        payload["response_format"] = {"type": "json_object"}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload
            )
            
            if response.status_code != 200:
                print(f"⚠️ Groq Error ({response.status_code}): {response.text}")
                return await _try_nvidia_fallback(prompt, system_prompt, is_json)
                
            data = response.json()
            return data["choices"][0]["message"]["content"]
            
    except Exception as e:
        print(f"⚠️ Groq Connection Failed: {str(e)}")
        return await _try_nvidia_fallback(prompt, system_prompt, is_json)

async def _try_nvidia_fallback(prompt: str, system_prompt: str, is_json: bool):
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("❌ NVIDIA_API_KEY not found in environment")
        raise Exception("No cloud AI providers available")
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "meta/llama-3.1-70b-instruct",
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 1024
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload
            )
            if response.status_code != 200:
                raise Exception(f"NVIDIA Error: {response.text}")
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"❌ NVIDIA Connection Failed: {str(e)}")
        raise e
