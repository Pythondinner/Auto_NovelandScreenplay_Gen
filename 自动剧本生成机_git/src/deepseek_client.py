"""
共享的 DeepSeek API 调用封装，brain / writer 两层都复用这一份，
统一处理请求、超时、错误和可选的 JSON 解析。
"""

import json
import os

import requests

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"


class ApiCallError(Exception):
    """模型调用失败（网络/超时/限流/返回内容不是预期格式等）。

    调用方应该捕获这个异常并提示用户重试，而不是让整个进程崩掉——
    调用失败时不会有任何副作用发生，上层的 snapshot/草稿都不会被写脏。
    """


def call(
    system_prompt: str,
    user_content: str,
    temperature: float = 0.7,
    json_mode: bool = False,
    max_tokens: int = None,
    return_finish_reason: bool = False,
):
    """调用一次模型。json_mode=True 时返回已解析的 dict，否则返回原始文本。
    max_tokens 不传就用 API 默认值。
    return_finish_reason=True 时改为返回 (content_or_dict, finish_reason) 元组，
    finish_reason == "length" 说明是被 max_tokens 截断的，不是模型自己写完的。"""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("未找到 DEEPSEEK_API_KEY，请检查 .env 文件")

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    try:
        resp = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ApiCallError(f"请求 DeepSeek API 失败：{e}") from e

    try:
        choice = resp.json()["choices"][0]
        content = choice["message"]["content"]
        finish_reason = choice.get("finish_reason")
    except (KeyError, IndexError) as e:
        raise ApiCallError(f"API 返回结构异常，不像预期的响应格式：{e}") from e

    if json_mode:
        try:
            content = json.loads(content)
        except json.JSONDecodeError as e:
            raise ApiCallError(f"模型返回的内容不是合法 JSON：{e}") from e

    return (content, finish_reason) if return_finish_reason else content
