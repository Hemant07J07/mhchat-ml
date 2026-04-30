import json
import os
from typing import Iterator, List

import requests

from src.ml.mcp_client import search_local_docs, search_web
from src.ml.model_config import get_model_config

MODEL_CONFIG = get_model_config()
OLLAMA_URL = MODEL_CONFIG["ollama_url"]
CHAT_MODEL = MODEL_CONFIG["chat_model"]
FALLBACK_CHAT_MODEL = MODEL_CONFIG["fallback_chat_model"]
LOCAL_TOP_K = int(os.getenv("LOCAL_TOP_K", "3"))
WEB_TOP_K = int(os.getenv("WEB_TOP_K", "2"))


def _build_prompt(question: str, kb_hits: List[str], local_chunks, web_results):
    kb_context = "\n\n".join([f"[KB {i+1}] {hit}" for i, hit in enumerate(kb_hits)])

    local_context = "\n\n".join(
        [f"[LOCAL {i+1}] {x['source']}\n{x['chunk']}" for i, x in enumerate(local_chunks)]
    )

    web_context = "\n\n".join(
        [f"[WEB {i+1}] {x['title']}\n{x['snippet']}\n{x['url']}" for i, x in enumerate(web_results)]
    )

    prompt = f"""
You are a helpful assistant for a mental health chat demo.

Return VALID JSON only with this schema:
{{
  "answer": "main answer in friendly chat style",
  "summary": "short 2-3 line summary",
  "web_highlights": ["short bullet 1", "short bullet 2"],
  "used_sources": ["source names or URLs"]
}}

Rules:
- Use the KB context first for safety guidance and grounding.
- Use local context for grounded recommendations.
- Use web context for freshness if needed.
- If the answer is weak or uncertain, say that clearly.
- Do not add markdown fences.
- Write `answer` in warm, natural conversational style.
- Keep `summary` to 1-2 concise lines.
- Keep `web_highlights` short and factual.
- Include only real sources you used in `used_sources`.

Question:
{question}

KB context:
{kb_context if kb_context else "No KB context found."}

Local context:
{local_context if local_context else "No local context found."}

Web context:
{web_context if web_context else "No web results found."}
""".strip()

    return prompt


def _build_messages(prompt: str):
    return [
        {
            "role": "system",
            "content": (
                "You are MHChat RAG, a helpful assistant. "
                "Respond in natural conversational language for the `answer` field, "
                "and return valid JSON only."
            ),
        },
        {"role": "user", "content": prompt},
    ]


def _parse_chat_content(content: str):
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {
            "answer": content,
            "summary": "",
            "web_highlights": [],
            "used_sources": [],
        }

    if "used_sources" not in data:
        data["used_sources"] = []

    normalized_sources = []
    for src in data.get("used_sources", []):
        if isinstance(src, str):
            value = src.strip()
            if value:
                normalized_sources.append(value)
            continue

        if isinstance(src, dict):
            value = src.get("url") or src.get("source") or src.get("title")
            if value:
                normalized_sources.append(str(value).strip())
            continue

        if src is not None:
            normalized_sources.append(str(src).strip())

    return {
        "answer": data.get("answer", ""),
        "summary": data.get("summary", ""),
        "web_highlights": data.get("web_highlights", []),
        "sources": list(dict.fromkeys([x for x in normalized_sources if x])),
    }


def _request_chat_with_model(model_name: str, messages, prompt):
    payload = {
        "model": model_name,
        "messages": messages,
        "format": "json",
        "stream": False,
        "keep_alive": "10m",
    }

    base_url = (OLLAMA_URL or "").rstrip("/")
    attempts = []
    content = None

    try:
        r = requests.post(f"{base_url}/api/chat", json=payload, timeout=300)
        if r.ok:
            data = r.json()
            if isinstance(data, dict) and isinstance(data.get("message"), dict):
                content = data["message"].get("content", "")
        if content is None:
            attempts.append(("/api/chat", r.status_code, r.text[:300]))
    except Exception as e:
        attempts.append(("/api/chat", "error", str(e)[:300]))

    if content is None:
        try:
            v1_payload = {
                "model": model_name,
                "messages": messages,
                "stream": False,
            }
            r = requests.post(f"{base_url}/v1/chat/completions", json=v1_payload, timeout=300)
            if r.ok:
                data = r.json()
                choices = data.get("choices", []) if isinstance(data, dict) else []
                if choices and isinstance(choices[0], dict):
                    msg = choices[0].get("message", {})
                    if isinstance(msg, dict):
                        content = msg.get("content", "")
            if content is None:
                attempts.append(("/v1/chat/completions", r.status_code, r.text[:300]))
        except Exception as e:
            attempts.append(("/v1/chat/completions", "error", str(e)[:300]))

    if content is None:
        try:
            generate_payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "10m",
            }
            r = requests.post(f"{base_url}/api/generate", json=generate_payload, timeout=300)
            if r.ok:
                data = r.json()
                if isinstance(data, dict):
                    content = data.get("response", "")
            if content is None:
                attempts.append(("/api/generate", r.status_code, r.text[:300]))
        except Exception as e:
            attempts.append(("/api/generate", "error", str(e)[:300]))

    if content is None:
        detail = "\n".join([f"- {p}: {code} {msg}" for p, code, msg in attempts])
        raise RuntimeError(
            f"Failed to get chat completion for model '{model_name}'. "
            "Tried /api/chat, /v1/chat/completions, and /api/generate.\n" + detail
        )

    return content


def ask_ollama(question: str, kb_hits: List[str], local_chunks, web_results):
    prompt = _build_prompt(question, kb_hits, local_chunks, web_results)
    messages = _build_messages(prompt)

    chat_attempts = []
    content = None
    models_to_try = [CHAT_MODEL]
    if FALLBACK_CHAT_MODEL and FALLBACK_CHAT_MODEL != CHAT_MODEL:
        models_to_try.append(FALLBACK_CHAT_MODEL)

    for model_name in models_to_try:
        try:
            content = _request_chat_with_model(model_name, messages, prompt)
            break
        except Exception as exc:
            chat_attempts.append(str(exc))

    if content is None:
        raise RuntimeError("\n\n".join(chat_attempts))

    return _parse_chat_content(content)


def _stream_ollama_json(question: str, kb_hits: List[str], local_chunks, web_results) -> Iterator[str]:
    prompt = _build_prompt(question, kb_hits, local_chunks, web_results)
    messages = _build_messages(prompt)
    base_url = (OLLAMA_URL or "").rstrip("/")

    models_to_try = [CHAT_MODEL]
    if FALLBACK_CHAT_MODEL and FALLBACK_CHAT_MODEL != CHAT_MODEL:
        models_to_try.append(FALLBACK_CHAT_MODEL)

    errors = []

    for model_name in models_to_try:
        payload = {
            "model": model_name,
            "messages": messages,
            "format": "json",
            "stream": True,
            "keep_alive": "10m",
        }

        try:
            with requests.post(f"{base_url}/api/chat", json=payload, timeout=300, stream=True) as r:
                r.raise_for_status()
                saw_content = False
                for raw_line in r.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    data = json.loads(raw_line)
                    msg = data.get("message", {}) if isinstance(data, dict) else {}
                    token = msg.get("content", "") if isinstance(msg, dict) else ""
                    if token:
                        saw_content = True
                        yield token

                if saw_content:
                    return

                raise RuntimeError("No streamed content returned from /api/chat")
        except Exception as exc:
            errors.append(f"[{model_name}] {exc}")

    raise RuntimeError("\n".join(errors))


def chat(question: str, kb_hits: List[str]):
    local_chunks = search_local_docs(question, top_k=LOCAL_TOP_K)
    web_results = search_web(question, max_results=WEB_TOP_K)

    try:
        result = ask_ollama(question, kb_hits, local_chunks, web_results)
    except Exception as e:
        if kb_hits:
            result = {
                "answer": kb_hits[0],
                "summary": "",
                "web_highlights": [],
                "sources": [],
            }
        else:
            result = {
                "answer": (
                    "I could not reach the local model server right now. "
                    "Please check Ollama and model availability, then try again."
                ),
                "summary": str(e),
                "web_highlights": [],
                "sources": [],
            }

    if not result.get("sources"):
        result["sources"] = [x.get("source", "") for x in local_chunks if x.get("source")] + [
            x.get("url", "") for x in web_results if x.get("url")
        ]

    result["local_chunks"] = local_chunks
    result["web_results"] = web_results
    return result


def chat_stream(question: str, kb_hits: List[str]):
    local_chunks = search_local_docs(question, top_k=LOCAL_TOP_K)
    web_results = search_web(question, max_results=WEB_TOP_K)

    collected = []
    try:
        for token in _stream_ollama_json(question, kb_hits, local_chunks, web_results):
            collected.append(token)
            yield {"type": "token", "value": token}

        parsed = _parse_chat_content("".join(collected))
        if not parsed["sources"]:
            parsed["sources"] = [x.get("source", "") for x in local_chunks if x.get("source")] + [
                x.get("url", "") for x in web_results if x.get("url")
            ]
        yield {"type": "final", "data": parsed}
    except Exception as e:
        try:
            result = ask_ollama(question, kb_hits, local_chunks, web_results)
            if result.get("answer"):
                yield {"type": "token", "value": result["answer"]}
            if not result.get("sources"):
                result["sources"] = [x.get("source", "") for x in local_chunks if x.get("source")] + [
                    x.get("url", "") for x in web_results if x.get("url")
                ]
            yield {"type": "final", "data": result}
        except Exception:
            if kb_hits:
                fallback = {
                    "answer": kb_hits[0],
                    "summary": "",
                    "web_highlights": [],
                    "sources": [],
                }
                yield {"type": "token", "value": kb_hits[0]}
                yield {"type": "final", "data": fallback}
            else:
                fallback = {
                    "answer": (
                        "I could not reach the local model server right now. "
                        "Please check Ollama and model availability, then try again."
                    ),
                    "summary": str(e),
                    "web_highlights": [],
                    "sources": [x.get("source", "") for x in local_chunks if x.get("source")] + [
                        x.get("url", "") for x in web_results if x.get("url")
                    ],
                }
                yield {"type": "final", "data": fallback}
