import hashlib
import json
import os
from typing import Iterator, List, Optional

import requests

import re

from src.ml.mcp_client import search_local_docs, search_web
from src.ml.model_config import get_model_config

MODEL_CONFIG = get_model_config()
OLLAMA_URL = MODEL_CONFIG["ollama_url"]
CHAT_MODEL = MODEL_CONFIG["chat_model"]
FALLBACK_CHAT_MODEL = MODEL_CONFIG["fallback_chat_model"]
LOCAL_TOP_K = int(os.getenv("LOCAL_TOP_K", "3"))
WEB_TOP_K = int(os.getenv("WEB_TOP_K", "2"))
CHAT_TIMEOUT_S = int(os.getenv("OLLAMA_CHAT_TIMEOUT", "10"))

STYLE_HINTS = [
    "Keep the tone warm and encouraging.",
    "Use a calm, grounding tone with gentle questions.",
    "Keep the response concise and practical.",
    "Use a supportive tone with a small actionable step.",
    "Use empathetic wording and avoid repeating the same opening.",
]

WEB_QUERY_RE = re.compile(
    r"\b(web|internet|online|duckduckgo|search|browse|site|source|sources|reference|references|cite|citations|link|links|latest|recent|today|news|update|updated|current)\b",
    re.IGNORECASE,
)


def _should_use_web_search(question: str) -> bool:
    if not question:
        return False
    # Always allow web search for general knowledge queries.
    return True


def _build_embedding_fallback(question: str, kb_hits, local_chunks, web_results, web_used: bool):
    if kb_hits:
        answer = kb_hits[0]
        summary = ""
        sources = []
    elif local_chunks:
        top = local_chunks[0]
        answer = f"Based on the closest local match, here is the most relevant excerpt:\n\n{top.get('chunk', '')}"
        summary = ""
        sources = [top.get("source", "")] if top.get("source") else []
    elif web_used and web_results:
        top = web_results[0]
        answer = (
            "I could not reach the local model, so I am sharing the top web match:\n\n"
            f"{top.get('title', '')}\n{top.get('snippet', '')}\n{top.get('url', '')}"
        )
        summary = ""
        sources = [top.get("url", "")] if top.get("url") else []
    else:
        answer = (
            "I could not reach the local model right now, and I do not have a strong match to return. "
            "Please try again in a moment."
        )
        summary = ""
        sources = []

    return {
        "answer": answer,
        "summary": summary,
        "web_highlights": [x.get("snippet", "") for x in (web_results or [])[:2] if x.get("snippet")],
        "sources": [s for s in sources if s],
    }


def _build_prompt(
    question: str,
    kb_hits: List[str],
    local_chunks,
    web_results,
    fallback_hint: Optional[str] = None,
):
    kb_context = "\n\n".join([f"[KB {i+1}] {hit}" for i, hit in enumerate(kb_hits)])

    local_context = "\n\n".join(
        [f"[LOCAL {i+1}] {x['source']}\n{x['chunk']}" for i, x in enumerate(local_chunks)]
    )

    web_context = "\n\n".join(
        [f"[WEB {i+1}] {x['title']}\n{x['snippet']}\n{x['url']}" for i, x in enumerate(web_results)]
    )

    style_hint = ""
    if question:
        style_hint = STYLE_HINTS[int(hashlib.md5(question.encode("utf-8")).hexdigest(), 16) % len(STYLE_HINTS)]

        prompt = f"""
You are a helpful assistant for a mental health chat demo.

Return VALID JSON only with this schema:
{{
    "answer": "natural, complete reply in friendly chat style",
    "summary": "short 1-2 line summary",
    "web_highlights": ["short factual bullet 1", "short factual bullet 2"],
    "used_sources": ["source names or URLs"]
}}

Rules:
- Write a natural, realistic answer like ChatGPT would.
- If local and web context exist, synthesize them into one coherent answer.
- Use KB context only when it clearly applies (safety and grounding).
- Avoid generic filler or therapy prompts unless the user asked about feelings.
- Use the fallback guidance to keep continuity with the previous flow.
- {style_hint}
- If the answer is weak or uncertain, say that clearly.
- Do not add markdown fences.
- Keep `summary` concise (1-2 lines).
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

Fallback guidance:
{fallback_hint if fallback_hint else "No fallback guidance provided."}
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


def _request_chat_with_model(model_name: str, messages, prompt, timeout_s: int):
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
        r = requests.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=(timeout_s, timeout_s),
        )
        if r.ok:
            data = r.json()
            if isinstance(data, dict) and isinstance(data.get("message"), dict):
                content = data["message"].get("content", "")
        if not content:
            attempts.append(("/api/chat", r.status_code, "empty response"))
    except Exception as e:
        attempts.append(("/api/chat", "error", str(e)[:300]))

    if content is None:
        try:
            v1_payload = {
                "model": model_name,
                "messages": messages,
                "stream": False,
            }
            r = requests.post(
                f"{base_url}/v1/chat/completions",
                json=v1_payload,
                timeout=(timeout_s, timeout_s),
            )
            if r.ok:
                data = r.json()
                choices = data.get("choices", []) if isinstance(data, dict) else []
                if choices and isinstance(choices[0], dict):
                    msg = choices[0].get("message", {})
                    if isinstance(msg, dict):
                        content = msg.get("content", "")
            if not content:
                attempts.append(("/v1/chat/completions", r.status_code, "empty response"))
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
            r = requests.post(
                f"{base_url}/api/generate",
                json=generate_payload,
                timeout=(timeout_s, timeout_s),
            )
            if r.ok:
                data = r.json()
                if isinstance(data, dict):
                    content = data.get("response", "")
            if not content:
                attempts.append(("/api/generate", r.status_code, "empty response"))
        except Exception as e:
            attempts.append(("/api/generate", "error", str(e)[:300]))

    if not content:
        detail = "\n".join([f"- {p}: {code} {msg}" for p, code, msg in attempts])
        raise RuntimeError(
            f"Failed to get chat completion for model '{model_name}'. "
            "Tried /api/chat, /v1/chat/completions, and /api/generate.\n" + detail
        )

    return content


def ask_ollama(
    question: str,
    kb_hits: List[str],
    local_chunks,
    web_results,
    fallback_hint: Optional[str] = None,
):
    prompt = _build_prompt(question, kb_hits, local_chunks, web_results, fallback_hint)
    messages = _build_messages(prompt)

    chat_attempts = []
    models_to_try = [CHAT_MODEL]
    if FALLBACK_CHAT_MODEL and FALLBACK_CHAT_MODEL != CHAT_MODEL:
        models_to_try.append(FALLBACK_CHAT_MODEL)

    for model_name in models_to_try:
        try:
            content = _request_chat_with_model(model_name, messages, prompt, CHAT_TIMEOUT_S)
            parsed = _parse_chat_content(content)
            if not str(parsed.get("answer", "")).strip():
                raise RuntimeError("Empty answer returned by model")
            return parsed
        except Exception as exc:
            chat_attempts.append(str(exc))

    raise RuntimeError("\n\n".join(chat_attempts))


def _stream_ollama_json(
    question: str,
    kb_hits: List[str],
    local_chunks,
    web_results,
    fallback_hint: Optional[str] = None,
) -> Iterator[str]:
    prompt = _build_prompt(question, kb_hits, local_chunks, web_results, fallback_hint)
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
            with requests.post(
                f"{base_url}/api/chat",
                json=payload,
                timeout=(CHAT_TIMEOUT_S, CHAT_TIMEOUT_S),
                stream=True,
            ) as r:
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


def chat(question: str, kb_hits: List[str], fallback_hint: Optional[str] = None):
    local_chunks = search_local_docs(question, top_k=LOCAL_TOP_K)
    web_used = _should_use_web_search(question)
    web_results = search_web(question, max_results=WEB_TOP_K) if web_used else []

    try:
        result = ask_ollama(question, kb_hits, local_chunks, web_results, fallback_hint)
    except Exception as e:
        if fallback_hint:
            result = {
                "answer": fallback_hint,
                "summary": "",
                "web_highlights": [],
                "sources": [],
            }
        else:
            result = _build_embedding_fallback(question, kb_hits, local_chunks, web_results, web_used)
            result["summary"] = str(e)

    if not result.get("sources"):
        result["sources"] = [x.get("source", "") for x in local_chunks if x.get("source")] + [
            x.get("url", "") for x in web_results if x.get("url")
        ]

    result["local_chunks"] = local_chunks
    result["web_results"] = web_results
    return result


def chat_stream(question: str, kb_hits: List[str], fallback_hint: Optional[str] = None):
    local_chunks = search_local_docs(question, top_k=LOCAL_TOP_K)
    web_used = _should_use_web_search(question)
    web_results = search_web(question, max_results=WEB_TOP_K) if web_used else []

    collected = []
    try:
        for token in _stream_ollama_json(question, kb_hits, local_chunks, web_results, fallback_hint):
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
            result = ask_ollama(question, kb_hits, local_chunks, web_results, fallback_hint)
            if result.get("answer"):
                yield {"type": "token", "value": result["answer"]}
            if not result.get("sources"):
                result["sources"] = [x.get("source", "") for x in local_chunks if x.get("source")] + [
                    x.get("url", "") for x in web_results if x.get("url")
                ]
            yield {"type": "final", "data": result}
        except Exception:
            if fallback_hint:
                fallback = {
                    "answer": fallback_hint,
                    "summary": "",
                    "web_highlights": [],
                    "sources": [],
                }
                yield {"type": "token", "value": fallback_hint}
                yield {"type": "final", "data": fallback}
            else:
                fallback = _build_embedding_fallback(question, kb_hits, local_chunks, web_results, web_used)
                fallback["summary"] = str(e)
                yield {"type": "token", "value": fallback["answer"]}
                yield {"type": "final", "data": fallback}
