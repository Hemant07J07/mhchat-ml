import asyncio
import concurrent.futures
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.ml.rag_retrieval import search_local_docs as local_search_fallback
from src.ml.rag_retrieval import search_web as web_search_fallback

MCP_ENABLED = os.getenv("MCP_ENABLED", "false").lower() == "true"
MCP_SERVER_PATH = os.getenv("MCP_SERVER_PATH", "mcp_server.py")


def _resolve_server_path() -> Path:
    candidate = Path(MCP_SERVER_PATH)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    base_dir = Path(__file__).resolve().parent
    in_base = base_dir / candidate
    if in_base.exists():
        return in_base

    in_parent = base_dir.parent / candidate
    if in_parent.exists():
        return in_parent

    return in_base


def _extract_tool_result(result):
    content = getattr(result, "content", None)
    if not content:
        return []

    for item in content:
        text = getattr(item, "text", None)
        if not text:
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return []

    return []


async def _call_tool_async(tool_name: str, args: dict):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_path = _resolve_server_path()
    params = StdioServerParameters(command=sys.executable, args=[str(server_path)])

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, args)
            return _extract_tool_result(result)


def _call_tool(tool_name: str, args: dict):
    coro = _call_tool_async(tool_name, args)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lambda: asyncio.run(coro))
        return future.result()


def search_local_docs(query: str, top_k: int = 4):
    if not MCP_ENABLED:
        return local_search_fallback(query, top_k)

    try:
        result = _call_tool("search_local_docs", {"query": query, "top_k": top_k})
        return result if isinstance(result, list) else []
    except Exception:
        return local_search_fallback(query, top_k)


def search_web(query: str, max_results: int = 3):
    if not MCP_ENABLED:
        return web_search_fallback(query, max_results)

    try:
        result = _call_tool("search_web", {"query": query, "max_results": max_results})
        return result if isinstance(result, list) else []
    except Exception:
        return web_search_fallback(query, max_results)


def check_mcp_connection():
    if not MCP_ENABLED:
        return {"enabled": False, "ok": True, "message": "MCP disabled; using local fallback."}

    try:
        result = _call_tool("search_web", {"query": "hello", "max_results": 1})
        return {
            "enabled": True,
            "ok": True,
            "message": "MCP connected.",
            "sample_result_count": len(result) if isinstance(result, list) else 0,
        }
    except Exception as e:
        return {
            "enabled": True,
            "ok": False,
            "message": f"MCP unavailable, fallback will be used: {e}",
        }
