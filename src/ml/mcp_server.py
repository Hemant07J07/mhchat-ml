from mcp.server.fastmcp import FastMCP

from src.ml.rag_retrieval import search_local_docs as shared_search_local_docs
from src.ml.rag_retrieval import search_web as shared_search_web

mcp = FastMCP("MHChatTools")


@mcp.tool()
def search_local_docs(query: str, top_k: int = 4):
    return shared_search_local_docs(query, top_k)


@mcp.tool()
def search_web(query: str, max_results: int = 3):
    return shared_search_web(query, max_results)


if __name__ == "__main__":
    mcp.run(transport="stdio")
