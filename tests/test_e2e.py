"""End-to-end (E2E) automated tests for RAGFlow MCP Server.

Following software engineering best practices:
- Tests the complete system black-box through the Model Context Protocol (MCP) JSON-RPC stdio transport.
- Uses a real MCP client (`mcp.client.stdio.stdio_client` and `ClientSession`).
- Uses an ephemeral in-process HTTP mock server (`aiohttp.web`) simulating RAGFlow's REST API on TCP loopback.
- Validates protocol handshake, capability negotiation, tool listing, end-to-end execution,
  header propagation (Cloudflare Zero Trust), and error resilience.
- Marked with `@pytest.mark.e2e`.
"""

import asyncio
import contextlib
import json
import os
import sys
from typing import Any, AsyncGenerator, Dict, List, Optional

from aiohttp import web
import pytest
from mcp import types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


class MockRAGFlowBackend:
    """Simulates the RAGFlow REST API on an ephemeral local TCP port for hermetic E2E tests."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.host = host
        self.port = port
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self.assigned_port: int = 0
        self.received_requests: List[Dict[str, Any]] = []

        # Configurable responses and fault-injection hooks
        self.datasets: List[Dict[str, Any]] = [
            {"id": "dataset-ai-001", "name": "AI Research Papers"},
            {"id": "dataset-fin-002", "name": "Financial Reports"},
        ]
        self.documents: Dict[str, List[Dict[str, Any]]] = {
            "dataset-ai-001": [
                {
                    "id": "doc-attention-001",
                    "name": "attention_is_all_you_need_2024.pdf",
                    "update_time": "2024-05-01 10:00:00",
                }
            ],
            "dataset-fin-002": [
                {
                    "id": "doc-q4-002",
                    "name": "annual_financial_report_2024.pdf",
                    "update_time": "2024-04-15 08:30:00",
                }
            ],
        }
        self.chunks: Dict[str, List[Dict[str, Any]]] = {
            "doc-attention-001": [
                {
                    "id": "chunk-101",
                    "content": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
                    "similarity": 0.94,
                    "document_keyword": "attention_is_all_you_need_2024.pdf",
                },
                {
                    "id": "chunk-102",
                    "content": "The Transformer is the first transduction model relying entirely on self-attention to compute representations.",
                    "similarity": 0.89,
                    "document_keyword": "attention_is_all_you_need_2024.pdf",
                },
            ]
        }
        self.simulate_api_error: bool = False
        self.simulate_http_status: Optional[int] = None

        self._setup_routes()

    def _setup_routes(self) -> None:
        self.app.router.add_get("/api/v1/datasets", self._handle_list_datasets)
        self.app.router.add_post("/api/v1/retrieval", self._handle_retrieval)
        self.app.router.add_get(
            "/api/v1/datasets/{dataset_id}/documents", self._handle_list_documents
        )
        self.app.router.add_get(
            "/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks",
            self._handle_get_chunks,
        )

    async def _record_request(self, request: web.Request) -> Dict[str, Any]:
        body = None
        if request.can_read_body:
            try:
                body = await request.json()
            except Exception:
                body = await request.text()

        record = {
            "method": request.method,
            "path": request.path,
            "headers": dict(request.headers),
            "query": dict(request.query),
            "body": body,
        }
        self.received_requests.append(record)
        return record

    async def _handle_list_datasets(self, request: web.Request) -> web.Response:
        await self._record_request(request)

        if self.simulate_http_status:
            return web.Response(status=self.simulate_http_status, text="Simulated HTTP Error")

        if self.simulate_api_error:
            return web.json_response(
                {"code": 102, "message": "Simulated RAGFlow internal error"}
            )

        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return web.json_response({"code": 100, "message": "Unauthorized"}, status=401)

        return web.json_response(
            {"code": 0, "data": self.datasets, "total": len(self.datasets)}
        )

    async def _handle_retrieval(self, request: web.Request) -> web.Response:
        record = await self._record_request(request)

        if self.simulate_http_status:
            return web.Response(status=self.simulate_http_status, text="Simulated HTTP Error")

        if self.simulate_api_error:
            return web.json_response(
                {"code": 102, "message": "Simulated RAGFlow retrieval error"}
            )

        body = record["body"] or {}
        requested_ids = body.get("dataset_ids", [])

        returned_chunks = []
        for ds_id in requested_ids:
            docs = self.documents.get(ds_id, [])
            for doc in docs:
                returned_chunks.extend(self.chunks.get(doc["id"], []))

        return web.json_response(
            {
                "code": 0,
                "data": {
                    "chunks": returned_chunks,
                    "total": len(returned_chunks),
                },
            }
        )

    async def _handle_list_documents(self, request: web.Request) -> web.Response:
        await self._record_request(request)

        if self.simulate_http_status:
            return web.Response(status=self.simulate_http_status, text="Simulated HTTP Error")

        dataset_id = request.match_info["dataset_id"]
        docs = self.documents.get(dataset_id, [])
        return web.json_response(
            {
                "code": 0,
                "data": {
                    "docs": docs,
                    "total": len(docs),
                },
            }
        )

    async def _handle_get_chunks(self, request: web.Request) -> web.Response:
        await self._record_request(request)

        if self.simulate_http_status:
            return web.Response(status=self.simulate_http_status, text="Simulated HTTP Error")

        document_id = request.match_info["document_id"]
        chunks = self.chunks.get(document_id, [])
        return web.json_response(
            {
                "code": 0,
                "data": {
                    "chunks": chunks,
                    "total": len(chunks),
                },
            }
        )

    async def start(self) -> str:
        """Start the mock server on an ephemeral TCP port and return the base URL."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        self.assigned_port = self.site._server.sockets[0].getsockname()[1]
        return f"http://{self.host}:{self.assigned_port}"

    async def stop(self) -> None:
        """Stop and cleanup server resources."""
        if self.runner:
            await self.runner.cleanup()


@contextlib.asynccontextmanager
async def ephemeral_ragflow_server() -> AsyncGenerator[MockRAGFlowBackend, None]:
    """Context manager providing an ephemeral MockRAGFlowBackend instance with guaranteed cleanup."""
    server = MockRAGFlowBackend()
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


class MCPClientSessionContext:
    """Helper to launch the MCP server as a subprocess and establish an MCP ClientSession."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "valid-test-token-long-enough",
        extra_env: Optional[Dict[str, str]] = None,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.extra_env = extra_env or {}
        self._session_ctx = None
        self._stdio_ctx = None

    async def __aenter__(self) -> ClientSession:
        env = os.environ.copy()
        env["RAGFLOW_BASE_URL"] = self.base_url
        env["RAGFLOW_API_KEY"] = self.api_key
        env.update(self.extra_env)

        # Launch the MCP server as an independent subprocess using stdio transport
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "ragflow_claude_mcp"],
            env=env,
        )
        self._stdio_ctx = stdio_client(params)
        read_stream, write_stream = await self._stdio_ctx.__aenter__()

        self._session_ctx = ClientSession(read_stream, write_stream)
        session = await self._session_ctx.__aenter__()
        await session.initialize()
        return session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            if self._session_ctx:
                await self._session_ctx.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            if self._stdio_ctx:
                await self._stdio_ctx.__aexit__(exc_type, exc_val, exc_tb)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_e2e_handshake_and_tool_discovery():
    """Verify standard MCP lifecycle: handshake, capabilities negotiation, and tool discovery."""
    async with ephemeral_ragflow_server() as mock_ragflow:
        base_url = f"http://127.0.0.1:{mock_ragflow.assigned_port}"

        async with MCPClientSessionContext(base_url) as session:
            # 1. Tool discovery
            tools_response = await session.list_tools()
            tool_names = [t.name for t in tools_response.tools]

            # 2. Check all 8 expected MCP tools are registered
            expected_tools = [
                "ragflow_list_datasets",
                "ragflow_list_documents",
                "ragflow_get_chunks",
                "ragflow_list_sessions",
                "ragflow_reset_session",
                "ragflow_retrieval",
                "ragflow_retrieval_by_name",
                "ragflow_list_documents_by_name",
            ]
            for tool in expected_tools:
                assert tool in tool_names, f"Expected tool '{tool}' not found in registered tools"

            # 3. Validate tool JSON schemas
            retrieval_tool = next(
                t for t in tools_response.tools if t.name == "ragflow_retrieval_by_name"
            )
            assert "dataset_names" in retrieval_tool.inputSchema["properties"]
            assert "query" in retrieval_tool.inputSchema["properties"]
            assert set(retrieval_tool.inputSchema["required"]) == {"dataset_names", "query"}


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_e2e_list_datasets_flow():
    """Verify full end-to-end execution of `ragflow_list_datasets` tool."""
    async with ephemeral_ragflow_server() as mock_ragflow:
        base_url = f"http://127.0.0.1:{mock_ragflow.assigned_port}"
        api_key = "secure-secret-api-token"

        async with MCPClientSessionContext(base_url, api_key=api_key) as session:
            result = await session.call_tool("ragflow_list_datasets", arguments={})

            assert len(result.content) == 1
            assert result.content[0].type == "text"
            payload = json.loads(result.content[0].text)

            assert payload["code"] == 0
            assert len(payload["data"]) == 2
            assert payload["data"][0]["name"] == "AI Research Papers"

            # Verify that the server sent the Authorization header to the RAGFlow REST API
            last_req = mock_ragflow.received_requests[-1]
            assert last_req["path"] == "/api/v1/datasets"
            assert last_req["headers"]["Authorization"] == f"Bearer {api_key}"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_e2e_retrieval_by_name_flow():
    """Verify complete retrieval flow by dataset name: cache resolution, query execution, chunk extraction."""
    async with ephemeral_ragflow_server() as mock_ragflow:
        base_url = f"http://127.0.0.1:{mock_ragflow.assigned_port}"

        async with MCPClientSessionContext(base_url) as session:
            result = await session.call_tool(
                "ragflow_retrieval_by_name",
                arguments={
                    "dataset_names": ["AI Research Papers"],
                    "query": "What is the Transformer architecture?",
                    "similarity_threshold": 0.5,
                    "top_k": 50,
                },
            )

            assert len(result.content) == 1
            payload = json.loads(result.content[0].text)

            # Structure asserted:
            # - datasets_found matches input name with resolved ID
            # - retrieval_result contains chunks with scores and content
            assert "datasets_found" in payload
            assert payload["datasets_found"][0]["name"] == "AI Research Papers"
            assert payload["datasets_found"][0]["id"] == "dataset-ai-001"

            retrieval_res = payload["retrieval_result"]
            assert retrieval_res["code"] == 0
            chunks = retrieval_res["data"]["chunks"]
            assert len(chunks) == 2
            assert "Transformer" in chunks[1]["content"]
            assert chunks[0]["similarity"] == 0.94


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_e2e_direct_retrieval_flow():
    """Verify direct `ragflow_retrieval` by dataset ID and parameter forwarding."""
    async with ephemeral_ragflow_server() as mock_ragflow:
        base_url = f"http://127.0.0.1:{mock_ragflow.assigned_port}"

        async with MCPClientSessionContext(base_url) as session:
            result = await session.call_tool(
                "ragflow_retrieval",
                arguments={
                    "dataset_ids": ["dataset-ai-001"],
                    "query": "self-attention mechanism",
                    "page": 1,
                    "page_size": 5,
                },
            )

            payload = json.loads(result.content[0].text)
            assert payload["code"] == 0
            assert len(payload["data"]["chunks"]) >= 1

            # Assert parameters sent to backend HTTP endpoint
            retrieval_req = next(
                r for r in mock_ragflow.received_requests if r["path"] == "/api/v1/retrieval"
            )
            assert retrieval_req["body"]["dataset_ids"] == ["dataset-ai-001"]
            assert retrieval_req["body"]["question"] == "self-attention mechanism"
            assert retrieval_req["body"]["page_size"] == 5


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_e2e_documents_and_chunks_flow():
    """Verify document listing and chunk inspection tools."""
    async with ephemeral_ragflow_server() as mock_ragflow:
        base_url = f"http://127.0.0.1:{mock_ragflow.assigned_port}"

        async with MCPClientSessionContext(base_url) as session:
            # 1. List documents by dataset name
            doc_result = await session.call_tool(
                "ragflow_list_documents_by_name",
                arguments={"dataset_name": "AI Research Papers"},
            )
            doc_payload = json.loads(doc_result.content[0].text)
            assert doc_payload["dataset_found"]["id"] == "dataset-ai-001"
            docs = doc_payload["documents"]["data"]["docs"]
            assert len(docs) == 1
            doc_id = docs[0]["id"]
            assert doc_id == "doc-attention-001"

            # 2. Get chunks for document
            chunk_result = await session.call_tool(
                "ragflow_get_chunks",
                arguments={"dataset_id": "dataset-ai-001", "document_id": doc_id},
            )
            chunk_payload = json.loads(chunk_result.content[0].text)
            assert chunk_payload["code"] == 0
            assert len(chunk_payload["data"]["chunks"]) == 2


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_e2e_cloudflare_zero_trust_headers():
    """Verify that Cloudflare Zero Trust service token headers are transmitted across network requests."""
    async with ephemeral_ragflow_server() as mock_ragflow:
        base_url = f"http://127.0.0.1:{mock_ragflow.assigned_port}"
        cf_id = "test-cf-client-id.access"
        cf_secret = "test-cf-secret-key-12345"

        extra_env = {
            "CF_ACCESS_CLIENT_ID": cf_id,
            "CF_ACCESS_CLIENT_SECRET": cf_secret,
        }

        async with MCPClientSessionContext(base_url, extra_env=extra_env) as session:
            await session.call_tool("ragflow_list_datasets", arguments={})

            last_req = mock_ragflow.received_requests[-1]
            assert "CF-Access-Client-Id" in last_req["headers"]
            assert last_req["headers"]["CF-Access-Client-Id"] == cf_id
            assert "CF-Access-Client-Secret" in last_req["headers"]
            assert last_req["headers"]["CF-Access-Client-Secret"] == cf_secret


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_e2e_resilience_dataset_not_found():
    """Verify server resilience when a requested dataset does not exist."""
    async with ephemeral_ragflow_server() as mock_ragflow:
        base_url = f"http://127.0.0.1:{mock_ragflow.assigned_port}"

        async with MCPClientSessionContext(base_url) as session:
            result = await session.call_tool(
                "ragflow_retrieval_by_name",
                arguments={
                    "dataset_names": ["NonExistentDatasetXYZ"],
                    "query": "Find missing docs",
                },
            )

            # Server must not crash or drop connection; it must return a user-friendly error response
            assert len(result.content) == 1
            error_message = result.content[0].text
            assert "NonExistentDatasetXYZ" in error_message
            assert "Available datasets:" in error_message


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_e2e_resilience_backend_api_error():
    """Verify server error handling when RAGFlow backend returns a 500 status code."""
    async with ephemeral_ragflow_server() as mock_ragflow:
        base_url = f"http://127.0.0.1:{mock_ragflow.assigned_port}"
        mock_ragflow.simulate_http_status = 500

        async with MCPClientSessionContext(base_url) as session:
            result = await session.call_tool("ragflow_list_datasets", arguments={})

            assert len(result.content) == 1
            assert "API Error" in result.content[0].text
