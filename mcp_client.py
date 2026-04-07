"""
mcp_client.py — MCP (Model Context Protocol) client for the hackathon environment.

Supports:
  - Connecting to MCP servers (SSE or stdio)
  - Listing available tools from MCP servers
  - Calling MCP tools and returning results
  - Integrating MCP tool results into the environment step
"""

from __future__ import annotations

import json
import asyncio
import threading
import requests
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str

    def to_display_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "schema": self.input_schema,
            "source": f"MCP:{self.server_name}",
        }


@dataclass
class MCPServer:
    name: str
    url: str          # SSE endpoint or stdio command
    server_type: str  # "sse" or "stdio"
    connected: bool = False
    tools: List[MCPTool] = field(default_factory=list)
    error: Optional[str] = None


class MCPClient:
    """
    Lightweight MCP client that connects to SSE-based MCP servers.
    For the hackathon, we support SSE servers (HTTP-based) since they work
    with remote Claude.ai connected servers.
    """

    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}

    def add_server(self, name: str, url: str, server_type: str = "sse") -> MCPServer:
        server = MCPServer(name=name, url=url, server_type=server_type)
        self.servers[name] = server
        return server

    def connect_server(self, name: str) -> Tuple[bool, str]:
        """Try to connect to an MCP server and fetch its tool list."""
        server = self.servers.get(name)
        if not server:
            return False, f"Server '{name}' not registered"

        if server.server_type == "sse":
            return self._connect_sse(server)
        else:
            return False, f"Server type '{server.server_type}' not yet supported in UI mode"

    def _connect_sse(self, server: MCPServer) -> Tuple[bool, str]:
        """Connect to SSE MCP server and fetch tools via JSON-RPC."""
        try:
            # Initialize connection
            init_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "hackathon-env", "version": "1.0.0"},
                },
            }
            # For SSE servers, we POST to the base URL
            resp = requests.post(
                server.url,
                json=init_payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            resp.raise_for_status()

            # List tools
            tools_payload = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {},
            }
            resp2 = requests.post(
                server.url,
                json=tools_payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            resp2.raise_for_status()
            tools_data = resp2.json()
            raw_tools = tools_data.get("result", {}).get("tools", [])

            server.tools = [
                MCPTool(
                    name=t["name"],
                    description=t.get("description", ""),
                    input_schema=t.get("inputSchema", {}),
                    server_name=server.name,
                )
                for t in raw_tools
            ]
            server.connected = True
            server.error = None
            return True, f"Connected to {server.name} — {len(server.tools)} tools available"

        except requests.exceptions.ConnectionError:
            server.error = "Connection refused — is the MCP server running?"
            return False, server.error
        except requests.exceptions.Timeout:
            server.error = "Connection timed out"
            return False, server.error
        except Exception as e:
            server.error = str(e)
            return False, f"Error: {e}"

    def call_tool(self, server_name: str, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool and return the result."""
        server = self.servers.get(server_name)
        if not server or not server.connected:
            return {"error": f"Server '{server_name}' not connected"}

        payload = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": args,
            },
        }
        try:
            resp = requests.post(
                server.url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
            if "error" in result:
                return {"error": result["error"]}
            content = result.get("result", {}).get("content", [])
            # Flatten text content
            text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
            return {"result": "\n".join(text_parts), "raw": result}
        except Exception as e:
            return {"error": str(e)}

    def get_all_tools(self) -> List[MCPTool]:
        """Return all tools from all connected servers."""
        all_tools = []
        for server in self.servers.values():
            if server.connected:
                all_tools.extend(server.tools)
        return all_tools

    def get_tool_names(self) -> List[str]:
        return [t.name for t in self.get_all_tools()]

    def disconnect_server(self, name: str):
        server = self.servers.get(name)
        if server:
            server.connected = False
            server.tools = []

    def remove_server(self, name: str):
        self.servers.pop(name, None)

    def status_summary(self) -> str:
        lines = []
        for name, srv in self.servers.items():
            status = "✅ Connected" if srv.connected else f"❌ {srv.error or 'Not connected'}"
            tool_count = len(srv.tools) if srv.connected else 0
            lines.append(f"  {name} ({srv.url}) — {status} — {tool_count} tools")
        return "\n".join(lines) if lines else "No MCP servers registered"


# Global singleton
_mcp_client = MCPClient()

def get_mcp_client() -> MCPClient:
    return _mcp_client