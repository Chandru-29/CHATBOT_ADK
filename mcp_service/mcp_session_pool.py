"""
mcp_session_pool.py — Async pool of persistent MCP ClientSessions.

Replaces the single shared ClientSession bottleneck with an asyncio.Queue-based pool
of N persistent subprocess sessions. Each session runs its own mcp_service/server.py
subprocess and holds one open DB connection.

Under high concurrency (100+ users):
  - Up to N queries execute simultaneously without any subprocess spawn overhead.
  - Queries beyond N queue in the asyncio.Queue and are served in < 1 ms when a
    session becomes free (no 300-500 ms subprocess startup cost).
  - Failed sessions are automatically replaced by a fresh subprocess.

Configuration:
    MCP_POOL_SIZE  (env)  — number of persistent sessions (default: 5)
"""

# ── MODULE TAG: MCP Async Session Pool ──
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from core.config.settings import MCP_POOL_SIZE
from core.config.logger import get_logger

log = get_logger(__name__)


class _PooledSession:
    """Wrapper around a ClientSession tracking context managers for clean teardown.

    Attributes:
        session (ClientSession): Active MCP ClientSession object.
        cm_stdio: Stdio client context manager.
        cm_sess: Client session context manager.
        index (int): Session index in the pool.
        healthy (bool): Health indicator flag.
    """

    def __init__(
        self,
        session: ClientSession,
        cm_stdio,
        cm_sess,
        index: int,
    ) -> None:
        """Initialize _PooledSession instance.

        Args:
            session (ClientSession): ClientSession object.
            cm_stdio: Stdio context manager.
            cm_sess: Session context manager.
            index (int): Session index integer.
        """
        self.session = session
        self.cm_stdio = cm_stdio
        self.cm_sess = cm_sess
        self.index = index
        self.healthy = True

    async def close(self) -> None:
        """Cleanly tear down the session and terminate its subprocess."""
        try:
            await self.cm_sess.__aexit__(None, None, None)
        except Exception as e:
            log.debug(f"MCPSessionPool[{self.index}]: session close warning: {e}")
        try:
            await self.cm_stdio.__aexit__(None, None, None)
        except Exception as e:
            log.debug(f"MCPSessionPool[{self.index}]: stdio close warning: {e}")
        self.healthy = False


class MCPSessionPool:
    """Async pool of N persistent MCP ClientSessions backed by asyncio.Queue.

    Attributes:
        _pool_size (int): Total number of persistent sessions in pool.
    """

    def __init__(self, pool_size: int = MCP_POOL_SIZE) -> None:
        """Initialize MCPSessionPool instance.

        Args:
            pool_size (int, optional): Configured pool capacity. Defaults to MCP_POOL_SIZE.
        """
        self._pool_size = pool_size
        self._queue: asyncio.Queue[_PooledSession] = asyncio.Queue()
        self._all_sessions: list[_PooledSession] = []
        self._server_params: Optional[StdioServerParameters] = None
        self._started = False

    async def start(self, server_params: StdioServerParameters) -> None:
        """Spawn `pool_size` MCP subprocesses and initialize pool queue.

        Args:
            server_params (StdioServerParameters): Subprocess parameters object.
        """
        if self._started:
            log.warning("MCPSessionPool: start() called more than once — ignoring.")
            return

        self._server_params = server_params
        self._started = True
        successful = 0

        for i in range(self._pool_size):
            pooled = await self._spawn_session(i)
            if pooled is not None:
                self._all_sessions.append(pooled)
                await self._queue.put(pooled)
                successful += 1

        if successful == 0:
            log.error(
                "MCPSessionPool: All session spawns failed. "
                "Requests will fall back to per-request subprocess spawning."
            )
        else:
            log.info(
                f"MCPSessionPool: Pool ready — {successful}/{self._pool_size} sessions active."
            )

    async def stop(self) -> None:
        """Cleanly shut down all sessions during application shutdown."""
        for pooled in self._all_sessions:
            await pooled.close()
        self._all_sessions.clear()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._started = False
        log.info("MCPSessionPool: All sessions closed.")

    @asynccontextmanager
    async def acquire(self, timeout: float = 4.0):
        """Acquire a healthy session from the pool asynchronously.

        Args:
            timeout (float, optional): Maximum wait time in seconds. Defaults to 4.0.

        Yields:
            ClientSession: Healthy MCP ClientSession instance.

        Raises:
            RuntimeError: If no session is acquired within the timeout window.
        """
        if not self._started or self._queue.empty() and not self._all_sessions:
            raise RuntimeError("MCPSessionPool: Pool not started or empty.")

        try:
            pooled = await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"MCPSessionPool: All {self._pool_size} sessions are busy. "
                f"Request waited {timeout}s without acquiring a slot. "
                f"Consider increasing MCP_POOL_SIZE."
            )

        if not pooled.healthy:
            log.warning(f"MCPSessionPool[{pooled.index}]: Unhealthy session — replacing.")
            pooled = await self._replace_session(pooled)

        try:
            yield pooled.session
        except Exception:
            log.warning(f"MCPSessionPool[{pooled.index}]: Exception during session use — marking for check.")
            raise
        finally:
            if pooled.healthy:
                await self._queue.put(pooled)
            else:
                asyncio.create_task(self._async_replace_and_return(pooled))

    async def _spawn_session(self, index: int) -> Optional["_PooledSession"]:
        """Spawn a single MCP subprocess and return a wrapped session.

        Args:
            index (int): Pool session index.

        Returns:
            Optional[_PooledSession]: Wrapped pooled session, or None on failure.
        """
        try:
            cm_stdio = stdio_client(self._server_params)
            read, write = await cm_stdio.__aenter__()
            cm_sess = ClientSession(read, write)
            session = await cm_sess.__aenter__()
            await session.initialize()
            await session.list_tools()
            log.info(f"MCPSessionPool[{index}]: Session spawned successfully.")
            return _PooledSession(session=session, cm_stdio=cm_stdio, cm_sess=cm_sess, index=index)
        except Exception as e:
            log.error(f"MCPSessionPool[{index}]: Failed to spawn session: {e}")
            return None

    async def _replace_session(self, old: "_PooledSession") -> "_PooledSession":
        """Tear down an unhealthy session and spawn a replacement.

        Args:
            old (_PooledSession): Unhealthy pooled session instance.

        Returns:
            _PooledSession: Fresh replacement session instance.
        """
        await old.close()
        fresh = await self._spawn_session(old.index)
        if fresh is not None:
            for i, s in enumerate(self._all_sessions):
                if s.index == old.index:
                    self._all_sessions[i] = fresh
                    break
            return fresh
        log.error(f"MCPSessionPool[{old.index}]: Replacement spawn failed. Pool has reduced capacity.")
        old.healthy = True
        return old

    async def _async_replace_and_return(self, old: "_PooledSession") -> None:
        """Asynchronously replace a dead session and return the fresh instance to pool.

        Args:
            old (_PooledSession): Unhealthy session instance to replace.
        """
        fresh = await self._replace_session(old)
        await self._queue.put(fresh)

    @property
    def size(self) -> int:
        """Return the configured pool capacity limit.

        Returns:
            int: Pool size count.
        """
        return self._pool_size

    @property
    def available(self) -> int:
        """Return the number of available sessions currently in queue.

        Returns:
            int: Available sessions count.
        """
        return self._queue.qsize()

    @property
    def is_ready(self) -> bool:
        """Check whether pool is started and contains active sessions.

        Returns:
            bool: True if pool is active and ready, False otherwise.
        """
        return self._started and len(self._all_sessions) > 0


# Process-global pool singleton
mcp_pool = MCPSessionPool()
