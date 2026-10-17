from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt

from ._native import FlairSearcher as _NativeSearcher
from .config import IOOptions, OptionValue, SearchParams, SearcherConfig
from .errors import FlairError
from .result import BatchSearchResult, SearchResult


class Searcher:
    """Search a Flair index loaded from disk.

    Usage::

        with Searcher(SearcherConfig(dimension=128, index_type="bruteforce")) as s:
            s.open("/path/to/index")
            result = s.search(query, SearchParams(topk=10))
    """

    def __init__(self, config: SearcherConfig) -> None:
        self._config = config
        self._native = _NativeSearcher.Create(config.to_native_options())
        self._lifecycle_token: int = 0
        self._opened: bool = False
        self._closed: bool = False

    @property
    def dimension(self) -> int:
        """Return the current vector dimension (refreshed after ``open()``)."""
        return int(self._native.Dimension)

    @property
    def index_type(self) -> str:
        """Return the configured index backend type."""
        return str(self._native.IndexType)

    def open(self, path: str, io_options: Optional[IOOptions] = None) -> None:
        """Load an index snapshot from disk.

        Args:
            path: Path to the index file or directory.
            io_options: Optional IO configuration (reader type, mmap settings, etc.).

        Raises:
            FlairError: If already open or if this facade was previously closed.
        """
        if self._opened:
            raise FlairError("searcher is already open")
        if self._closed:
            raise FlairError("searcher is single-use; create a new Searcher after close()")

        if io_options is None:
            effective_io_options = IOOptions()
        elif isinstance(io_options, IOOptions):
            effective_io_options = io_options
        else:
            raise FlairError("io_options must be a flair.IOOptions instance")

        self._native.Open(path, effective_io_options.to_native_options())
        self._lifecycle_token += 1
        self._opened = True

    def close(self) -> None:
        """Close the current snapshot and end this facade's documented lifetime."""
        self._ensure_open("close")
        self._native.Close()
        self._lifecycle_token += 1
        self._opened = False
        self._closed = True

    def __enter__(self) -> Searcher:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if self._opened:
            try:
                self.close()
            except Exception:
                # Suppress close errors to avoid masking the original exception.
                if exc_type is None:
                    raise
        return False

    def info(self) -> Dict[str, Any]:
        """Return metadata about the loaded index (count, dimension, etc.).

        Raises:
            FlairError: If the searcher is not open.
        """
        self._ensure_open("info")
        return dict(self._native.Info())

    def search(
        self,
        query: npt.NDArray[np.float32],
        params: Optional[SearchParams] = None,
    ) -> SearchResult:
        """Run a nearest-neighbor search.

        Args:
            query: Query vector, shape ``(dimension,)``, dtype ``float32``.
            params: Search parameters (topk, etc.). Defaults to ``SearchParams(topk=10)``.

        Returns:
            A :class:`~flair.SearchResult` with ``ids``, ``distances``, and ``stats``.

        Raises:
            FlairError: If the searcher is not open or *params* has wrong type.
        """
        self._ensure_open("search")

        if params is None:
            effective_params = SearchParams()
        elif isinstance(params, SearchParams):
            effective_params = params
        else:
            raise FlairError("params must be a flair.SearchParams instance")

        native_result = self._native.Search(query, effective_params.to_native_options())
        return SearchResult(
            ids=native_result.ids,
            distances=native_result.distances,
            stats=dict(native_result.searchStats),
        )

    def search_batch(
        self,
        queries: npt.NDArray[np.float32],
        params: Optional[SearchParams] = None,
    ) -> BatchSearchResult:
        """Run nearest-neighbor search on a batch of queries.

        Args:
            queries: Query matrix, shape ``(nq, dimension)``, dtype ``float32``.
            params: Search parameters (topk, etc.). Defaults to ``SearchParams(topk=10)``.

        Returns:
            A :class:`~flair.BatchSearchResult` with ``ids`` and ``distances``
            arrays of shape ``(nq, topk)``.

        Raises:
            FlairError: If the searcher is not open or *params* has wrong type.
        """
        self._ensure_open("search_batch")

        if params is None:
            effective_params = SearchParams()
        elif isinstance(params, SearchParams):
            effective_params = params
        else:
            raise FlairError("params must be a flair.SearchParams instance")

        native_result = self._native.SearchBatch(queries, effective_params.to_native_options())
        return BatchSearchResult(
            ids=native_result.ids,
            distances=native_result.distances,
        )

    def _capture_lifecycle_token(self) -> int:
        self._ensure_open("capturing extension state")
        return self._lifecycle_token

    def _has_lifecycle_token(self, token: int) -> bool:
        return token == self._lifecycle_token and self._opened

    def _ensure_open(self, action: str) -> None:
        if not self._opened:
            raise FlairError("searcher must be open before {}".format(action))

    @classmethod
    def from_options(cls, options: Dict[str, OptionValue]) -> Searcher:
        """Create a Searcher from a flat options dictionary (escape hatch)."""
        return cls(SearcherConfig.from_options(options))
