from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt

from ._native import FlairStreamer as _NativeStreamer
from .config import OptionValue, SearchParams, StreamerConfig
from .errors import FlairError
from .result import SearchResult


class Streamer:
    """Real-time, in-memory vector indexing facade.

    Usage::

        with Streamer(StreamerConfig(dimension=128, index_type="bruteforce",
                                     distance_metric="l2")) as s:
            s.insert(vec, id=42)
            result = s.search(query, SearchParams(topk=10))

    ``bruteforce.streamer.capacity`` bounds the row count without reserving it: row
    blocks are allocated on demand.

    Thread-safety mirrors the C++ contract: ``search`` is safe to call
    concurrently from multiple threads; ``insert`` requires external
    single-writer serialization. Deletion through
    :class:`~flair.experimental.MutableStreamExtension` shares that same writer
    domain, so ``delete`` must be serialized with ``insert`` and with itself.
    """

    def __init__(self, config: StreamerConfig) -> None:
        self._config = config
        self._native = _NativeStreamer.Create(config.to_native_options())

    @property
    def dimension(self) -> int:
        """Return the configured vector dimension."""
        return int(self._native.Dimension)

    @property
    def index_type(self) -> str:
        """Return the configured backend index type."""
        return str(self._native.IndexType)

    def insert(self, vector: npt.NDArray[np.float32], id: int) -> None:
        """Insert a single float32 vector with the caller-assigned id.

        Ids MUST be unique across all inserts of a streamer instance. Inserting a
        duplicate id is a caller error and yields undefined behavior that depends on
        the backend implementation; deduplicating ids upstream is the caller's
        responsibility.
        """
        self._native.Insert(vector, int(id))

    def search(
        self,
        query: npt.NDArray[np.float32],
        params: Optional[SearchParams] = None,
    ) -> SearchResult:
        """Run a nearest-neighbor search on the live index."""
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

    def info(self) -> Dict[str, Any]:
        """Return live streamer metadata (count, dimension, index_type)."""
        return dict(self._native.Info())

    def __enter__(self) -> "Streamer":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        return False

    @classmethod
    def from_options(cls, options: Dict[str, OptionValue]) -> "Streamer":
        """Create a Streamer from a flat options dictionary (escape hatch)."""
        return cls(StreamerConfig.from_options(options))
