from __future__ import annotations

from typing import Any, Dict, Iterator, Tuple

import numpy as np
import numpy.typing as npt


class SearchResult:
    """Container for search results returned by :meth:`Searcher.search`."""

    ids: npt.NDArray[np.uint64]
    """Vector IDs of the nearest neighbors."""

    distances: npt.NDArray[np.float32]
    """Distances to the nearest neighbors, in the same order as *ids*."""

    stats: Dict[str, str]
    """Backend-provided search statistics."""

    def __init__(
        self,
        ids: npt.NDArray[np.uint64],
        distances: npt.NDArray[np.float32],
        stats: Dict[str, str],
    ) -> None:
        self.ids = ids
        self.distances = distances
        self.stats = stats

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self) -> Iterator[Tuple[int, float]]:
        for i in range(len(self.ids)):
            yield (int(self.ids[i]), float(self.distances[i]))

    def __repr__(self) -> str:
        return "SearchResult(ids={!r}, distances={!r}, stats={!r})".format(
            self.ids,
            self.distances,
            self.stats,
        )


class BatchSearchResult:
    """Container for batch search results returned by :meth:`Searcher.search_batch`."""

    ids: npt.NDArray[np.uint64]
    """Vector IDs, shape ``(nq, topk)``."""

    distances: npt.NDArray[np.float32]
    """Distances, shape ``(nq, topk)``."""

    def __init__(
        self,
        ids: npt.NDArray[np.uint64],
        distances: npt.NDArray[np.float32],
    ) -> None:
        self.ids = ids
        self.distances = distances

    @property
    def nq(self) -> int:
        return self.ids.shape[0]

    @property
    def topk(self) -> int:
        return self.ids.shape[1]

    def __repr__(self) -> str:
        return "BatchSearchResult(nq={}, topk={})".format(self.nq, self.topk)
