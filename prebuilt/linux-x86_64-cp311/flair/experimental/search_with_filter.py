from __future__ import annotations

import weakref
from typing import Any, Callable, Optional

import numpy as np
import numpy.typing as npt

from .._native import SearchWithFilterExtension as _NativeSearchWithFilterExtension
from ..config import SearchParams
from ..errors import FlairError
from ..result import SearchResult


class SearchWithFilterExtension:
    """Experimental: per-query filtering during nearest-neighbor search.

    Usage::

        filtered = SearchWithFilterExtension().attach(searcher)
        result = filtered.search(query, lambda vid: vid % 2 == 0, SearchParams(topk=5))
    """

    def __init__(self) -> None:
        self._native = _NativeSearchWithFilterExtension()
        self._attached_once: bool = False
        self._searcher_ref: Optional[weakref.ref[Any]] = None
        self._searcher_lifecycle_token: Optional[int] = None

    def attach(self, searcher: Any) -> SearchWithFilterExtension:
        """Attach this extension to an open :class:`~flair.Searcher`.

        Must be called exactly once. After the searcher is closed or reopened,
        this wrapper is invalidated and a new one must be created.

        Args:
            searcher: An open :class:`~flair.Searcher` instance.

        Returns:
            ``self``, for chaining (e.g. ``ext.attach(s).search(...)``).

        Raises:
            FlairError: If *searcher* is not a Searcher, not open, or already attached.
        """
        from ..searcher import Searcher

        if not isinstance(searcher, Searcher):
            raise FlairError("searcher must be a flair.Searcher instance")
        searcher._ensure_open("attaching SearchWithFilterExtension")

        if self._attached_once:
            raise FlairError("extension wrappers are single-use; create a new SearchWithFilterExtension")

        searcher._native.Attach(self._native)
        self._attached_once = True
        self._searcher_ref = weakref.ref(searcher)
        self._searcher_lifecycle_token = searcher._capture_lifecycle_token()
        return self

    def search(
        self,
        query: npt.NDArray[np.float32],
        filter_fn: Callable[[int], bool],
        params: Optional[SearchParams] = None,
    ) -> SearchResult:
        """Run a filtered nearest-neighbor search.

        Args:
            query: Query vector, shape ``(dimension,)``, dtype ``float32``.
            filter_fn: Predicate ``(vector_id) -> bool``. Return ``True`` to keep a candidate.
            params: Search parameters. Defaults to ``SearchParams(topk=10)``.

        Returns:
            A :class:`~flair.SearchResult`.

        Raises:
            FlairError: If not attached, searcher closed, or *filter_fn* is not callable.
        """
        if filter_fn is None or not callable(filter_fn):
            raise FlairError("filter_fn must be callable")

        searcher = self._require_attached_searcher()
        if params is None:
            effective_params = SearchParams()
        elif isinstance(params, SearchParams):
            effective_params = params
        else:
            raise FlairError("params must be a flair.SearchParams instance")

        searcher._ensure_open("filtered search")

        native_options = effective_params.to_native_options()

        native_result = self._native.SearchWithFilter(
            query,
            filter_fn,
            native_options,
            searcher.dimension,
            searcher.index_type,
        )

        return SearchResult(
            ids=native_result.ids,
            distances=native_result.distances,
            stats=dict(native_result.searchStats),
        )

    def _require_attached_searcher(self) -> Any:
        if self._searcher_ref is None:
            raise FlairError("extension is not attached to a searcher")

        searcher = self._searcher_ref()
        if searcher is None:
            raise FlairError("attached searcher is no longer alive")
        if not searcher._has_lifecycle_token(self._searcher_lifecycle_token):
            raise FlairError(
                "attached searcher lifecycle changed; create a new SearchWithFilterExtension and attach it again"
            )

        return searcher
