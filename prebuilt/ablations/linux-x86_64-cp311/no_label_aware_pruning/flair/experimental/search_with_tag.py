from __future__ import annotations

import weakref
from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt

from .._native import SearchWithTagExtension as _NativeSearchWithTagExtension
from ..config import SearchParams
from ..errors import FlairError
from ..result import SearchResult


class SearchWithTagExtension:
    """Experimental: tag-based filtering during nearest-neighbor search.

    Usage::

        ext = SearchWithTagExtension().attach(searcher)
        result = ext.search(
            query,
            {"$and": [{"color": {"$eq": "blue"}}, {"size": {"$in": ["S", "M"]}}]},
            SearchParams(topk=10),
        )
    """

    def __init__(self) -> None:
        self._native = _NativeSearchWithTagExtension()
        self._attached_once: bool = False
        self._searcher_ref: Optional[weakref.ref[Any]] = None
        self._searcher_lifecycle_token: Optional[int] = None

    def attach(self, searcher: Any) -> SearchWithTagExtension:
        """Attach this extension to an open :class:`~flair.Searcher`.

        Must be called exactly once. After the searcher is closed or reopened,
        this wrapper is invalidated and a new one must be created.

        Args:
            searcher: An open :class:`~flair.Searcher` instance.

        Returns:
            ``self``, for chaining.

        Raises:
            FlairError: If *searcher* is not a Searcher, not open, or already attached.
        """
        from ..searcher import Searcher

        if not isinstance(searcher, Searcher):
            raise FlairError("searcher must be a flair.Searcher instance")
        searcher._ensure_open("attaching SearchWithTagExtension")

        if self._attached_once:
            raise FlairError("extension wrappers are single-use; create a new SearchWithTagExtension")

        searcher._native.Attach(self._native)
        self._attached_once = True
        self._searcher_ref = weakref.ref(searcher)
        self._searcher_lifecycle_token = searcher._capture_lifecycle_token()
        return self

    def search(
        self,
        query: npt.NDArray[np.float32],
        tag_filter: Dict[str, Any],
        params: Optional[SearchParams] = None,
    ) -> SearchResult:
        """Run a tag-filtered nearest-neighbor search.

        Args:
            query: Query vector, shape ``(dimension,)``, dtype ``float32``.
            tag_filter: Tag filter condition as a dict. Supports:

                - Logical operators: ``{"$and": [...]}`` or ``{"$or": [...]}``
                - Comparison: ``{"key": {"$eq": value}}``, ``{"key": {"$gt": value}}``, etc.
                - Set operator: ``{"key": {"$in": [values]}}``

            params: Search parameters. Defaults to ``SearchParams(topk=10)``.

        Returns:
            A :class:`~flair.SearchResult`.

        Raises:
            FlairError: If not attached, searcher closed, or filter is invalid.
        """
        searcher = self._require_attached_searcher()

        if params is None:
            effective_params = SearchParams()
        elif isinstance(params, SearchParams):
            effective_params = params
        else:
            raise FlairError("params must be a flair.SearchParams instance")

        searcher._ensure_open("tag search")

        native_result = self._native.SearchWithTag(
            query,
            tag_filter,
            effective_params.to_native_options(),
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
                "attached searcher lifecycle changed; create a new SearchWithTagExtension and attach it again"
            )

        return searcher
