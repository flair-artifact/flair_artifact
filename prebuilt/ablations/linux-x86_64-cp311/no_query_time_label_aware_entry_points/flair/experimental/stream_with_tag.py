from __future__ import annotations

import weakref
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import numpy.typing as npt

from .._native import StreamWithTagExtension as _NativeStreamWithTagExtension
from ..config import SearchParams
from ..errors import FlairError
from ..result import SearchResult


class StreamWithTagExtension:
    """Experimental: tag-aware insert and tag-filtered search on a live Streamer.

    Usage::

        ext = StreamWithTagExtension().attach(streamer)
        ext.insert_with_tag(vec, 42, [
            {"tagk_name": "color", "data": "blue", "data_type": "string"},
        ])
        result = ext.search(query, {"color": {"$eq": "blue"}}, SearchParams(topk=10))

    The streamer must be created with ``extension.build.tag.tag_schema`` in its
    options; without a tag schema the backend rejects :meth:`attach`.
    """

    def __init__(self) -> None:
        self._native = _NativeStreamWithTagExtension()
        self._attached_once: bool = False
        self._streamer_ref: Optional[weakref.ref[Any]] = None

    def attach(self, streamer: Any) -> StreamWithTagExtension:
        """Attach this extension to a :class:`~flair.Streamer`.

        Must be called exactly once on this wrapper. Unlike
        :class:`MutableStreamExtension`, this one may be attached after inserts have
        already happened; rows written before it carry no tags.

        Args:
            streamer: A :class:`~flair.Streamer` instance.

        Returns:
            ``self``, for chaining.

        Raises:
            FlairError: If *streamer* is not a Streamer, this wrapper was already
                attached, or the backend has no tag schema.
        """
        from ..streamer import Streamer

        if not isinstance(streamer, Streamer):
            raise FlairError("streamer must be a flair.Streamer instance")
        if self._attached_once:
            raise FlairError("extension wrappers are single-use; create a new StreamWithTagExtension")

        streamer._native.Attach(self._native)
        self._attached_once = True
        self._streamer_ref = weakref.ref(streamer)
        return self

    def insert_with_tag(
        self,
        vector: npt.NDArray[np.float32],
        id: int,
        tag_data: List[Dict[str, Any]],
    ) -> None:
        """Insert a single vector together with its tags.

        Shares the streamer's single-writer domain, so calls must be serialized with
        each other, with ``Streamer.insert``, and with ``MutableStreamExtension.delete``.

        Args:
            vector: Data vector, shape ``(dimension,)``, dtype ``float32``.
            id: Caller-assigned unique id, as in :meth:`Streamer.insert`.
            tag_data: One dict per tag dimension this row carries. Each dict contains:

                - ``"tagk_name"``: Tag key name (str), matching the tag schema.
                - ``"data"``: A single value, or a list for a multi-value dimension.
                - ``"data_type"``: Required. One of ``"string"``, ``"int64"``,
                  ``"double"``, ``"int32"``, ``"float"``. Must match what the schema
                  declares for that dimension; the native side compares the two for
                  exact equality and rejects a mismatch rather than converting. Filter
                  values in :meth:`search` are checked the same way, so an ``int32`` or
                  ``float`` dimension needs ``np.int32(...)`` / ``np.float32(...)``
                  there -- a plain Python number reads as ``int64`` / ``double``.

        Raises:
            FlairError: If not attached, the streamer is gone, or the tags do not
                match the schema.

        Note:
            Any failed insert ends the streamer's write life: the instance is no longer
            writable and only queries are supported. Retrying the insert -- corrected or
            not -- is outside the contract; build a new streamer to keep writing.
        """
        # Pin a strong reference across the native call: InsertWithTag releases the GIL,
        # so without it another thread could drop the last Streamer reference and destroy
        # the backend mid-call.
        streamer = self._require_attached_streamer()
        self._native.InsertWithTag(vector, int(id), tag_data, streamer.dimension)
        del streamer

    def search(
        self,
        query: npt.NDArray[np.float32],
        tag_filter: Dict[str, Any],
        params: Optional[SearchParams] = None,
    ) -> SearchResult:
        """Run a tag-filtered nearest-neighbor search on the live index.

        Args:
            query: Query vector, shape ``(dimension,)``, dtype ``float32``.
            tag_filter: Tag filter condition as a dict. Supports:

                - Logical operators: ``{"$and": [...]}`` or ``{"$or": [...]}``
                - Comparison: ``{"key": {"$eq": value}}``, ``{"key": {"$gte": value}}``, etc.
                - Set operator: ``{"key": {"$in": [values]}}``

            params: Search parameters. Defaults to ``SearchParams(topk=10)``.

        Returns:
            A :class:`~flair.SearchResult`.

        Raises:
            FlairError: If not attached, the streamer is gone, or the filter is invalid.
        """
        streamer = self._require_attached_streamer()
        effective_params = self._require_params(params)

        native_result = self._native.SearchWithTag(
            query,
            tag_filter,
            effective_params.to_native_options(),
            streamer.dimension,
            streamer.index_type,
        )

        return SearchResult(
            ids=native_result.ids,
            distances=native_result.distances,
            stats=dict(native_result.searchStats),
        )

    def search_with_tag_and_filter(
        self,
        query: npt.NDArray[np.float32],
        tag_filter: Dict[str, Any],
        filter_fn: Callable[[int], bool],
        params: Optional[SearchParams] = None,
    ) -> SearchResult:
        """Run a tag-filtered search intersected (AND) with a caller predicate.

        The tag filter and *filter_fn* are combined with AND semantics: a candidate must both
        match ``tag_filter`` and be kept by *filter_fn*. Use this when a predicate cannot be
        expressed as tags (e.g. an external allow-list).

        Args:
            query: Query vector, shape ``(dimension,)``, dtype ``float32``.
            tag_filter: Tag filter condition as a dict, same shape as :meth:`search`.
            filter_fn: Predicate ``(vector_id) -> bool``. Return ``True`` to keep a candidate.
                Called from the native search thread; it must not mutate the streamer.
            params: Search parameters. Defaults to ``SearchParams(topk=10)``.

        Returns:
            A :class:`~flair.SearchResult`.

        Raises:
            FlairError: If not attached, the streamer is gone, *filter_fn* is not callable,
                or the filter is invalid.
        """
        streamer = self._require_attached_streamer()
        effective_params = self._require_params(params)

        if filter_fn is None or not callable(filter_fn):
            raise FlairError("filter_fn must be callable")

        native_result = self._native.SearchWithTagAndFilter(
            query,
            tag_filter,
            filter_fn,
            effective_params.to_native_options(),
            streamer.dimension,
            streamer.index_type,
        )

        return SearchResult(
            ids=native_result.ids,
            distances=native_result.distances,
            stats=dict(native_result.searchStats),
        )

    @staticmethod
    def _require_params(params: Optional[SearchParams]) -> SearchParams:
        if params is None:
            return SearchParams()
        if isinstance(params, SearchParams):
            return params
        raise FlairError("params must be a flair.SearchParams instance")

    def _require_attached_streamer(self) -> Any:
        if self._streamer_ref is None:
            raise FlairError("extension is not attached to a streamer")

        streamer = self._streamer_ref()
        if streamer is None:
            raise FlairError("attached streamer is no longer alive")

        return streamer
