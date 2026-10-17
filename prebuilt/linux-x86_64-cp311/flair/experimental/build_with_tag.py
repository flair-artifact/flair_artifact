from __future__ import annotations

import weakref
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt

from .._native import BuildWithTagExtension as _NativeBuildWithTagExtension
from ..errors import FlairError


class BuildWithTagExtension:
    """Experimental: tag-aware index building for DiskANN.

    Usage::

        ext = BuildWithTagExtension().attach(builder)
        ext.insert_batch_with_tag(vectors, ids, [
            {"tagk_name": "color", "data": ["red", "blue", "red"], "data_type": "string"},
            {"tagk_name": "size",  "data": ["S", "M", "L"], "data_type": "string"},
        ])
        builder.dump("/path/to/index")
    """

    def __init__(self) -> None:
        self._native = _NativeBuildWithTagExtension()
        self._attached_once: bool = False
        self._builder_ref: Optional[weakref.ref[Any]] = None

    def attach(self, builder: Any) -> BuildWithTagExtension:
        """Attach this extension to a :class:`~flair.Builder`.

        Must be called exactly once before inserting tag data.

        Args:
            builder: A :class:`~flair.Builder` instance.

        Returns:
            ``self``, for chaining.

        Raises:
            FlairError: If *builder* is not a Builder or already attached.
        """
        from ..builder import Builder

        if not isinstance(builder, Builder):
            raise FlairError("builder must be a flair.Builder instance")

        if self._attached_once:
            raise FlairError("extension wrappers are single-use; create a new BuildWithTagExtension")

        builder._native.Attach(self._native)
        self._attached_once = True
        self._builder_ref = weakref.ref(builder)
        return self

    def insert_batch_with_tag(
        self,
        vectors: npt.NDArray[np.float32],
        ids: npt.NDArray[np.uint64],
        tag_data: List[Dict[str, Any]],
    ) -> None:
        """Insert a batch of vectors with per-tag-dimension data.

        Args:
            vectors: Data vectors, shape ``(count, dimension)``, dtype ``float32``.
            ids: Unique IDs for each vector, shape ``(count,)``, dtype ``uint64``.
            tag_data: List of dicts, one per tag dimension. Each dict contains:

                - ``"tagk_name"``: Tag key name (str).
                - ``"data"``: List of tag values per vector. Each element can be
                  a single value (str/int/float) or a list for multi-value tags.
                - ``"data_type"``: Required. One of ``"string"``, ``"int64"``,
                  ``"double"``, ``"int32"``, ``"float"``.

        Raises:
            FlairError: If not attached or arguments are invalid.
        """
        builder = self._require_attached_builder()
        self._native.InsertBatchWithTag(vectors, ids, tag_data, builder.dimension)

    def _require_attached_builder(self) -> Any:
        if self._builder_ref is None:
            raise FlairError("extension is not attached to a builder")

        builder = self._builder_ref()
        if builder is None:
            raise FlairError("attached builder is no longer alive")

        return builder
