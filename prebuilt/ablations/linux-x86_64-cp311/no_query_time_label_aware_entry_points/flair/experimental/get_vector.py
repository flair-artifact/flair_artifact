from __future__ import annotations

import weakref
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from .._native import GetVectorExtension as _NativeGetVectorExtension
from .._native import StreamGetVectorExtension as _NativeStreamGetVectorExtension
from ..errors import FlairError

Batch = Tuple[npt.NDArray[np.float32], npt.NDArray[np.uint64]]
TagBatch = Tuple[npt.NDArray[np.float32], npt.NDArray[np.uint64], List[Dict[str, Any]]]


class _ExportReader:
    """Lifetime plumbing shared by the export cursors.

    A native cursor holds a read hook into the backend that produced it, so every call re-checks
    that the owning facade is still usable and pins a strong reference to it across the native
    call. ``next_batch`` releases the GIL inside, and without the pin another thread could drop
    the last owner reference and destroy the backend mid-export.

    Args:
        extension: The attached extension wrapper that created this cursor.
        dimension: Vector dimension used to shape the exported batch.
    """

    def __init__(self, extension: Any, dimension: int) -> None:
        self._extension = extension
        self._dimension = int(dimension)

    def _pin_owner(self) -> Any:
        return self._extension._require_attached_owner()


class VectorBatchReader(_ExportReader):
    """Forward-only cursor over exported vectors and ids.

    Returned by :meth:`GetVectorExtension.create_reader` and
    :meth:`StreamGetVectorExtension.create_reader`. Cursors are independent and read-only, so
    several may iterate at once.
    """

    def __init__(self, native_reader: Any, extension: Any, dimension: int) -> None:
        super().__init__(extension, dimension)
        self._native = native_reader

    def next_batch(self) -> Optional[Batch]:
        """Fetch the next batch of rows.

        The batch size is decided by the backend; consume rows based on what is returned rather
        than assuming a fixed size. Rows come back in index row order for a searcher. A streamer
        returns active rows in storage order: ``bruteforce`` matches insertion order, while
        ``flatnsw`` uses node slot order and can differ after slot reuse.

        Returns:
            ``(vectors, ids)`` where ``vectors`` has shape ``(rows, dimension)`` and dtype
            ``float32``, and ``ids`` has shape ``(rows,)`` and dtype ``uint64``. ``None`` once
            the cursor is exhausted; a searcher cursor keeps returning ``None`` after that.

        Raises:
            FlairError: If the owner is gone, a searcher was closed or reopened, or the backend
                cannot serve original vectors.
        """
        owner = self._pin_owner()
        batch = self._native.NextBatch(self._dimension)
        del owner
        return batch


class IdBatchReader(_ExportReader):
    """Forward-only cursor over exported ids, without touching vector data.

    Returned by :meth:`GetVectorExtension.create_id_reader`. On a disk backend the ids are already
    resident from ``open()``, so a full pass performs no vector IO.
    """

    def __init__(self, native_reader: Any, extension: Any, dimension: int) -> None:
        super().__init__(extension, dimension)
        self._native = native_reader

    def next_batch(self) -> Optional[npt.NDArray[np.uint64]]:
        """Fetch the next batch of ids.

        Returns:
            An ``uint64`` ndarray of shape ``(rows,)``, or ``None`` once the cursor is exhausted.

        Raises:
            FlairError: If the owner is gone, a searcher was closed or reopened, or the backend
                does not support id export.
        """
        owner = self._pin_owner()
        ids = self._native.NextBatch()
        del owner
        return ids


class TagBatchReader(_ExportReader):
    """Forward-only cursor over exported vectors, ids, and raw tag values.

    Returned by :meth:`StreamGetVectorExtension.create_reader_with_tag`.
    """

    def __init__(self, native_reader: Any, extension: Any, dimension: int) -> None:
        super().__init__(extension, dimension)
        self._native = native_reader

    def next_batch(self) -> Optional[TagBatch]:
        """Fetch the next batch of rows together with their original tag values.

        Returns:
            ``(vectors, ids, tag_data)`` where ``vectors`` and ``ids`` match
            :meth:`VectorBatchReader.next_batch`, and ``tag_data`` is one dict per tag dimension
            in schema compile order (enum dimensions first, then range). Each dict contains:

                - ``"tagk_name"``: Tag key name (str).
                - ``"data"``: One list of values per row, so a multi-value dimension yields
                  several entries for that row and a single-value dimension yields one. Rows
                  inserted without tags yield an empty list.
                - ``"data_type"``: One of ``"string"``, ``"int64"``, ``"double"``, ``"int32"``,
                  ``"float"``.

            ``None`` once the cursor is exhausted. The shape matches what
            ``insert_batch_with_tag`` accepts, so an exported batch can be written back without
            re-typing. Raw values are exported verbatim: value order, duplicates, empty strings,
            and NaN survive exactly as inserted.

        Raises:
            FlairError: If the owner is gone, or the streamer carries no tag schema.
        """
        owner = self._pin_owner()
        batch = self._native.NextBatch(self._dimension)
        del owner
        return batch


class GetVectorExtension:
    """Experimental: export raw vectors and ids from an open Searcher.

    Usage::

        ext = GetVectorExtension().attach(searcher)
        reader = ext.create_reader()
        while True:
            batch = reader.next_batch()
            if batch is None:
                break
            vectors, ids = batch

    Only exact original values are exported: ``bruteforce`` and ``flatnsw`` need ``rawf32``
    encoding and ``diskann`` needs a ``rawf32`` layout, i.e. built without disk encoding. A
    quantized or disk-encoded index still attaches, but :meth:`create_reader` then raises
    ``FlairError``; :meth:`create_id_reader` works for any encoding.

    Discard readers before closing the searcher.
    """

    def __init__(self) -> None:
        self._native = _NativeGetVectorExtension()
        self._attached_once: bool = False
        self._searcher_ref: Optional[Any] = None
        self._searcher_lifecycle_token: Optional[int] = None

    def attach(self, searcher: Any) -> GetVectorExtension:
        """Attach this extension to an open :class:`~flair.Searcher`.

        Must be called exactly once on this wrapper. After the searcher is closed or reopened,
        this wrapper and every reader it produced are invalidated.

        Args:
            searcher: An open :class:`~flair.Searcher` instance.

        Returns:
            ``self``, for chaining.

        Raises:
            FlairError: If *searcher* is not a Searcher, is not open, was already attached to
                this wrapper, or the backend does not implement this extension.
        """
        from ..searcher import Searcher

        if not isinstance(searcher, Searcher):
            raise FlairError("searcher must be a flair.Searcher instance")
        searcher._ensure_open("attaching GetVectorExtension")

        if self._attached_once:
            raise FlairError("extension wrappers are single-use; create a new GetVectorExtension")

        searcher._native.Attach(self._native)
        self._attached_once = True
        self._searcher_ref = weakref.ref(searcher)
        self._searcher_lifecycle_token = searcher._capture_lifecycle_token()
        return self

    def create_reader(self) -> VectorBatchReader:
        """Create a cursor over the index vectors and ids.

        Returns:
            A :class:`VectorBatchReader` starting at row 0.

        Raises:
            FlairError: If not attached, the searcher is gone, or the index cannot serve
                original vectors because it is quantized or disk-encoded.
        """
        searcher = self._require_attached_owner()
        return VectorBatchReader(self._native.CreateReader(), self, searcher.dimension)

    def create_id_reader(self) -> IdBatchReader:
        """Create a cursor over the index ids only.

        Available for any encoding, including quantized and disk-encoded indexes.

        Returns:
            An :class:`IdBatchReader` starting at row 0.

        Raises:
            FlairError: If not attached, the searcher is gone, or the backend does not support
                id export.
        """
        searcher = self._require_attached_owner()
        return IdBatchReader(self._native.CreateIdReader(), self, searcher.dimension)

    def _require_attached_owner(self) -> Any:
        if self._searcher_ref is None:
            raise FlairError("extension is not attached to a searcher")

        searcher = self._searcher_ref()
        if searcher is None:
            raise FlairError("attached searcher is no longer alive")
        if not searcher._has_lifecycle_token(self._searcher_lifecycle_token):
            raise FlairError(
                "attached searcher lifecycle changed; create a new GetVectorExtension and attach it again"
            )

        return searcher


class StreamGetVectorExtension:
    """Experimental: export raw vectors, ids, and tags from a live Streamer.

    Usage::

        ext = StreamGetVectorExtension().attach(streamer)
        streamer.insert(vec, 42)
        reader = ext.create_reader()
        while True:
            batch = reader.next_batch()
            if batch is None:
                break
            vectors, ids = batch

    Attach MUST happen before the first insert. ``bruteforce`` rejects attach only while it has
    visible rows; ``flatnsw`` rejects attach after its first insert even if every earlier row was
    deleted. Without a successful attach the backend allocates no raw-value storage. Only
    ``bruteforce`` and ``flatnsw`` support streaming export.

    Treat a ``None`` batch as the end-of-data signal: do not insert concurrently while draining
    a reader. Discard readers before the streamer goes away.
    """

    def __init__(self) -> None:
        self._native = _NativeStreamGetVectorExtension()
        self._attached_once: bool = False
        self._streamer_ref: Optional[Any] = None

    def attach(self, streamer: Any) -> StreamGetVectorExtension:
        """Attach this extension to an empty :class:`~flair.Streamer`.

        Must be called exactly once on this wrapper, before any insert.

        Args:
            streamer: A :class:`~flair.Streamer` instance that holds no visible rows yet.

        Returns:
            ``self``, for chaining.

        Raises:
            FlairError: If *streamer* is not a Streamer, was already attached to this wrapper,
                already holds visible rows, or the backend does not implement this extension.
        """
        from ..streamer import Streamer

        if not isinstance(streamer, Streamer):
            raise FlairError("streamer must be a flair.Streamer instance")
        if self._attached_once:
            raise FlairError("extension wrappers are single-use; create a new StreamGetVectorExtension")

        streamer._native.Attach(self._native)
        self._attached_once = True
        self._streamer_ref = weakref.ref(streamer)
        return self

    def create_reader(self) -> VectorBatchReader:
        """Create a cursor over the inserted vectors and ids.

        Returns:
            A :class:`VectorBatchReader` starting at row 0. Deleted rows are skipped.

        Raises:
            FlairError: If not attached, the streamer is gone, or the backend cannot serve
                original vectors.
        """
        streamer = self._require_attached_owner()
        return VectorBatchReader(self._native.CreateReader(), self, streamer.dimension)

    def create_reader_with_tag(self) -> TagBatchReader:
        """Create a cursor over the inserted vectors, ids, and original tag values.

        Requires the streamer to carry ``extension.build.tag.tag_schema``. Rows inserted through
        the plain ``Streamer.insert`` export an empty value list for every dimension.

        Returns:
            A :class:`TagBatchReader` starting at row 0. Deleted rows are skipped.

        Raises:
            FlairError: If not attached, the streamer is gone, or it has no tag schema.
        """
        streamer = self._require_attached_owner()
        return TagBatchReader(self._native.CreateReaderWithTag(), self, streamer.dimension)

    def _require_attached_owner(self) -> Any:
        if self._streamer_ref is None:
            raise FlairError("extension is not attached to a streamer")

        streamer = self._streamer_ref()
        if streamer is None:
            raise FlairError("attached streamer is no longer alive")

        return streamer
