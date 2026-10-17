from __future__ import annotations

import weakref
from typing import Any, Optional

from .._native import MutableStreamExtension as _NativeMutableStreamExtension
from ..errors import FlairError


class MutableStreamExtension:
    """Experimental: stream-time mutation (deletion) of an attached Streamer.

    Usage::

        ext = MutableStreamExtension().attach(streamer)
        streamer.insert(vec, id=42)
        ext.delete(42)

    The extension MUST be attached before the first insert; attaching later is
    rejected by the backend. Backends that do not support mutation reject
    :meth:`attach` with a ``FlairError`` (``NotSupported``).
    """

    def __init__(self) -> None:
        self._native = _NativeMutableStreamExtension()
        self._attached_once: bool = False
        self._streamer_ref: Optional[weakref.ref[Any]] = None

    def attach(self, streamer: Any) -> MutableStreamExtension:
        """Attach this extension to a :class:`~flair.Streamer`.

        Must be called exactly once, before the first insert on *streamer*.

        Args:
            streamer: A :class:`~flair.Streamer` instance.

        Returns:
            ``self``, for chaining (e.g. ``ext.attach(s)``).

        Raises:
            FlairError: If *streamer* is not a Streamer, this wrapper was
                already attached, the backend does not support mutation, or an
                insert has already happened on *streamer*.
        """
        from ..streamer import Streamer

        if not isinstance(streamer, Streamer):
            raise FlairError("streamer must be a flair.Streamer instance")
        if self._attached_once:
            raise FlairError("extension wrappers are single-use; create a new MutableStreamExtension")

        streamer._native.Attach(self._native)
        self._attached_once = True
        self._streamer_ref = weakref.ref(streamer)
        return self

    def delete(self, id: int) -> None:
        """Delete a previously inserted vector by its external id.

        Not thread-safe with concurrent writers: ``delete`` shares the streamer's
        single-writer domain with ``insert``, so calls to ``delete`` must be
        serialized with each other and with ``insert``. It is safe to call
        concurrently with ``search``.

        Args:
            id: The external id passed to :meth:`Streamer.insert`.

        Raises:
            FlairError: If this wrapper is not attached, the attached streamer
                has already been garbage collected, or the id does not exist
                (including already deleted).
        """
        # Pin a strong reference across the native call: Delete releases the GIL, so
        # without it another thread could drop the last Streamer reference and destroy
        # the backend mid-call, leaving the extension's delete callback dangling.
        streamer = self._require_attached_streamer()
        self._native.Delete(int(id))
        del streamer

    def _require_attached_streamer(self) -> Any:
        if self._streamer_ref is None:
            raise FlairError("extension is not attached to a streamer")

        streamer = self._streamer_ref()
        if streamer is None:
            raise FlairError("attached streamer is no longer alive")

        return streamer
