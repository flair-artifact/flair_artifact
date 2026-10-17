from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt

from ._native import FlairBuilder as _NativeBuilder
from .config import BuilderConfig, IOOptions, OptionValue
from .errors import FlairError


class Builder:
    """Build a Flair index from vectors.

    Usage::

        with Builder(BuilderConfig(dimension=128, index_type="bruteforce", distance_metric="l2")) as builder:
            builder.insert(vectors, ids)
            builder.dump("/path/to/index")
    """

    def __init__(self, config: BuilderConfig) -> None:
        self._config = config
        self._native = _NativeBuilder.Create(config.to_native_options())

    @property
    def dimension(self) -> int:
        """Return the configured vector dimension."""
        return int(self._native.Dimension)

    @property
    def status(self) -> str:
        """Return the current builder status."""
        return str(self._native.Status)

    def pretrain(self, vectors: npt.NDArray[np.float32], n: Optional[int] = None) -> None:
        """Pre-train the index encoding (e.g. PQ codebook) using a representative sample.

        Args:
            vectors: Training vectors, shape ``(sample_count, dimension)``.
            n: Number of vectors to use from *vectors*. Defaults to ``len(vectors)``.
        """
        self._native.Pretrain(vectors, n)

    def insert(self, vectors: npt.NDArray[np.float32], ids: npt.NDArray[np.uint64]) -> None:
        """Insert a batch of vectors into the index.

        Args:
            vectors: Data vectors, shape ``(count, dimension)``, dtype ``float32``.
            ids: Unique IDs for each vector, shape ``(count,)``, dtype ``uint64``.
        """
        self._native.InsertBatch(vectors, ids)

    def insert_from(self, dataset: Any) -> None:
        """Insert vectors from a dataset adapter.

        The *dataset* must implement ``Dim() -> int``, ``TotalSize() -> int``,
        and ``GetNextBatch() -> Optional[Tuple[ndarray, ndarray]]``.
        """
        self._native.InsertFrom(dataset)

    def __enter__(self) -> Builder:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        return False

    def dump(self, path: str, io_options: Optional[IOOptions] = None) -> None:
        """Write the built index to disk.

        Args:
            path: Output file path.
            io_options: Optional IO configuration.
        """
        if io_options is None:
            effective_io_options = IOOptions()
        elif isinstance(io_options, IOOptions):
            effective_io_options = io_options
        else:
            raise FlairError("io_options must be a flair.IOOptions instance")

        self._native.Dump(path, effective_io_options.to_native_options())

    @classmethod
    def from_options(cls, options: Dict[str, OptionValue]) -> Builder:
        """Create a Builder from a flat options dictionary (escape hatch)."""
        return cls(BuilderConfig.from_options(options))
