from __future__ import annotations

from typing import Dict, Optional, Union

OptionValue = Union[int, float, bool, str]

_BUILDER_RESERVED_KEYS = frozenset([
    "distance.metric",
    "encoding.type",
    "index.dimension",
    "index.type",
])
_STREAMER_RESERVED_KEYS = frozenset([
    "distance.metric",
    "encoding.type",
    "index.dimension",
    "index.type",
])
_SEARCHER_RESERVED_KEYS = frozenset([
    "index.dimension",
    "index.type",
])
_SEARCHER_FORBIDDEN_EXTRA_KEYS = frozenset([
    "distance.metric",
])
_SEARCH_PARAMS_RESERVED_KEYS = frozenset([
    "search.topk",
])
_IO_RESERVED_KEYS = frozenset([
    "index.path",
    "io.reader.type",
    "io.reader.mmap.lock_mode",
    "io.verify_crc",
])

def _merge_options(
    base: Dict[str, OptionValue],
    extra_options: Dict[str, OptionValue],
) -> Dict[str, OptionValue]:
    merged = dict(base)
    merged.update(extra_options)
    return merged


def _validate_extra_options(
    owner_name: str,
    extra_options: Dict[str, OptionValue],
    reserved_keys: frozenset[str],
) -> None:
    conflicting_keys = sorted(set(extra_options).intersection(reserved_keys))
    if conflicting_keys:
        raise ValueError(
            "{}.extra_options must not override explicit fields: {}".format(
                owner_name, ", ".join(conflicting_keys)
            )
        )


def _validate_no_io_extra_options(
    owner_name: str,
    extra_options: Dict[str, OptionValue],
) -> None:
    conflicting_keys = sorted(
        key for key in extra_options if key == "index.path" or key.startswith("io.")
    )
    if conflicting_keys:
        raise ValueError(
            "{}.extra_options must not contain IO options; use flair.IOOptions instead: {}".format(
                owner_name, ", ".join(conflicting_keys)
            )
        )


def _validate_searcher_extra_options(
    extra_options: Dict[str, OptionValue],
) -> None:
    conflicting_keys = sorted(set(extra_options).intersection(_SEARCHER_FORBIDDEN_EXTRA_KEYS))
    if conflicting_keys:
        raise ValueError(
            "SearcherConfig.extra_options must not contain unsupported searcher options: {}".format(
                ", ".join(conflicting_keys)
            )
        )


class BuilderConfig:
    """Configuration for creating a :class:`~flair.Builder`."""

    def __init__(
        self,
        dimension: int,
        index_type: str,
        distance_metric: Optional[str] = None,
        encoding_type: Optional[str] = None,
        extra_options: Optional[Dict[str, OptionValue]] = None,
    ) -> None:
        self.dimension = int(dimension)
        self.index_type = index_type
        self.distance_metric = distance_metric
        self.encoding_type = encoding_type
        self.extra_options: Dict[str, OptionValue] = {} if extra_options is None else dict(extra_options)
        _validate_extra_options("BuilderConfig", self.extra_options, _BUILDER_RESERVED_KEYS)
        _validate_no_io_extra_options("BuilderConfig", self.extra_options)

    def to_native_options(self) -> Dict[str, OptionValue]:
        _validate_extra_options("BuilderConfig", self.extra_options, _BUILDER_RESERVED_KEYS)
        _validate_no_io_extra_options("BuilderConfig", self.extra_options)
        options: Dict[str, OptionValue] = {
            "index.dimension": int(self.dimension),
            "index.type": self.index_type,
        }
        if self.distance_metric is not None:
            options["distance.metric"] = self.distance_metric
        if self.encoding_type is not None:
            options["encoding.type"] = self.encoding_type
        return _merge_options(options, self.extra_options)

    @classmethod
    def from_options(cls, options: Dict[str, OptionValue]) -> BuilderConfig:
        extra_options = dict(options)
        try:
            dimension = int(extra_options.pop("index.dimension"))
            index_type = str(extra_options.pop("index.type"))
        except KeyError as exc:
            raise ValueError(f"missing required option: {exc.args[0]}") from exc
        distance_metric = extra_options.pop("distance.metric", None)
        encoding_type = extra_options.pop("encoding.type", None)
        return cls(
            dimension=dimension,
            index_type=index_type,
            distance_metric=None if distance_metric is None else str(distance_metric),
            encoding_type=None if encoding_type is None else str(encoding_type),
            extra_options=extra_options,
        )


class SearcherConfig:
    """Configuration for creating a :class:`~flair.Searcher`."""

    def __init__(
        self,
        dimension: int,
        index_type: str,
        extra_options: Optional[Dict[str, OptionValue]] = None,
    ) -> None:
        self.dimension = int(dimension)
        self.index_type = index_type
        self.extra_options: Dict[str, OptionValue] = {} if extra_options is None else dict(extra_options)
        _validate_extra_options("SearcherConfig", self.extra_options, _SEARCHER_RESERVED_KEYS)
        _validate_no_io_extra_options("SearcherConfig", self.extra_options)
        _validate_searcher_extra_options(self.extra_options)

    def to_native_options(self) -> Dict[str, OptionValue]:
        _validate_extra_options("SearcherConfig", self.extra_options, _SEARCHER_RESERVED_KEYS)
        _validate_no_io_extra_options("SearcherConfig", self.extra_options)
        _validate_searcher_extra_options(self.extra_options)
        options: Dict[str, OptionValue] = {
            "index.dimension": int(self.dimension),
            "index.type": self.index_type,
        }
        return _merge_options(options, self.extra_options)

    @classmethod
    def from_options(cls, options: Dict[str, OptionValue]) -> SearcherConfig:
        extra_options = dict(options)
        try:
            dimension = int(extra_options.pop("index.dimension"))
            index_type = str(extra_options.pop("index.type"))
        except KeyError as exc:
            raise ValueError(f"missing required option: {exc.args[0]}") from exc
        return cls(
            dimension=dimension,
            index_type=index_type,
            extra_options=extra_options,
        )


class SearchParams:
    """Parameters for a single search query."""

    def __init__(
        self,
        topk: int = 10,
        extra_options: Optional[Dict[str, OptionValue]] = None,
    ) -> None:
        self.topk = int(topk)
        self.extra_options: Dict[str, OptionValue] = {} if extra_options is None else dict(extra_options)
        _validate_extra_options("SearchParams", self.extra_options, _SEARCH_PARAMS_RESERVED_KEYS)

    def to_native_options(self) -> Dict[str, OptionValue]:
        _validate_extra_options("SearchParams", self.extra_options, _SEARCH_PARAMS_RESERVED_KEYS)
        options: Dict[str, OptionValue] = {
            "search.topk": int(self.topk),
        }
        return _merge_options(options, self.extra_options)


class IOOptions:
    """IO settings for ``Builder.dump()`` and ``Searcher.open()``."""

    def __init__(
        self,
        reader_type: Optional[str] = None,
        mmap_lock_mode: Optional[str] = None,
        verify_crc: Optional[bool] = None,
        extra_options: Optional[Dict[str, OptionValue]] = None,
    ) -> None:
        self.reader_type = reader_type
        self.mmap_lock_mode = mmap_lock_mode
        self.verify_crc = verify_crc
        self.extra_options: Dict[str, OptionValue] = {} if extra_options is None else dict(extra_options)
        _validate_extra_options("IOOptions", self.extra_options, _IO_RESERVED_KEYS)

    def to_native_options(self) -> Dict[str, OptionValue]:
        _validate_extra_options("IOOptions", self.extra_options, _IO_RESERVED_KEYS)
        options: Dict[str, OptionValue] = {}
        if self.reader_type is not None:
            options["io.reader.type"] = self.reader_type
        if self.mmap_lock_mode is not None:
            options["io.reader.mmap.lock_mode"] = self.mmap_lock_mode
        if self.verify_crc is not None:
            options["io.verify_crc"] = bool(self.verify_crc)
        return _merge_options(options, self.extra_options)

    @classmethod
    def from_options(cls, options: Dict[str, OptionValue]) -> IOOptions:
        extra_options = dict(options)
        if "index.path" in extra_options:
            raise ValueError("IOOptions.from_options() must not include index.path; pass path to open()/dump()")
        reader_type = extra_options.pop("io.reader.type", None)
        mmap_lock_mode = extra_options.pop("io.reader.mmap.lock_mode", None)
        verify_crc = extra_options.pop("io.verify_crc", None)
        return cls(
            reader_type=None if reader_type is None else str(reader_type),
            mmap_lock_mode=None if mmap_lock_mode is None else str(mmap_lock_mode),
            verify_crc=None if verify_crc is None else bool(verify_crc),
            extra_options=extra_options,
        )


class StreamerConfig:
    """Configuration for creating a :class:`~flair.Streamer`."""

    def __init__(
        self,
        dimension: int,
        index_type: str,
        distance_metric: Optional[str] = None,
        encoding_type: Optional[str] = None,
        extra_options: Optional[Dict[str, OptionValue]] = None,
    ) -> None:
        self.dimension = int(dimension)
        self.index_type = index_type
        self.distance_metric = distance_metric
        self.encoding_type = encoding_type
        self.extra_options: Dict[str, OptionValue] = {} if extra_options is None else dict(extra_options)
        _validate_extra_options("StreamerConfig", self.extra_options, _STREAMER_RESERVED_KEYS)
        _validate_no_io_extra_options("StreamerConfig", self.extra_options)

    def to_native_options(self) -> Dict[str, OptionValue]:
        _validate_extra_options("StreamerConfig", self.extra_options, _STREAMER_RESERVED_KEYS)
        _validate_no_io_extra_options("StreamerConfig", self.extra_options)
        options: Dict[str, OptionValue] = {
            "index.dimension": int(self.dimension),
            "index.type": self.index_type,
        }
        if self.distance_metric is not None:
            options["distance.metric"] = self.distance_metric
        if self.encoding_type is not None:
            options["encoding.type"] = self.encoding_type
        return _merge_options(options, self.extra_options)

    @classmethod
    def from_options(cls, options: Dict[str, OptionValue]) -> "StreamerConfig":
        extra_options = dict(options)
        try:
            dimension = int(extra_options.pop("index.dimension"))
            index_type = str(extra_options.pop("index.type"))
        except KeyError as exc:
            raise ValueError(f"missing required option: {exc.args[0]}") from exc
        distance_metric = extra_options.pop("distance.metric", None)
        encoding_type = extra_options.pop("encoding.type", None)
        return cls(
            dimension=dimension,
            index_type=index_type,
            distance_metric=None if distance_metric is None else str(distance_metric),
            encoding_type=None if encoding_type is None else str(encoding_type),
            extra_options=extra_options,
        )
