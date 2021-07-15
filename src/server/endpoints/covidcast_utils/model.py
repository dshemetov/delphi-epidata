from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Generator, Optional, Dict, List, Set, Tuple, Iterable, Counter, Union
from pathlib import Path
import re
from numpy.lib.utils import source
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

from ..._params import SourceSignalPair
from .smooth_diff import generate_smooth_rows, generate_row_diffs


IDENTITY = lambda rows, **kwargs: rows
DIFF = lambda rows, **kwargs: generate_row_diffs(rows, **kwargs)
SMOOTH = lambda rows, **kwargs: generate_smooth_rows(rows, **kwargs)
DIFF_SMOOTH = lambda rows, **kwargs: generate_smooth_rows(generate_row_diffs(rows, **kwargs), **kwargs)


class HighValuesAre(str, Enum):
    bad = "bad"
    good = "good"
    neutral = "neutral"


class SignalFormat(str, Enum):
    per100k = "per100k"
    percent = "percent"
    fraction = "fraction"
    raw_count = "raw_count"
    raw = "raw"
    count = "count"


class SignalCategory(str, Enum):
    public = "public"
    early = "early"
    late = "late"
    other = "other"


class TimeType(str, Enum):
    day = "day"
    week = "week"


@dataclass
class WebLink:
    alt: str
    href: str


def _fix_links(link: Optional[str]) -> List[WebLink]:
    # fix the link structure as given in (multiple) optional markdown link formats
    if not link:
        return []

    reg = re.compile("\[(.+)\]\s*\((.*)\)")

    def parse(l: str) -> Optional[WebLink]:
        l = l.strip()
        if not l:
            return None
        m = reg.match(l)
        if not m:
            return WebLink("API Documentation", l)
        return WebLink(m.group(1), m.group(2))

    return [l for l in map(parse, link.split(",")) if l]


@dataclass
class DataSignal:
    source: str
    signal: str
    signal_basename: str
    name: str
    short_description: str
    description: str
    time_label: str
    value_label: str
    format: SignalFormat = SignalFormat.raw
    category: SignalCategory = SignalCategory.other
    high_values_are: HighValuesAre = HighValuesAre.neutral
    is_smoothed: bool = False
    is_weighted: bool = False
    is_cumulative: bool = False
    has_stderr: bool = False
    has_sample_size: bool = False
    link: List[WebLink] = field(default_factory=list)
    compute_from_base: bool = False
    time_type: TimeType = TimeType.day

    def __post_init__(self):
        self.link = _fix_links(self.link)

    def initialize(self, source_map: Dict[str, "DataSource"], map: Dict[Tuple[str, str], "DataSignal"], initialized: Set[Tuple[str, str]]):
        # mark as initialized
        initialized.add(self.key)

        base = map.get((self.source, self.signal_basename))
        if base and base.key not in initialized:
            # initialize base first
            base.initialize(source_map, map, initialized)

        source = source_map.get(self.source)

        if not self.name:
            self.name = base.name if base else self.signal
        if not self.description:
            if base:
                self.description = base.description or base.short_description or "No description available"
            else:
                self.description = self.short_description or "No description available"
        if not self.short_description:
            if base:
                self.short_description = base.short_description or (base.description[:10] if base.description else "No description available")
            else:
                self.short_description = self.description[:10]
        if not self.link and base:
            self.link = base.link
        if not self.value_label:
            self.value_label = base.value_label if base else "Value"
        if not self.category:
            self.value_label = base.category if base else SignalCategory.other
        if not self.high_values_are:
            self.high_values_are = base.high_values_are if base else HighValuesAre.neutral

        self._replace_placeholders(base, source)

    def _replace_placeholders(self, base: Optional["DataSignal"], source: Optional["DataSource"]):
        text_replacements = {
            "base_description": base.description if base else "",
            "base_short_description": base.short_description if base else "",
            "base_name": base.name if base else "",
            "source_name": source.name if source else "",
            "source_description": source.description if source else "",
        }

        def replace_group(match: re.Match) -> str:
            key = match.group(1)
            if key and key in text_replacements:
                return text_replacements[key]
            return key

        def replace_replacements(text: str) -> str:
            return re.sub(r"\{([\w_]+)\}", replace_group, text)

        self.name = replace_replacements(self.name)
        # add new replacement on the fly for the next one
        text_replacements["name"] = self.name
        self.short_description = replace_replacements(self.short_description)
        text_replacements["short_description"] = self.short_description
        self.description = replace_replacements(self.description)

    def asdict(self):
        return asdict(self)

    @property
    def key(self) -> Tuple[str, str]:
        return (self.source, self.signal)


@dataclass
class DataSource:
    source: str
    db_source: str
    name: str
    active: bool
    description: str
    reference_signal: str
    license: Optional[str] = None
    link: List[WebLink] = field(default_factory=list)
    dua: Optional[str] = None

    signals: List[DataSignal] = field(default_factory=list)

    def __post_init__(self):
        self.link = _fix_links(self.link)
        if not self.db_source:
            self.db_source = self.source

    def asdict(self):
        r = asdict(self)
        r["signals"] = [r.asdict() for r in self.signals]
        return r

    @property
    def uses_db_alias(self):
        return self.source != self.db_source


def _clean_column(c: str) -> str:
    r = c.lower().replace(" ", "_").replace("-", "_").strip()
    if r == "source_subdivision":
        return "source"
    return r


_base_dir = Path(__file__).parent


def _load_data_sources():
    data_sources_df: pd.DataFrame = pd.read_csv(_base_dir / "db_sources.csv")
    data_sources_df = data_sources_df.replace({np.nan: None})
    data_sources_df.columns = map(_clean_column, data_sources_df.columns)
    data_sources: List[DataSource] = [DataSource(**d) for d in data_sources_df.to_dict(orient="records")]
    data_sources_df.set_index("source")
    return data_sources, data_sources_df


data_sources, data_sources_df = _load_data_sources()
data_sources_by_id = {d.source: d for d in data_sources}


def _load_data_signals(sources: List[DataSource]):
    by_id = {d.source: d for d in sources}
    data_signals_df: pd.DataFrame = pd.read_csv(_base_dir / "db_signals.csv")
    data_signals_df = data_signals_df.replace({np.nan: None})
    data_signals_df.columns = map(_clean_column, data_signals_df.columns)
    ignore_columns = {"base_is_other"}
    data_signals: List[DataSignal] = [
        DataSignal(**{k: v for k, v in d.items() if k not in ignore_columns}) for d in data_signals_df.to_dict(orient="records")
    ]
    data_signals_df.set_index(["source", "signal"])

    by_source_id = {d.key: d for d in data_signals}
    initialized: Set[Tuple[str, str]] = set()
    for ds in data_signals:
        ds.initialize(by_id, by_source_id, initialized)

    for ds in data_signals:
        source = by_id.get(ds.source)
        if source:
            source.signals.append(ds)

    return data_signals, data_signals_df


data_signals, data_signals_df = _load_data_signals(data_sources)
data_signals_by_key = {d.key: d for d in data_signals}


def get_related_signals(signal: DataSignal) -> List[DataSignal]:
    return [s for s in data_signals if s != signal and s.signal_basename == signal.signal_basename]


def create_source_signal_alias_mapper(source_signals: List[SourceSignalPair]) -> Tuple[List[SourceSignalPair], Optional[Callable[[str, str], str]]]:
    alias_to_data_sources: Dict[str, List[DataSource]] = {}
    transformed_pairs: List[SourceSignalPair] = []
    for pair in source_signals:
        source = data_sources_by_id.get(pair.source)
        if not source or not source.uses_db_alias:
            transformed_pairs.append(pair)
            continue
        # uses an alias
        alias_to_data_sources.setdefault(source.db_source, []).append(source)
        if pair.signal == True:
            # list all signals of this source (*) so resolve to a plain list of all in this alias
            transformed_pairs.append(SourceSignalPair(source.db_source, [s.signal for s in source.signals]))
        else:
            transformed_pairs.append(SourceSignalPair(source.db_source, pair.signal))

    if not alias_to_data_sources:
        # no alias needed
        return source_signals, None

    def map_row(source: str, signal: str) -> str:
        """
        maps a given row source back to its alias version
        """
        possible_data_sources = alias_to_data_sources.get(source)
        if not possible_data_sources:
            # nothing to transform
            return source
        if len(possible_data_sources) == 1:
            return possible_data_sources[0].source
        # need the signal to decide
        signal_source = next((f for f in possible_data_sources if any((s.signal == signal for s in f.signals))), None)
        if not signal_source:
            # take the first one
            signal_source = possible_data_sources[0]
        return signal_source.source

    return transformed_pairs, map_row


def _resolve_all_signals(
    source_signals: Union[SourceSignalPair, List[SourceSignalPair]], data_sources_by_id: DataFrame
) -> Union[SourceSignalPair, List[SourceSignalPair]]:
    if isinstance(source_signals, SourceSignalPair):
        if source_signals.signal == True:
            source = data_sources_by_id.get(source_signals.source)
            if source:
                return SourceSignalPair(source.source, [s.signal for s in source.signals])
        return source_signals
    if isinstance(source_signals, list):
        return [_resolve_all_signals(pair, data_sources_by_id) for pair in source_signals]
    raise TypeError("source_signals is not Union[SourceSignalPair, List[SourceSignalPair]].")


def _buffer_and_tag_iterator(
    it: Iterable[Dict], name_dict: Dict[Tuple[str, str], List[Tuple[str, str]]], keyfunc: Callable[[Dict], Tuple[str, str]]
) -> Iterable[Dict]:
    """Buffer an iterator for repeated passes.

    Parameters
    ----------
    it: Iterable[Dict]
        The iterator of dictionaries.
    name_dict: Dict[Tuple[str, str], List[Tuple[str, str]]]
        A dictionary with keys pointing to lists. If an entry in the iterable matches a key in the name_dict,
        the entry will be buffered and repeated the number of times equivalent to the length of the list
        iterable with the keyfunc value. E.g. each iterable is a row value for a signal and the Counter keys
        are (source, signal) tuples.
    keyfunc: Callable
        A function used to derive a name_dict key from a dictionary in the iterator.

    Returns
    ----------
    An iterator that runs through the original iterator values that do not match a name_dict key once, then runs
    through the values as specified by counts. Additionally, sets the value of the "_tag" key for every Dict in
    the iterator according to the number of times the given sequence of iterable values have been repeated (starting
    from 1).
    """
    buffer = dict()
    _name_dict = deepcopy(name_dict)

    # First iterator pass.
    for x in it:
        key: Tuple[str, str] = keyfunc(x)
        if _name_dict.get(key):
            buffer.setdefault(key, []).append(x)
            continue
        yield x, key

    # Buffer pass as needed.
    for key in buffer:
        for value in _name_dict[key]:
            for x in buffer[key]:
                yield x, value


def _get_basename_signals(
    source_signals: List[SourceSignalPair],
    data_sources_by_id: Dict[str, DataFrame] = data_sources_by_id,
    data_signals_by_key: Dict[str, DataFrame] = data_signals_by_key,
) -> Tuple[List[SourceSignalPair], Dict]:

    source_signals = _resolve_all_signals(source_signals, data_sources_by_id)
    base_signal_pairs: List[SourceSignalPair] = []
    name_dict = dict()

    for pair in source_signals:
        source_name: str = pair.source
        signals: List[str] = []

        if isinstance(pair.signal, bool):
            base_signal_pairs.append(pair)
            continue

        for signal_name in pair.signal:
            signal: DataSignal = data_signals_by_key.get((source_name, signal_name))
            if not signal or not signal.compute_from_base:
                signals.append(signal_name)
                continue

            signals.append(signal.signal_basename)
            name_dict.setdefault((source_name, signal.signal_basename), []).append((source_name, signal_name))
        base_signal_pairs.append(SourceSignalPair(pair.source, signals))

    return base_signal_pairs, name_dict


def _get_parent_signal(signal: DataSignal, data_signals_by_key: Dict[Tuple[str, str], DataFrame] = data_signals_by_key) -> DataSignal:
    parent_signal = data_signals_by_key.get((signal.source, signal.signal_basename))
    if parent_signal:
        return parent_signal
    return signal


def get_parent_transform(signal: Union[DataSignal, Tuple[str, str]], data_signals_by_key: Dict[Tuple[str, str], DataFrame] = data_signals_by_key) -> Callable:
    if isinstance(signal, DataSignal):
        if signal.format not in [SignalFormat.raw, SignalFormat.raw_count, SignalFormat.count]:
            return IDENTITY

        parent_signal = _get_parent_signal(signal, data_signals_by_key)
        if signal.is_cumulative and signal.is_smoothed:
            return SMOOTH
        if not signal.is_cumulative and not signal.is_smoothed:
            return DIFF if parent_signal.is_cumulative else IDENTITY
        if not signal.is_cumulative and signal.is_smoothed:
            return DIFF_SMOOTH if parent_signal.is_cumulative else SMOOTH
        return IDENTITY
    if isinstance(signal, tuple):
        signal = data_signals_by_key.get(signal)
        if signal:
            return get_parent_transform(signal, data_signals_by_key)
        return IDENTITY

    raise TypeError("signal must be either str or DataSignal.")


def create_basename_signal_transformer(source_signals: List[SourceSignalPair],
    data_sources_by_id: Dict[str, DataFrame] = data_sources_by_id,
    data_signals_by_key: Dict[str, DataFrame] = data_signals_by_key) -> Tuple[List[SourceSignalPair]]:
    base_signal_pairs, name_dict = _get_basename_signals(source_signals, data_sources_by_id, data_signals_by_key)
    iterator_buffer = lambda it, keyfunc: _buffer_and_tag_iterator(it, name_dict, keyfunc)

    return base_signal_pairs, iterator_buffer
