from collections import Counter
from itertools import groupby
from typing import Dict, Iterable
import unittest
from pandas import DataFrame, to_datetime, date_range
from pandas.testing import assert_frame_equal

from delphi.epidata.server._params import SourceSignalPair
from delphi.epidata.server.endpoints.covidcast_utils.model import (
    IDENTITY,
    DIFF,
    SMOOTH,
    DIFF_SMOOTH,
    DataSource,
    DataSignal,
    _resolve_all_signals,
    _buffer_and_tag_iterator,
    _get_basename_signals,
    get_parent_transform
)

data_signals_by_key = {
    ("src", "sig_diff"): DataSignal(
        source="src",
        signal="sig_diff",
        signal_basename="sig_base",
        name="src",
        short_description="",
        description="",
        time_label="",
        value_label="",
        is_cumulative=False,
        compute_from_base=True,
    ),
    ("src", "sig_smooth"): DataSignal(
        source="src",
        signal="sig_smooth",
        signal_basename="sig_base",
        name="src",
        short_description="",
        description="",
        time_label="",
        value_label="",
        is_cumulative=True,
        is_smoothed=True,
        compute_from_base=True,
    ),
    ("src", "sig_diff_smooth"): DataSignal(
        source="src",
        signal="sig_diff_smooth",
        signal_basename="sig_base",
        name="src",
        short_description="",
        description="",
        time_label="",
        value_label="",
        is_cumulative=False,
        is_smoothed=True,
        compute_from_base=True,
    ),
    ("src", "sig_base"): DataSignal(
        source="src",
        signal="sig_base",
        signal_basename="sig_base",
        name="src",
        short_description="",
        description="",
        time_label="",
        value_label="",
        is_cumulative=True,
    ),
    ("src2", "sig_base"): DataSignal(
        source="src2",
        signal="sig_base",
        signal_basename="sig_base",
        name="sig_base",
        short_description="",
        description="",
        time_label="",
        value_label="",
        is_cumulative=True,
    ),
    ("src2", "sig_diff_smooth"): DataSignal(
        source="src2",
        signal="sig_smooth",
        signal_basename="sig_base",
        name="sig_smooth",
        short_description="",
        description="",
        time_label="",
        value_label="",
        is_cumulative=False,
        is_smoothed=True,
        compute_from_base=True,
    ),
}

data_sources_by_id = {
    "src": DataSource(
        source="src",
        db_source="src",
        name="src",
        active=True,
        description="",
        reference_signal="sig_base",
        signals=[data_signals_by_key[key] for key in data_signals_by_key if key[0] == "src"],
    ),
    "src2": DataSource(
        source="src2",
        db_source="src2",
        name="src2",
        active=True,
        description="",
        reference_signal="sig_base",
        signals=[data_signals_by_key[key] for key in data_signals_by_key if key[0] == "src2"],
    ),
}


class TestModel(unittest.TestCase):
    def test__resolve_all_signals(self):
        source_signal_pair = SourceSignalPair("src", True)
        expected_source_signal_pair = SourceSignalPair(source="src", signal=["sig_diff", "sig_smooth", "sig_diff_smooth", "sig_base"])
        source_signal_pair = _resolve_all_signals(source_signal_pair, data_sources_by_id)
        assert source_signal_pair == expected_source_signal_pair

    def test_get_parent_transform(self):
        assert get_parent_transform(data_signals_by_key[("src", "sig_diff")], data_signals_by_key) == DIFF
        assert get_parent_transform(("src", "sig_diff"), data_signals_by_key) == DIFF
        assert get_parent_transform(data_signals_by_key[("src", "sig_smooth")], data_signals_by_key) == SMOOTH
        assert get_parent_transform(data_signals_by_key[("src", "sig_diff_smooth")], data_signals_by_key) == DIFF_SMOOTH
        assert get_parent_transform(data_signals_by_key[("src", "sig_base")], data_signals_by_key) == IDENTITY
        assert get_parent_transform(("src", "sig_unknown"), data_signals_by_key) == IDENTITY

    def test__buffer_and_tag_iterator(self):
        data = DataFrame(
            {
                "source": ["src"] * 10,
                "signal": ["sig_base"] * 5 + ["sig_other"] * 5,
                "timestamp": to_datetime(date_range("2021-05-01", "2021-05-5").to_list() * 2),
                "geo_type": ["state"] * 10,
                "geo_value": ["ca"] * 10,
                "value": list(range(10)),
            }
        )
        expected_df = DataFrame(
            {
                "source": ["src"] * 15,
                "signal": ["sig_other"] * 5 + ["sig_base"] * 5 + ["sig_base"] * 5,
                "timestamp": to_datetime(date_range("2021-05-01", "2021-05-5").to_list() * 3),
                "geo_type": ["state"] * 15,
                "geo_value": ["ca"] * 15,
                "value": list(range(5, 10)) + list(range(5)) * 2,
            }
        )
        with self.subTest("plain repeat"):
            name_dict = {("src", "sig_base"): [("src", "sig_smooth"), ("src", "sig_diff")], ("src_extra", "sig_base"): [("src_extra", "sig_smooth")]}
            repeat_df = DataFrame.from_records(
                [row for row, key in _buffer_and_tag_iterator(data.to_dict(orient="records"), name_dict, lambda x: (x["source"], x["signal"]))]
            )
            assert_frame_equal(repeat_df, expected_df)

        with self.subTest("compare values"):
            repeat_iterator = _buffer_and_tag_iterator(data.to_dict(orient="records"), name_dict, lambda x: (x["source"], x["signal"]))
            group_keyfunc = lambda entry: (entry[0]["source"], entry[0]["signal"], entry[1])
            groups_df = [DataFrame.from_records([row for row, key in group]) for _, group in groupby(repeat_iterator, group_keyfunc)]
            assert_frame_equal(groups_df[0].reset_index(drop=True), expected_df.iloc[0:5].reset_index(drop=True))
            assert_frame_equal(groups_df[1].reset_index(drop=True), expected_df.iloc[5:10].reset_index(drop=True))
            assert_frame_equal(groups_df[2].reset_index(drop=True), expected_df.iloc[10:15].reset_index(drop=True))

        with self.subTest("empty iterator"):
            repeat_iterator = [row for row, key in _buffer_and_tag_iterator({}, name_dict, lambda x: (x["source"], x["signal"]))]
            assert list(repeat_iterator) == []

        with self.subTest("empty name dict"):
            name_dict = {}
            repeat_iterator = [
                row for row, key in _buffer_and_tag_iterator(data.to_dict(orient="records"), name_dict, lambda x: (x["source"], x["signal"]))
            ]
            repeat_df = DataFrame.from_records(repeat_iterator)
            assert_frame_equal(repeat_df, data)

    def test__get_basename_signals(self):
        with self.subTest("none to transform"):
            source_signal_pairs = [SourceSignalPair("src", signal=["sig_base"])]
            expected_basename_pairs = [SourceSignalPair("src", signal=["sig_base"])]
            expected_name_dict = dict()
            basename_pairs, name_dict = _get_basename_signals(source_signal_pairs, data_sources_by_id, data_signals_by_key)
            assert basename_pairs == expected_basename_pairs
            assert name_dict == expected_name_dict

        with self.subTest("unrecognized signal"):
            source_signal_pairs = [SourceSignalPair("src", signal=["sig_unknown"])]
            expected_basename_pairs = [SourceSignalPair("src", signal=["sig_unknown"])]
            expected_name_dict = dict()
            basename_pairs, name_dict = _get_basename_signals(source_signal_pairs, data_sources_by_id, data_signals_by_key)
            assert basename_pairs == expected_basename_pairs
            assert name_dict == expected_name_dict

        with self.subTest("plain"):
            source_signal_pairs = [
                SourceSignalPair("src", signal=["sig_diff", "sig_smooth", "sig_diff_smooth", "sig_base"]),
                SourceSignalPair("src2", signal=["sig"]),
            ]
            expected_basename_pairs = [
                SourceSignalPair(source="src", signal=["sig_base", "sig_base", "sig_base", "sig_base"]),
                SourceSignalPair("src2", signal=["sig"]),
            ]
            expected_name_dict = {("src", "sig_base"): [("src", "sig_diff"), ("src", "sig_smooth"), ("src", "sig_diff_smooth")]}
            basename_pairs, name_dict = _get_basename_signals(source_signal_pairs, data_sources_by_id, data_signals_by_key)
            assert basename_pairs == expected_basename_pairs
            assert name_dict == expected_name_dict

        with self.subTest("resolve signals called"):
            source_signal_pairs = [SourceSignalPair("src", True)]
            expected_basename_pairs = [SourceSignalPair(source="src", signal=["sig_base", "sig_base", "sig_base", "sig_base"])]
            expected_name_dict = {("src", "sig_base"): [("src", "sig_diff"), ("src", "sig_smooth"), ("src", "sig_diff_smooth")]}
            basename_pairs, name_dict = _get_basename_signals(source_signal_pairs, data_sources_by_id, data_signals_by_key)
            assert basename_pairs == expected_basename_pairs
            assert name_dict == expected_name_dict
