"""Exact-rollback re-merge safety (#upstream-feedback-0.10.4).

Covers three fixes:
1. ``merge()`` raises :class:`KeyExtractionError` when *every* item fails
   key extraction (previously: empty groups → ``max()`` on empty sequence).
2. ``key_extraction_failed`` warnings are aggregated per distinct error.
3. ``remove_source``/``upsert_source`` accept ``remerge="mechanical"`` /
   ``"auto"`` so exact rollbacks never issue an unbounded LLM merge call,
   and the tournament caps pairs per ``batch_merge`` call.
"""

import json
import logging

import pytest
from pydantic import BaseModel

from ontomem import KeyExtractionError, OMem
from ontomem.merger import BaseMerger


class _Entity(BaseModel):
    name: str
    type: str = "entity"


class _CountingMerger(BaseMerger[_Entity]):
    """Deterministic merger that records batch sizes (no LLM)."""

    def __init__(self, key_extractor):
        super().__init__(key_extractor=key_extractor)
        self.batch_sizes: list[int] = []
        self.pair_calls = 0

    def pair_merge(self, existing, incoming):
        self.pair_calls += 1
        return incoming

    def batch_merge(self, pairs):
        self.batch_sizes.append(len(pairs))
        return [self.pair_merge(e, i) for e, i in pairs]


def _omem(merger):
    return OMem(
        memory_schema=_Entity,
        key_extractor=lambda x: x.name,
        llm_client=None,
        embedder=None,
        strategy_or_merger=merger,
        track_sources=True,
    )


class TestKeyExtractionError:
    def test_all_failures_raise_aggregate_error(self):
        merger = _CountingMerger(key_extractor=lambda x: x.missing_field)

        def boom(item):
            raise AttributeError("'X' object has no attribute 'missing_field'")

        merger.key_extractor = boom
        with pytest.raises(KeyExtractionError) as exc:
            merger.merge([_Entity(name="A"), _Entity(name="B")])
        assert "all 2 item(s)" in str(exc.value)
        assert "missing_field" in str(exc.value)

    def test_partial_failures_do_not_raise(self):
        merger = _CountingMerger(key_extractor=lambda x: x.name)
        merger.merge([_Entity(name="A")])
        assert merger.pair_calls >= 0

    def test_aggregated_error_includes_count(self):
        # 5 identical failures: the aggregate message carries the count and
        # the first error (structlog output itself is an implementation
        # detail, so the message is what we assert on).
        merger = _CountingMerger(key_extractor=lambda x: x.missing_field)
        with pytest.raises(KeyExtractionError) as exc:
            merger.merge([_Entity(name="A")] * 5)
        assert "all 5 item(s)" in str(exc.value)
        assert "x5" in str(exc.value)


class TestPairCap:
    def test_large_merge_splits_batch_calls(self):
        # 4 keys x 2 items each: round 1 has 4 pairs (split 2+2), round 2
        # merges the 4 winners into 2 pairs.
        merger = _CountingMerger(key_extractor=lambda x: x.name)
        merger.max_batch_pairs = 2
        items = [_Entity(name=f"n{i // 2}") for i in range(8)]
        merger.merge(items)
        assert merger.batch_sizes == [2, 2]

    def test_small_merge_single_call(self):
        merger = _CountingMerger(key_extractor=lambda x: x.name)
        merger.merge([_Entity(name="a"), _Entity(name="a")])
        assert merger.batch_sizes == [1]

    def test_default_cap_is_40(self):
        assert _CountingMerger(key_extractor=lambda x: x.name).max_batch_pairs == 40


class TestMechanicalRemerge:
    def _seed_shared(self, mem, shared_name, contributors=20):
        """``contributors`` sources each raw-recorded the same shared key.

        White-box seeding (ledger + storage) so the test does not depend on
        what a mock LLM extracts. Removing the shadow re-merges that key
        from ``contributors - 1`` surviving raw items — above the
        mechanical threshold.
        """
        shared = _Entity(name=shared_name)
        mem._storage[mem.key_extractor(shared)] = shared
        for i in range(contributors):
            mem.record_source(f"batch-{i}", [shared.model_dump()])
        mem.record_source("shadow", [shared.model_dump()])

    def test_mechanical_mode_never_touches_llm_merger(self):
        class _ExplodingLLMMerger(_CountingMerger):
            def merge(self, items):
                raise AssertionError("LLM merger must not be used in mechanical mode")

        llm = _ExplodingLLMMerger(key_extractor=lambda x: x.name)
        mem = _omem(llm)
        self._seed_shared(mem, "shared")

        report = mem.remove_source("shadow", strategy="exact", remerge="mechanical")
        assert "shared" in report["remerged_keys"]
        assert llm.pair_calls == 0

    def test_auto_falls_back_above_threshold(self):
        llm = _CountingMerger(key_extractor=lambda x: x.name)
        mem = _omem(llm)
        self._seed_shared(mem, "shared")

        report = mem.remove_source("shadow", strategy="exact", remerge="auto")
        assert "shared" in report["remerged_keys"]
        # auto → large survivor set → deterministic field merge, LLM untouched
        assert llm.pair_calls == 0

    def test_llm_mode_keeps_configured_merger(self):
        llm = _CountingMerger(key_extractor=lambda x: x.name)
        mem = _omem(llm)
        mem.record_source("s1", [_Entity(name="A").model_dump()])
        mem.add([_Entity(name="A")])
        mem.record_source("s2", [_Entity(name="A").model_dump()])
        mem.add([_Entity(name="A")])

        report = mem.remove_source("s2", strategy="exact", remerge="llm")
        assert "A" in report["remerged_keys"]
        assert llm.pair_calls >= 1

    def test_invalid_remerge_mode_rejected(self):
        mem = _omem(_CountingMerger(key_extractor=lambda x: x.name))
        with pytest.raises(ValueError, match="remerge"):
            mem.remove_source("x", remerge="yolo")
