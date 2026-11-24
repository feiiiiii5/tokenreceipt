"""Unit and offline-replay tests for tools/tokenreceipt_audit.py."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from tokenreceipt_audit import (  # noqa: E402
    CTranslate2NativeAdapter,
    PositionSlot,
    check_slot,
    compute_bytes_coherence,
    compute_transcript_coherence,
    replay_offline_matrix,
    summarize_slots,
)


class TokenReceiptAuditTests(unittest.TestCase):
    def test_invariants_pass_fail_abstain(self):
        # Clean slot passes I1, I2, I3 and abstains I4, I5 when byte/stream refs omitted.
        clean = PositionSlot(
            request_id="ok",
            position=0,
            claimed_id=24313,
            observed_score=-4.4375,
            candidates={6277: -2.25, 12: -2.9},
            genuine_top_set=True,
            reference_id=24313,
            reference_score=-4.4375,
        )
        v_clean = check_slot(clean)
        self.assertEqual(v_clean["I1"].status, "pass")
        self.assertEqual(v_clean["I2"].status, "pass")
        self.assertEqual(v_clean["I3"].status, "pass")
        self.assertEqual(v_clean["I4"].status, "abstain")
        self.assertEqual(v_clean["I5"].status, "abstain")

        # Equal-score ID swap fails I1 while I2 passes on identical numeric score.
        swap = PositionSlot(
            request_id="swap",
            position=0,
            claimed_id=6277,
            observed_score=-2.25,
            candidates={6277: -2.25, 999: -2.25},
            genuine_top_set=True,
            reference_id=999,
            reference_score=-2.25,
        )
        v_swap = check_slot(swap)
        self.assertEqual(v_swap["I1"].status, "fail")
        self.assertEqual(v_swap["I2"].status, "pass")

        # Unnormalized logit (>0) fails I2, missing row fails I3, U+FFFD fails I4, omitted stream fails I5.
        faulty = PositionSlot(
            request_id="bad",
            position=1,
            expected_extent=1,
            seen_extent=0,
            claimed_token="a",
            observed_score=19.3,
            observed_bytes=[0xEF, 0xBF, 0xBD],
            stream_has_logprobs=False,
        )
        v_bad = check_slot(faulty)
        self.assertEqual(v_bad["I1"].status, "abstain")
        self.assertEqual(v_bad["I2"].status, "fail")
        self.assertEqual(v_bad["I3"].status, "fail")
        self.assertEqual(v_bad["I4"].status, "fail")
        self.assertEqual(v_bad["I5"].status, "fail")

    def test_ctranslate2_native_adapter(self):
        rows = [
            {"token_id": 10, "piece": "a", "log_prob": -0.5, "final_id": 10, "final_chosen_logp": -0.5, "elements": 98304, "final_vocab": 49152},
            {"token_id": 20, "piece": "b", "log_prob": -0.7, "final_id": 20, "final_chosen_logp": -0.7, "elements": 0, "final_vocab": 49152},
        ]
        slots = CTranslate2NativeAdapter.slots_from_batch_rows("ct2-batch", rows)
        summary = summarize_slots([slots])
        self.assertEqual(summary["per_position"]["I1"], {"pass": 2})
        self.assertEqual(summary["per_position"]["I2"], {"pass": 2})
        self.assertEqual(summary["per_position"]["I3"], {"fail": 2})

    def test_replay_matrix_on_reproducibility(self):
        repro = REPO / "reproducibility"
        if not (repro / "evidence/ifeval.jsonl").exists():
            repro = REPO
        if not (repro / "evidence/ifeval.jsonl").exists():
            self.skipTest("reproducibility directory unavailable")
        mat = replay_offline_matrix(repro)
        self.assertEqual(mat["mlx/original"]["per_position"]["I1"], {"pass": 960, "fail": 528})
        self.assertEqual(mat["mlx/repaired"]["per_position"]["I1"], {"pass": 1488})
        self.assertEqual(mat["ct2/joint-dup"]["per_position"]["I3"], {"fail": 8})
        self.assertEqual(mat["ct2/split-dup"]["per_position"]["I3"], {"pass": 8})
        self.assertEqual(mat["ct2/joint-distinct-eq"]["per_position"]["I3"], {"fail": 48})
        self.assertEqual(mat["ct2/split-distinct-eq"]["per_position"]["I3"], {"pass": 48})
        self.assertEqual(mat["ct2/singletons-distinct-eq"]["per_position"]["I3"], {"pass": 48})

    def test_blackbox_idless_substitution_and_token_candidates(self):
        # Single black-box request without integer IDs: substituted token disagrees with message.content
        rows = [
            {"token": "The", "logprob": -0.1, "top_logprobs": [{"token": "The", "logprob": -0.1}]},
            {"token": "Ġarchitecture", "logprob": -2.25, "top_logprobs": [{"token": "Ġarchitecture", "logprob": -2.25}]},
        ]
        coherent_bad = compute_transcript_coherence(rows, "The monumental", "length")
        self.assertFalse(coherent_bad)
        bad_slot = PositionSlot(
            request_id="bb-bad",
            position=1,
            claimed_token="Ġarchitecture",
            observed_score=-2.25,
            genuine_top_set=True,
            token_candidates={"Ġarchitecture": -2.25},
            transcript_coherent=coherent_bad,
        )
        v_bad = check_slot(bad_slot)
        self.assertEqual(v_bad["I1"].status, "fail")
        # Clean black-box ByteLevel request with unlisted token obeying cutoff bound
        rows_ok = [
            {"token": "The", "logprob": -0.1, "top_logprobs": [{"token": "The", "logprob": -0.1}]},
            {"token": "Ġmonumental", "logprob": -4.4375, "top_logprobs": [{"token": "Ġarchitecture", "logprob": -2.25}]},
        ]
        coherent_ok = compute_transcript_coherence(rows_ok, "The monumental", "length")
        self.assertTrue(coherent_ok)
        ok_slot = PositionSlot(
            request_id="bb-ok",
            position=1,
            claimed_token="Ġmonumental",
            observed_score=-4.4375,
            genuine_top_set=True,
            token_candidates={"Ġarchitecture": -2.25},
            transcript_coherent=coherent_ok,
        )
        v_ok = check_slot(ok_slot)
        self.assertEqual(v_ok["I1"].status, "pass")
        self.assertEqual(v_ok["I2"].status, "pass")

    def test_blackbox_bytes_coherence_i4(self):
        rows_good = [
            {"token": "Ch", "bytes": [67, 104]},
            {"token": "ào", "bytes": [195, 160, 111]},
        ]
        self.assertTrue(compute_bytes_coherence(rows_good, "Chào", "length"))
        slot_good = PositionSlot(
            request_id="bytes-ok",
            position=0,
            claimed_token="Ch",
            observed_bytes=[67, 104],
            bytes_coherent=True,
        )
        self.assertEqual(check_slot(slot_good)["I4"].status, "pass")

        rows_bad = [
            {"token": "Ch", "bytes": [67, 104]},
            {"token": "a", "bytes": [97, 111]},
        ]
        self.assertFalse(compute_bytes_coherence(rows_bad, "Chào", "length"))
        slot_bad = PositionSlot(
            request_id="bytes-bad",
            position=1,
            claimed_token="a",
            observed_bytes=[97, 111],
            bytes_coherent=False,
        )
        self.assertEqual(check_slot(slot_bad)["I4"].status, "fail")

    def test_streaming_extent_drop_fails_i5(self):
        dropped = PositionSlot(
            request_id="sglang-stream-drop",
            position=0,
            claimed_token="hi",
            observed_score=-0.2,
            observed_bytes=[104, 105],
            stream_has_logprobs=True,
            stream_extent_ok=False,
            stream_token="hi",
            stream_score=-0.2,
            stream_bytes=[104, 105],
        )
        v = check_slot(dropped)
        self.assertEqual(v["I5"].status, "fail")
        self.assertIn("count", v["I5"].detail)

    def test_streaming_score_token_parity_passes_i5(self):
        matched = PositionSlot(
            request_id="stream-ok",
            position=0,
            claimed_token="hi",
            observed_score=-0.25,
            observed_bytes=[104, 105],
            stream_has_logprobs=True,
            stream_extent_ok=True,
            stream_token="hi",
            stream_score=-0.25,
            stream_bytes=[104, 105],
        )
        v = check_slot(matched)
        self.assertEqual(v["I5"].status, "pass")


if __name__ == "__main__":
    unittest.main()
