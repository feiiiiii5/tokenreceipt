"""Exercise the generalized public wire-recovery client across tokenizers, widths, and abstentions."""
from __future__ import annotations

from copy import deepcopy
import math
import os
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from wire_recovery import WireClient  # noqa: E402


class ResponseScopeTest(unittest.TestCase):
    def setUp(self):
        # Minimal synthetic trie for "A" vs " A".
        self.client = WireClient.__new__(WireClient)
        self.client.trie = {ord("A"): {None: [10]}, ord(" "): {ord("A"): {None: [11]}}}
        self.client.tail = set()

        def packet(score, k=5):
            scores = [score] + ([score] if k >= 2 else []) + [-5.0 - i * 0.1 for i in range(max(0, k - 2))]
            candidates = [{"id": 10 + i, "logprob": logprob} for i, logprob in enumerate(scores[:k])]
            return {
                "object": "chat.completion",
                "choices": [{
                    "message": {"role": "assistant", "content": "A"},
                    "finish_reason": "length",
                    "logprobs": {"content": [{"top_logprobs": candidates}]},
                }],
                "usage": {"completion_tokens": 1},
            }

        self._packet = packet
        self.responses = [packet(score) for score in (-1, -2, -3, -4)]

    def test_supported_group(self):
        self.assertEqual(self.client.winner(self.responses), 0)

    def test_arbitrary_k_widths(self):
        # At k=1, candidate list only holds id=10 so " A" (id=11) is unlisted; sound abstention.
        k1_responses = [self._packet(s, k=1) for s in (-1.0, -2.0, -3.0, -4.0)]
        self.assertIsNone(self.client.winner(k1_responses))
        # When only "A" is in the trie, k=1 certifies winner 0.
        k1_only = WireClient.__new__(WireClient)
        k1_only.trie = {ord("A"): {None: [10]}}
        k1_only.tail = set()
        k1_only.expected_top_k = 1
        self.assertEqual(k1_only.winner(k1_responses), 0)

        for k in (2, 3, 5, 10):
            with self.subTest(k=k):
                responses = [self._packet(s, k=k) for s in (-1.0, -2.0, -3.0, -4.0)]
                self.assertEqual(self.client.winner(responses), 0)

        # Reject mixed width inside one response.
        mixed = self._packet(-1.0, k=3)
        mixed2 = self._packet(-2.0, k=2)
        mixed["choices"][0]["logprobs"]["content"].append(mixed2["choices"][0]["logprobs"]["content"][0])
        mixed["usage"]["completion_tokens"] = 2
        self.assertIsNone(self.client.bounds(mixed))

    def test_reasoning_and_tool_fields(self):
        for field in ("reasoning", "reasoning_content", "tool_calls"):
            with self.subTest(field=field):
                responses = deepcopy(self.responses)
                responses[1]["choices"][0]["message"][field] = "separate content"
                self.assertIsNone(self.client.bounds(responses[1]))
                self.assertIsNone(self.client.winner(responses))

    def test_usage_disagrees_with_rows(self):
        self.responses[1]["usage"]["completion_tokens"] = 2
        self.assertIsNone(self.client.bounds(self.responses[1]))
        self.assertIsNone(self.client.winner(self.responses))

    def test_smollm2_regression_parity_if_present(self):
        tok_path = REPO / "experiments/data/tokenizer.json"
        if not tok_path.exists():
            self.skipTest("Optional local experiments/data/tokenizer.json absent")
        smol = WireClient(tok_path, eos_ids=[2])
        self.assertEqual(smol.vocabulary_size, 49152)
        resp = {
            "object": "chat.completion",
            "choices": [{
                "message": {"role": "assistant", "content": "Hello world"},
                "finish_reason": "stop",
                "logprobs": {"content": [
                    {"top_logprobs": [{"id": 19556, "logprob": -0.5}, {"id": 0, "logprob": -2.0}, {"id": 1, "logprob": -3.0}]},
                    {"top_logprobs": [{"id": 905, "logprob": -0.4}, {"id": 0, "logprob": -2.0}, {"id": 1, "logprob": -3.0}]},
                    {"top_logprobs": [{"id": 2, "logprob": -0.1}, {"id": 0, "logprob": -2.0}, {"id": 1, "logprob": -3.0}]},
                ]},
            }],
            "usage": {"completion_tokens": 3},
        }
        lo, hi = smol.bounds(resp)
        self.assertEqual(lo, -math.inf)
        self.assertAlmostEqual(hi, (-0.5 - 0.4 - 0.1) / 3.0)

        # Including the leading-space alternative with equal score yields an exact tight interval.
        tight_resp = {
            "object": "chat.completion",
            "choices": [{
                "message": {"role": "assistant", "content": "Hello world"},
                "finish_reason": "stop",
                "logprobs": {"content": [
                    {"top_logprobs": [{"id": 19556, "logprob": -0.5}, {"id": 38699, "logprob": -0.5}, {"id": 0, "logprob": -2.0}]},
                    {"top_logprobs": [{"id": 905, "logprob": -0.4}, {"id": 0, "logprob": -2.0}, {"id": 1, "logprob": -3.0}]},
                    {"top_logprobs": [{"id": 2, "logprob": -0.1}, {"id": 0, "logprob": -2.0}, {"id": 1, "logprob": -3.0}]},
                ]},
            }],
            "usage": {"completion_tokens": 3},
        }
        t_lo, t_hi = smol.bounds(tight_resp)
        self.assertAlmostEqual(t_lo, (-0.5 - 0.4 - 0.1) / 3.0)
        self.assertAlmostEqual(t_hi, (-0.5 - 0.4 - 0.1) / 3.0)

    def test_qwen25_tokenizer_if_present(self):
        env_tok = os.environ.get("TOKENRECEIPT_QWEN_TOKENIZER")
        qwen_tok = Path(env_tok) if env_tok else None
        if not qwen_tok or not qwen_tok.exists():
            self.skipTest("Optional TOKENRECEIPT_QWEN_TOKENIZER not provided")
        qclient = WireClient(qwen_tok)
        self.assertGreater(qclient.vocabulary_size, 150000)
        self.assertTrue(len(qclient.eos_ids) >= 1)
        self.assertTrue(any(qclient.matches(b"Final Answer: A")))


if __name__ == "__main__":
    unittest.main()
