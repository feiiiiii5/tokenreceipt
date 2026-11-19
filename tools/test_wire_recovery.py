"""Exercise the public client's response validation with synthetic packets."""

from copy import deepcopy
import unittest

from wire_recovery import WireClient


class ResponseScopeTest(unittest.TestCase):
    def setUp(self):
        # A minimal renderer for the two possible one-token encodings of "A".
        self.client = WireClient.__new__(WireClient)
        self.client.trie = {ord("A"): {None: [10]}, ord(" "): {ord("A"): {None: [11]}}}
        self.client.tail = set()

        def packet(score):
            scores = (score, score, -5, -6, -7)
            candidates = [{"id": token_id, "logprob": logprob} for token_id, logprob in zip(range(10, 15), scores)]
            return {
                "object": "chat.completion",
                "choices": [{
                    "message": {"role": "assistant", "content": "A"},
                    "finish_reason": "length",
                    "logprobs": {"content": [{"top_logprobs": candidates}]},
                }],
                "usage": {"completion_tokens": 1},
            }

        self.responses = [packet(score) for score in (-1, -2, -3, -4)]

    def test_supported_group(self):
        self.assertEqual(self.client.winner(self.responses), 0)

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


if __name__ == "__main__":
    unittest.main()
