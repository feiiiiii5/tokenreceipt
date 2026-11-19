"""Bound the original scores of archived SmolLM2 chat responses."""

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path


TOKENIZER_HASH = "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"
VOCABULARY_SIZE = 49152
EOS = 2
UNBOUNDED = (-math.inf, 0.0)


class WireClient:
    """Read raw non-streaming MLX-LM 0.32.2 responses from a pinned tokenizer."""

    def __init__(self, tokenizer_path):
        raw = Path(tokenizer_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != TOKENIZER_HASH:
            raise ValueError("Expected the pinned SmolLM2-135M-Instruct tokenizer.json")
        vocabulary = json.loads(raw)["model"]["vocab"]
        literal = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
        table = {chr(value): value for value in literal}
        for offset, value in enumerate(sorted(set(range(256)) - literal)):
            table[chr(256 + offset)] = value

        self.trie = {}
        self.tail = set()
        for word, token_id in vocabulary.items():
            if token_id == EOS:
                continue
            # The stock detokenizer treats byte zero as a missing lookup.
            piece = b"".join(bytes([table[char]]) if table.get(char) else char.encode() for char in word)
            node = self.trie
            for value in piece:
                node = node.setdefault(value, {})
            node.setdefault(None, []).append(token_id)
            if piece == b" " or piece.decode("utf-8", "replace").endswith("\ufffd"):
                self.tail.add(token_id)

    @lru_cache(maxsize=16384)
    def matches(self, rendered):
        edges = []
        for start in range(len(rendered)):
            node = self.trie
            matches = []
            for end in range(start, len(rendered)):
                node = node.get(rendered[end])
                if node is None:
                    break
                matches.extend((end + 1, token_id) for token_id in node.get(None, ()))
            edges.append(matches)
        return edges

    def bounds(self, response):
        """Return a mean score interval, or None for an unsupported response."""
        try:
            if response["object"] != "chat.completion" or len(response["choices"]) != 1:
                return None
            choice = response["choices"][0]
            fields = choice["logprobs"]["content"]
            message = choice["message"]
            text = message["content"]
            finish = choice["finish_reason"]
            count = response["usage"]["completion_tokens"]
            if (not isinstance(fields, list) or not fields or type(count) is not int
                    or count != len(fields) or finish not in ("stop", "length")
                    or message["role"] != "assistant" or (text is not None and not isinstance(text, str))
                    or {"tool_calls", "reasoning", "reasoning_content"}.intersection(message)):
                return None

            scores_by_position = []
            for row in fields:
                candidates = row["top_logprobs"]
                if len(candidates) != 5:
                    return None
                scores = {}
                for candidate in candidates:
                    token_id, score = candidate["id"], candidate["logprob"]
                    if (type(token_id) is not int or not 0 <= token_id < VOCABULARY_SIZE
                            or type(score) not in (int, float) or not math.isfinite(score) or score > 0):
                        return None
                    scores[token_id] = score
                if len(scores) != 5:
                    return None
                scores_by_position.append(scores)
        except (KeyError, IndexError, TypeError, OverflowError):
            return None

        text = text or ""
        if "\ufffd" in text:
            return UNBOUNDED
        encoded = text.encode()
        stop = finish == "stop"
        totals = []
        for rendered in (encoded, b" " + encoded):
            edges = self.matches(rendered)
            # At each row, track the extrema of total scores at each byte offset.
            states = {0: (0.0, 0.0)}
            for position, scores in enumerate(scores_by_position):
                cutoff = min(scores.values())
                next_states = {}

                def extend(end, lower, upper):
                    old_lower, old_upper = next_states.get(end, (math.inf, -math.inf))
                    next_states[end] = min(old_lower, lower), max(old_upper, upper)

                for offset, (lower, upper) in states.items():
                    if position == count - 1 and stop:
                        if offset == len(rendered):
                            extend(offset, lower + scores.get(EOS, -math.inf), upper + scores.get(EOS, cutoff))
                    elif offset < len(rendered):
                        for end, token_id in edges[offset]:
                            extend(end, lower + scores.get(token_id, -math.inf), upper + scores.get(token_id, cutoff))
                    elif stop:
                        # An EOS response can hide an unflushed suffix of spaces or UTF-8 fragments.
                        tail_upper = max((score for token_id, score in scores.items() if token_id in self.tail),
                                         default=cutoff)
                        extend(offset, -math.inf, upper + tail_upper)
                states = next_states
                if not states:
                    break
            if len(rendered) in states:
                totals.append(states[len(rendered)])

        if not totals:
            return None
        return min(lower for lower, _ in totals) / count, max(upper for _, upper in totals) / count

    def winner(self, responses):
        """Return the zero-based certifiable winner of four draws, or None."""
        if len(responses) != 4:
            raise ValueError("Expected four responses in original draw order")
        bounds = [self.bounds(response) for response in responses]
        if any(interval is None for interval in bounds):
            return None
        winner = max(range(4), key=lambda index: (bounds[index][0], -index))
        if all(index == winner or (bounds[winner][0], -winner) >= (upper, -index)
               for index, (_, upper) in enumerate(bounds)):
            return winner
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True, help="pinned SmolLM2 tokenizer.json")
    parser.add_argument("responses", type=Path, nargs=4, metavar="RESPONSE", help="raw JSON in draw order")
    args = parser.parse_args()
    try:
        client = WireClient(args.tokenizer)
        responses = [json.loads(path.read_bytes()) for path in args.responses]
        print(json.dumps({"winner": client.winner(responses)}))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
