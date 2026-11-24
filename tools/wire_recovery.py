"""Generalized wire-recovery client supporting any HuggingFace tokenizer.json and candidate width k."""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

UNBOUNDED = (-math.inf, 0.0)


def _byte_level_table() -> dict[str, int]:
    literal = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    table = {chr(value): value for value in literal}
    for offset, value in enumerate(sorted(set(range(256)) - literal)):
        table[chr(256 + offset)] = value
    return table


def _infer_eos_ids(raw_tok: dict, tokenizer_config_path: Path | None = None) -> set[int]:
    vocab: dict[str, int] = raw_tok.get("model", {}).get("vocab", {})
    added_map = {
        item["content"]: int(item["id"])
        for item in raw_tok.get("added_tokens", [])
        if "id" in item and "content" in item
    }
    eos_ids: set[int] = set()
    if tokenizer_config_path and tokenizer_config_path.exists():
        cfg = json.loads(tokenizer_config_path.read_bytes())
        eos_tok = cfg.get("eos_token")
        if isinstance(eos_tok, dict):
            eos_tok = eos_tok.get("content")
        if isinstance(eos_tok, str):
            if eos_tok in added_map:
                eos_ids.add(added_map[eos_tok])
            elif eos_tok in vocab:
                eos_ids.add(int(vocab[eos_tok]))
    for item in raw_tok.get("added_tokens", []):
        content = item.get("content", "")
        if item.get("special") and content in ("<|endoftext|>", "<|im_end|>", "</s>", "<|eot_id|>", "<eos>"):
            eos_ids.add(int(item["id"]))
    return eos_ids


class WireClient:
    """Read raw non-streaming chat-completion responses for arbitrary HuggingFace tokenizers and top-k."""

    vocabulary_size: int | None = None
    eos_ids: set[int] | frozenset[int] = frozenset({2})
    expected_top_k: int | None = None

    def __init__(
        self,
        tokenizer_path: Path | str,
        *,
        tokenizer_config_path: Path | str | None = None,
        expected_sha256: str | None = None,
        eos_ids: Iterable[int] | None = None,
        top_k: int | None = None,
    ):
        tok_path = Path(tokenizer_path)
        raw_bytes = tok_path.read_bytes()
        self.tokenizer_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        if expected_sha256 is not None and self.tokenizer_sha256 != expected_sha256:
            raise ValueError(
                f"Tokenizer SHA-256 mismatch: got {self.tokenizer_sha256}, expected {expected_sha256}"
            )
        raw = json.loads(raw_bytes)
        model = raw.get("model", {})
        vocab: dict[str, int] = model.get("vocab", {})
        if not vocab:
            raise ValueError("tokenizer.json missing model.vocab")

        all_ids = [int(v) for v in vocab.values()] + [
            int(item["id"]) for item in raw.get("added_tokens", []) if "id" in item
        ]
        self.vocabulary_size = max(all_ids) + 1 if all_ids else len(vocab)

        if eos_ids is not None:
            self.eos_ids = {int(x) for x in eos_ids}
        else:
            cfg_path = Path(tokenizer_config_path) if tokenizer_config_path else tok_path.parent / "tokenizer_config.json"
            self.eos_ids = _infer_eos_ids(raw, cfg_path)
            if not self.eos_ids:
                raise ValueError("Could not infer EOS token IDs; pass eos_ids explicitly")

        self.expected_top_k = top_k
        decoder_type = (raw.get("decoder") or {}).get("type", "")
        is_metaspace = decoder_type == "Metaspace"
        table = _byte_level_table()

        self.trie: dict = {}
        self.tail: set[int] = set()
        for word, token_id in vocab.items():
            tid = int(token_id)
            if tid in self.eos_ids:
                continue
            if is_metaspace:
                piece = word.replace("\u2581", " ").encode("utf-8")
            else:
                # Match MLX-LM BPEStreamingDetokenizer (treats byte zero as a missing lookup).
                piece = b"".join(bytes([table[char]]) if table.get(char) else char.encode("utf-8") for char in word)
            if not piece:
                continue
            node = self.trie
            for value in piece:
                node = node.setdefault(value, {})
            node.setdefault(None, []).append(tid)
            if piece == b" " or piece.decode("utf-8", "replace").endswith("\ufffd"):
                self.tail.add(tid)

    @lru_cache(maxsize=16384)
    def matches(self, rendered: bytes) -> list[list[tuple[int, int]]]:
        edges: list[list[tuple[int, int]]] = []
        for start in range(len(rendered)):
            node = self.trie
            found: list[tuple[int, int]] = []
            for end in range(start, len(rendered)):
                node = node.get(rendered[end])
                if node is None:
                    break
                found.extend((end + 1, token_id) for token_id in node.get(None, ()))
            edges.append(found)
        return edges

    def bounds(self, response: dict) -> tuple[float, float] | None:
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
            if (
                not isinstance(fields, list)
                or not fields
                or type(count) is not int
                or count != len(fields)
                or finish not in ("stop", "length")
                or message["role"] != "assistant"
                or (text is not None and not isinstance(text, str))
                or {"tool_calls", "reasoning", "reasoning_content"}.intersection(message)
            ):
                return None

            row_width = self.expected_top_k
            scores_by_position: list[dict[int, float]] = []
            for row in fields:
                candidates = row["top_logprobs"]
                if not isinstance(candidates, list) or not candidates:
                    return None
                if row_width is None:
                    row_width = len(candidates)
                elif len(candidates) != row_width:
                    return None
                scores: dict[int, float] = {}
                for candidate in candidates:
                    token_id, score = candidate["id"], candidate["logprob"]
                    if (
                        type(token_id) is not int
                        or token_id < 0
                        or (self.vocabulary_size is not None and token_id >= self.vocabulary_size)
                        or type(score) not in (int, float)
                        or not math.isfinite(score)
                        or score > 0
                    ):
                        return None
                    scores[token_id] = float(score)
                if len(scores) != row_width:
                    return None
                scores_by_position.append(scores)
        except (KeyError, IndexError, TypeError, OverflowError):
            return None

        text = text or ""
        if "\ufffd" in text:
            return UNBOUNDED
        encoded = text.encode("utf-8")
        stop = finish == "stop"
        totals: list[tuple[float, float]] = []
        eos_lookup = self.eos_ids
        for rendered in (encoded, b" " + encoded):
            edges = self.matches(rendered)
            # At each row, track the extrema of total scores at each byte offset.
            states: dict[int, tuple[float, float]] = {0: (0.0, 0.0)}
            for position, scores in enumerate(scores_by_position):
                cutoff = min(scores.values())
                next_states: dict[int, tuple[float, float]] = {}

                def extend(end: int, lower: float, upper: float) -> None:
                    old_lower, old_upper = next_states.get(end, (math.inf, -math.inf))
                    next_states[end] = min(old_lower, lower), max(old_upper, upper)

                for offset, (lower, upper) in states.items():
                    if position == count - 1 and stop:
                        if offset == len(rendered):
                            eos_vals = [scores[eid] for eid in eos_lookup if eid in scores]
                            eos_lo = min(eos_vals) if eos_vals else -math.inf
                            eos_hi = max(eos_vals) if eos_vals else cutoff
                            extend(offset, lower + eos_lo, upper + eos_hi)
                    elif offset < len(rendered):
                        for end, token_id in edges[offset]:
                            extend(end, lower + scores.get(token_id, -math.inf), upper + scores.get(token_id, cutoff))
                    elif stop:
                        # An EOS response can hide an unflushed suffix of spaces or UTF-8 fragments.
                        tail_upper = max(
                            (score for token_id, score in scores.items() if token_id in self.tail),
                            default=cutoff,
                        )
                        extend(offset, -math.inf, upper + tail_upper)
                states = next_states
                if not states:
                    break
            if len(rendered) in states:
                totals.append(states[len(rendered)])

        if not totals:
            return None
        return min(lower for lower, _ in totals) / count, max(upper for _, upper in totals) / count

    def winner(self, responses: list[dict]) -> int | None:
        """Return the zero-based certifiable winner of responses in draw order, or None."""
        if not responses:
            raise ValueError("Expected at least one response in draw order")
        bounds = [self.bounds(response) for response in responses]
        if any(interval is None for interval in bounds):
            return None
        best = max(range(len(responses)), key=lambda index: (bounds[index][0], -index))
        if all(
            index == best or (bounds[best][0], -best) >= (upper, -index)
            for index, (_, upper) in enumerate(bounds)
        ):
            return best
        return None


WireClientV2 = WireClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True, help="HuggingFace tokenizer.json")
    parser.add_argument("--tokenizer-config", type=Path, default=None)
    parser.add_argument("--expected-sha256", type=str, default=None)
    parser.add_argument("--eos-id", type=int, action="append", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--include-bounds", action="store_true")
    parser.add_argument("responses", type=Path, nargs="+", metavar="RESPONSE")
    args = parser.parse_args()
    try:
        client = WireClient(
            args.tokenizer,
            tokenizer_config_path=args.tokenizer_config,
            expected_sha256=args.expected_sha256,
            eos_ids=args.eos_id,
            top_k=args.top_k,
        )
        responses = [json.loads(path.read_bytes()) for path in args.responses]
        out: dict[str, object] = {"winner": client.winner(responses)}
        if args.include_bounds:
            out["bounds"] = [client.bounds(r) for r in responses]
        print(json.dumps(out))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
