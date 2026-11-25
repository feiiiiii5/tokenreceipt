"""TokenReceipt serving-boundary contract auditor (live HTTP, native adapters, & offline replay-matrix)."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any
from urllib.request import ProxyHandler, Request, build_opener

HTTP_OPENER = build_opener(ProxyHandler({}))
UTF8_REPLACEMENT_BYTES = [0xEF, 0xBF, 0xBD]
STOP_TOKEN_MARKERS = frozenset(
    {"<|endoftext|>", "<|im_end|>", "</s>", "<|eot_id|>", "<eos>", "<end_of_turn>", "<|end|>"}
)


def _byte_level_table() -> dict[str, int]:
    literal = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    table = {chr(value): value for value in literal}
    for offset, value in enumerate(sorted(set(range(256)) - literal)):
        table[chr(256 + offset)] = value
    return table


BYTE_LEVEL_TABLE = _byte_level_table()


def _decode_byte_level_tokens(tokens: list[str]) -> str:
    chunks: list[bytes] = []
    for tok in tokens:
        for ch in tok:
            if ch in BYTE_LEVEL_TABLE:
                chunks.append(bytes([BYTE_LEVEL_TABLE[ch]]))
            else:
                chunks.append(ch.encode("utf-8"))
    return b"".join(chunks).decode("utf-8", "replace")


def _is_stop_piece(piece: str | None) -> bool:
    if not piece:
        return False
    return piece in STOP_TOKEN_MARKERS or (piece.startswith("<") and piece.endswith(">") and len(piece) <= 24)


def compute_bytes_coherence(
    rows: list[dict[str, Any]],
    text: str,
    finish_reason: str | None = None,
) -> bool | None:
    """Check whether per-token byte arrays decode coherently to rendered completion text."""
    if not rows:
        return None
    if not all(
        isinstance(r.get("bytes"), list) and all(type(b) is int and 0 <= b <= 255 for b in r["bytes"])
        for r in rows
    ):
        return None

    joined = bytes(b for r in rows for b in r["bytes"]).decode("utf-8", "replace")
    if "\ufffd" in joined and "\ufffd" not in text:
        return False
    if joined == text or joined.strip() == text.strip():
        return True
    if finish_reason == "stop" and len(rows) >= 2 and _is_stop_piece(rows[-1].get("token")):
        without_stop = bytes(b for r in rows[:-1] for b in r["bytes"]).decode("utf-8", "replace")
        if "\ufffd" in without_stop and "\ufffd" not in text:
            return False
        if without_stop == text or without_stop.strip() == text.strip():
            return True
    return False


def compute_transcript_coherence(
    rows: list[dict[str, Any]],
    text: str,
    finish_reason: str | None = None,
) -> bool | None:
    """Check whether reported token strings reconstruct rendered completion text."""
    if not rows or not all(isinstance(r.get("token"), str) for r in rows):
        return None

    pieces = [r["token"] for r in rows]
    candidates = [pieces]
    if finish_reason == "stop" and len(pieces) >= 2 and _is_stop_piece(pieces[-1]):
        candidates.append(pieces[:-1])

    for seq in candidates:
        literal = "".join(seq)
        if literal == text or literal.strip() == text.strip():
            return True
        decoded_bl = _decode_byte_level_tokens(seq)
        if decoded_bl == text or decoded_bl.strip() == text.strip():
            return True
        if any("\u2581" in p for p in seq):
            decoded_sp = "".join(p.replace("\u2581", " ") for p in seq)
            if decoded_sp == text or decoded_sp.strip() == text.strip():
                return True
    return False


@dataclass
class PositionSlot:
    """Single generated token emission observed at a serving interface."""

    request_id: str
    position: int
    expected_extent: int = 1
    seen_extent: int = 1
    claimed_id: int | None = None
    claimed_token: str | None = None
    observed_score: float | None = None
    observed_bytes: list[int] | None = None
    candidates: dict[int, float] | None = None
    genuine_top_set: bool = False
    reference_id: int | None = None
    reference_score: float | None = None
    reference_bytes: list[int] | None = None
    abs_tol: float = 1e-4
    rel_tol: float = 1e-5
    # Optional black-box / token-string witnesses for I1, I2, I4
    reference_token: str | None = None
    token_candidates: dict[str, float] | None = None
    transcript_coherent: bool | None = None
    bytes_coherent: bool | None = None
    # Optional native batch callback metadata for I3
    callback_elements: int | None = None
    expected_callback_elements: int | None = None
    # Optional cross-transport stream observation for I5
    stream_has_logprobs: bool | None = None
    stream_extent_ok: bool | None = None
    stream_token: str | None = None
    stream_score: float | None = None
    stream_bytes: list[int] | None = None


@dataclass
class InvariantVerdict:
    invariant: str
    status: str  # "pass" | "fail" | "abstain"
    detail: str = ""


def _float_eq(a: float | None, b: float | None, abs_tol: float, rel_tol: float) -> str:
    if a is None or b is None:
        return "abstain"
    if not (math.isfinite(a) and math.isfinite(b)):
        return "fail"
    if abs(a - b) <= abs_tol + rel_tol * abs(b):
        return "pass"
    return "fail"


def check_slot(slot: PositionSlot) -> dict[str, InvariantVerdict]:
    """Evaluate Invariants I1-I5 on one PositionSlot with explicit abstentions."""
    # I1: Selected-Identity Consistency (oracle ID -> paired token -> black-box transcript coherence)
    if slot.claimed_id is not None and slot.reference_id is not None:
        i1 = InvariantVerdict(
            "I1",
            "pass" if slot.claimed_id == slot.reference_id else "fail",
            f"claimed_id={slot.claimed_id}, ref_id={slot.reference_id}",
        )
    elif slot.claimed_token is not None and slot.reference_token is not None:
        ok = slot.claimed_token == slot.reference_token and slot.transcript_coherent is not False
        i1 = InvariantVerdict(
            "I1",
            "pass" if ok else "fail",
            f"claimed_token={slot.claimed_token!r}, ref_token={slot.reference_token!r}, transcript_coherent={slot.transcript_coherent}",
        )
    elif slot.transcript_coherent is not None:
        i1 = InvariantVerdict(
            "I1",
            "pass" if slot.transcript_coherent else "fail",
            "transcript matches message.content"
            if slot.transcript_coherent
            else "reported token sequence disagrees with rendered message.content",
        )
    else:
        i1 = InvariantVerdict("I1", "abstain", "missing identity witness")

    # I2: Score-Stage & Top-k Cutoff Consistency
    score_state = _float_eq(slot.observed_score, slot.reference_score, slot.abs_tol, slot.rel_tol)
    if slot.observed_score is not None and (not math.isfinite(slot.observed_score) or slot.observed_score > 0.0):
        i2 = InvariantVerdict("I2", "fail", f"invalid/positive logprob {slot.observed_score} (likely unnormalized logit)")
    elif score_state == "fail":
        i2 = InvariantVerdict("I2", "fail", f"observed={slot.observed_score}, ref={slot.reference_score}")
    elif slot.genuine_top_set and slot.observed_score is not None:
        id_ref = slot.reference_id if slot.reference_id is not None else (slot.claimed_id if i1.status == "pass" else None)
        if slot.candidates and id_ref is not None:
            if id_ref in slot.candidates:
                cand_state = _float_eq(slot.observed_score, slot.candidates[id_ref], slot.abs_tol, slot.rel_tol)
                i2 = InvariantVerdict(
                    "I2",
                    cand_state,
                    f"in-top-k score check: obs={slot.observed_score}, cand={slot.candidates[id_ref]}",
                )
            elif slot.observed_score > min(slot.candidates.values()) + slot.abs_tol:
                i2 = InvariantVerdict(
                    "I2",
                    "fail",
                    f"out-of-top-k ref_id={id_ref} has obs={slot.observed_score} > min_cand={min(slot.candidates.values())}",
                )
            else:
                i2 = InvariantVerdict("I2", "pass" if score_state == "pass" else "abstain", "cutoff bound satisfied")
        elif slot.token_candidates and slot.claimed_token is not None and slot.transcript_coherent is not False:
            if slot.claimed_token in slot.token_candidates:
                cand_state = _float_eq(
                    slot.observed_score,
                    slot.token_candidates[slot.claimed_token],
                    slot.abs_tol,
                    slot.rel_tol,
                )
                i2 = InvariantVerdict(
                    "I2",
                    cand_state,
                    f"token_candidates match: obs={slot.observed_score}, cand={slot.token_candidates[slot.claimed_token]}",
                )
            elif slot.observed_score > min(slot.token_candidates.values()) + slot.abs_tol:
                i2 = InvariantVerdict(
                    "I2",
                    "fail",
                    f"unlisted token={slot.claimed_token!r} has obs={slot.observed_score} > min_cand={min(slot.token_candidates.values())}",
                )
            else:
                i2 = InvariantVerdict("I2", "pass", "token_candidates cutoff bound satisfied")
        else:
            i2 = InvariantVerdict(
                "I2",
                score_state,
                "direct reference score comparison" if score_state != "abstain" else "no score witness",
            )
    else:
        i2 = InvariantVerdict(
            "I2",
            score_state,
            "direct reference score comparison" if score_state != "abstain" else "no score witness",
        )

    # I3: Per-Item / Per-Step Coverage & Extent
    if slot.callback_elements is not None and slot.expected_callback_elements is not None:
        ok = slot.callback_elements == slot.expected_callback_elements
        i3 = InvariantVerdict(
            "I3",
            "pass" if ok else "fail",
            f"callback_elements={slot.callback_elements}, expected={slot.expected_callback_elements}",
        )
    elif slot.seen_extent != slot.expected_extent:
        i3 = InvariantVerdict("I3", "fail", f"seen_extent={slot.seen_extent}, expected={slot.expected_extent}")
    else:
        i3 = InvariantVerdict("I3", "pass", "extent matched")

    # I4: Byte-Sequence Fidelity
    if slot.observed_bytes is not None and slot.reference_bytes is not None:
        if isinstance(slot.observed_bytes, list) and isinstance(slot.reference_bytes, list):
            ok = list(slot.observed_bytes) == list(slot.reference_bytes)
        else:
            ok = slot.observed_bytes == slot.reference_bytes
        i4 = InvariantVerdict("I4", "pass" if ok else "fail", f"obs_bytes={slot.observed_bytes}, ref_bytes={slot.reference_bytes}")
    elif slot.observed_bytes is not None:
        if isinstance(slot.observed_bytes, list):
            obs = list(slot.observed_bytes)
            has_spurious_fffd = UTF8_REPLACEMENT_BYTES == obs or (
                len(obs) >= 3
                and any(obs[i : i + 3] == UTF8_REPLACEMENT_BYTES for i in range(len(obs) - 2))
                and (slot.claimed_token is None or "\ufffd" not in slot.claimed_token)
            )
            if has_spurious_fffd:
                i4 = InvariantVerdict("I4", "fail", f"spurious U+FFFD bytes {obs}")
            elif slot.bytes_coherent is False:
                i4 = InvariantVerdict("I4", "fail", "concatenated byte sequence disagrees with rendered message.content")
            elif slot.bytes_coherent is True:
                i4 = InvariantVerdict("I4", "pass", "bytes coherent with rendered message.content")
            else:
                i4 = InvariantVerdict("I4", "abstain", "no independent reference bytes")
        elif slot.bytes_coherent is False:
            i4 = InvariantVerdict("I4", "fail", "byte sequence coherence failed")
        else:
            i4 = InvariantVerdict("I4", "abstain", "no independent reference bytes")
    elif slot.bytes_coherent is False:
        i4 = InvariantVerdict("I4", "fail", "byte sequence coherence failed")
    else:
        i4 = InvariantVerdict("I4", "abstain", "bytes not exposed by interface")

    # I5: Cross-Transport Stream vs Non-Stream Consistency
    if slot.stream_has_logprobs is False:
        i5 = InvariantVerdict("I5", "fail", "streaming response omitted logprobs while non-streaming/native emitted tokens")
    elif slot.stream_extent_ok is False:
        i5 = InvariantVerdict("I5", "fail", "streaming logprob row count disagreed with non-streaming")
    else:
        witnessed = False
        diffs: list[str] = []
        if slot.stream_token is not None and slot.claimed_token is not None:
            witnessed = True
            if slot.stream_token != slot.claimed_token:
                diffs.append(f"stream_token={slot.stream_token!r}!=nonstream={slot.claimed_token!r}")
        if slot.stream_bytes is not None and slot.observed_bytes is not None:
            witnessed = True
            bytes_match = (
                list(slot.stream_bytes) == list(slot.observed_bytes)
                if isinstance(slot.stream_bytes, list) and isinstance(slot.observed_bytes, list)
                else slot.stream_bytes == slot.observed_bytes
            )
            if not bytes_match:
                diffs.append(f"stream_bytes={slot.stream_bytes}, nonstream_bytes={slot.observed_bytes}")
        if slot.stream_score is not None and slot.observed_score is not None:
            witnessed = True
            s_ok = _float_eq(slot.stream_score, slot.observed_score, slot.abs_tol, slot.rel_tol)
            if s_ok != "pass":
                diffs.append(f"stream_score={slot.stream_score}, nonstream_score={slot.observed_score}")

        if diffs:
            i5 = InvariantVerdict("I5", "fail", "; ".join(diffs))
        elif witnessed:
            i5 = InvariantVerdict("I5", "pass", "streaming and non-streaming parity verified")
        elif slot.stream_extent_ok is True:
            i5 = InvariantVerdict("I5", "abstain", "stream extent matched without per-slot score/byte witness")
        else:
            i5 = InvariantVerdict("I5", "abstain", "no paired stream witness")

    return {"I1": i1, "I2": i2, "I3": i3, "I4": i4, "I5": i5}


def summarize_slots(slots_by_request: list[list[PositionSlot]]) -> dict[str, Any]:
    """Aggregate per-position and per-request counts separately for I1-I5."""
    pos_counts = {inv: Counter() for inv in ("I1", "I2", "I3", "I4", "I5")}
    req_counts = {inv: Counter() for inv in ("I1", "I2", "I3", "I4", "I5")}
    failures: list[dict[str, Any]] = []
    for req_slots in slots_by_request:
        if not req_slots:
            continue
        req_states = {inv: [] for inv in ("I1", "I2", "I3", "I4", "I5")}
        for slot in req_slots:
            verdicts = check_slot(slot)
            for inv, v in verdicts.items():
                pos_counts[inv][v.status] += 1
                req_states[inv].append(v.status)
                if v.status == "fail" and len(failures) < 50:
                    failures.append(
                        {
                            "request_id": slot.request_id,
                            "position": slot.position,
                            "invariant": inv,
                            "detail": v.detail,
                        }
                    )
        for inv, states in req_states.items():
            overall = "fail" if "fail" in states else "abstain" if "abstain" in states else "pass"
            req_counts[inv][overall] += 1

    return {
        "requests": len(slots_by_request),
        "positions": sum(len(s) for s in slots_by_request),
        "per_position": {k: dict(v) for k, v in pos_counts.items()},
        "per_request": {k: dict(v) for k, v in req_counts.items()},
        "sample_failures": failures,
    }


def _http_post_json(url: str, payload: dict, headers: dict[str, str] | None = None, timeout: float = 45.0) -> dict:
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with HTTP_OPENER.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post_sse(url: str, payload: dict, headers: dict[str, str] | None = None, timeout: float = 45.0) -> list[dict]:
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    chunks: list[dict] = []
    with HTTP_OPENER.open(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:].strip()
            if body == "[DONE]":
                break
            if body:
                chunks.append(json.loads(body))
    return chunks


class OpenAICompatibleHTTPAdapter:
    """Live HTTP contract probe against any OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(self, base_url: str, model: str = "default_model", api_key: str | None = None):
        self.endpoint = base_url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.api_key = api_key

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        h = dict(extra or {})
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def probe_prompt(
        self,
        prompt: str,
        *,
        study_id: str,
        seed: int,
        top_k: int = 5,
        temperature: float = 0.8,
        max_tokens: int = 16,
        check_stream: bool = True,
        same_event_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Run baseline (k=0), candidate (k=top_k), and optional stream (I5) probes."""
        base_payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": 1.0,
            "top_k": 0,
            "max_tokens": max_tokens,
            "seed": seed,
            "logprobs": True,
        }

        r0 = _http_post_json(
            self.endpoint,
            {**base_payload, "top_logprobs": 0, "stream": False},
            headers=self._headers({"X-Study-Id": f"{study_id}-k0"}),
        )
        rk = _http_post_json(
            self.endpoint,
            {**base_payload, "top_logprobs": top_k, "stream": False},
            headers=self._headers({"X-Study-Id": f"{study_id}-k{top_k}"}),
        )

        sse_chunks = None
        if check_stream:
            sse_chunks = _http_post_sse(
                self.endpoint,
                {**base_payload, "top_logprobs": top_k, "stream": True},
                headers=self._headers({"X-Study-Id": f"{study_id}-stream"}),
            )

        rows0 = ((r0["choices"][0].get("logprobs") or {}).get("content")) or []
        rowsk = ((rk["choices"][0].get("logprobs") or {}).get("content")) or []
        usagek = rk.get("usage", {}).get("completion_tokens", len(rowsk))
        text0 = r0["choices"][0]["message"].get("content") or ""
        textk = rk["choices"][0]["message"].get("content") or ""
        finishk = rk["choices"][0].get("finish_reason")

        native_k = None
        if same_event_dir is not None:
            nat_path = same_event_dir / f"{study_id}-k{top_k}.native.json"
            if nat_path.exists():
                native_k = json.loads(nat_path.read_text())

        stream_rows: list[dict] = []
        stream_text = ""
        stream_usage: int | None = None
        if sse_chunks is not None:
            for ch in sse_chunks:
                usage = ch.get("usage")
                if isinstance(usage, dict) and "completion_tokens" in usage:
                    stream_usage = usage["completion_tokens"]
                choices = ch.get("choices") or []
                if not choices:
                    continue
                c0 = choices[0]
                delta = c0.get("delta") or {}
                if delta.get("content"):
                    stream_text += delta["content"]
                lp = (c0.get("logprobs") or {}).get("content") or []
                stream_rows.extend(lp)

        bytes_coherent = compute_bytes_coherence(rowsk, textk, finishk)
        transcript_coherent = compute_transcript_coherence(rowsk, textk, finishk)

        slots_k: list[PositionSlot] = []
        same_text_across_k = text0 == textk and len(rows0) == len(rowsk)
        same_text_stream = sse_chunks is not None and stream_text == textk
        stream_extent_ok: bool | None = None
        if sse_chunks is not None:
            if same_text_stream:
                stream_extent_ok = len(stream_rows) == len(rowsk)
            elif stream_usage is not None and len(stream_rows) > 0:
                stream_extent_ok = len(stream_rows) == stream_usage

        for pos in range(max(len(rowsk), usagek)):
            row = rowsk[pos] if pos < len(rowsk) else {}
            ref_id = None
            ref_score = None
            ref_token = None
            if native_k is not None and pos < len(native_k):
                ref_id = native_k[pos]["id"]
                ref_score = native_k[pos]["logprob"]
            elif same_text_across_k and pos < len(rows0):
                ref_id = rows0[pos].get("id")
                ref_score = rows0[pos].get("logprob")
                ref_token = rows0[pos].get("token")

            raw_top = row.get("top_logprobs") or []
            cands = {c["id"]: c["logprob"] for c in raw_top if "id" in c and "logprob" in c}
            tok_cand_pairs = [
                (c.get("token"), c.get("logprob"))
                for c in raw_top
                if isinstance(c.get("token"), str) and "logprob" in c
            ]
            token_cands = (
                {t: lp for t, lp in tok_cand_pairs}
                if tok_cand_pairs and len({t for t, _ in tok_cand_pairs}) == len(tok_cand_pairs)
                else None
            )

            stream_has_lp = None
            stream_token = None
            stream_score = None
            stream_bytes = None
            if sse_chunks is not None:
                if same_text_stream:
                    if pos < len(rowsk):
                        stream_has_lp = pos < len(stream_rows)
                        if stream_has_lp:
                            stream_token = stream_rows[pos].get("token")
                            stream_score = stream_rows[pos].get("logprob")
                            stream_bytes = stream_rows[pos].get("bytes")
                    else:
                        # Position omitted by both non-streaming and streaming (I3 row-omission, not I5 drift)
                        stream_has_lp = None
                else:
                    if len(stream_rows) == 0 and len(rowsk) > 0:
                        stream_has_lp = False
                    elif stream_usage is not None:
                        stream_has_lp = len(stream_rows) > 0

            slots_k.append(
                PositionSlot(
                    request_id=f"{study_id}-k{top_k}",
                    position=pos,
                    expected_extent=1,
                    seen_extent=1 if pos < len(rowsk) and row.get("logprob") is not None else 0,
                    claimed_id=row.get("id"),
                    claimed_token=row.get("token"),
                    observed_score=row.get("logprob"),
                    observed_bytes=row.get("bytes"),
                    candidates=cands or None,
                    genuine_top_set=bool(cands or token_cands),
                    reference_id=ref_id,
                    reference_score=ref_score,
                    reference_token=ref_token,
                    token_candidates=token_cands,
                    transcript_coherent=transcript_coherent,
                    bytes_coherent=bytes_coherent,
                    stream_has_logprobs=stream_has_lp,
                    stream_extent_ok=stream_extent_ok,
                    stream_token=stream_token,
                    stream_score=stream_score,
                    stream_bytes=stream_bytes,
                )
            )

        return {
            "study_id": study_id,
            "same_text_k0_vs_k": same_text_across_k,
            "same_text_stream": same_text_stream if sse_chunks is not None else None,
            "transcript_coherent": transcript_coherent,
            "bytes_coherent": bytes_coherent,
            "text_k0": text0,
            "text_k": textk,
            "stream_text": stream_text if sse_chunks is not None else None,
            "stream_logprob_rows": len(stream_rows) if sse_chunks is not None else None,
            "usage_tokens_k": usagek,
            "slots_k": slots_k,
        }


class CTranslate2NativeAdapter:
    """Build PositionSlots from CTranslate2 batch callback + final result records."""

    @staticmethod
    def slots_from_batch_rows(call_id: str, rows: list[dict[str, Any]]) -> list[PositionSlot]:
        slots: list[PositionSlot] = []
        for idx, row in enumerate(rows):
            slots.append(
                PositionSlot(
                    request_id=call_id,
                    position=idx,
                    expected_extent=1,
                    seen_extent=1 if row.get("log_prob") is not None else 0,
                    claimed_id=row.get("token_id"),
                    claimed_token=row.get("piece"),
                    observed_score=row.get("log_prob"),
                    reference_id=row.get("final_id"),
                    reference_score=row.get("final_chosen_logp"),
                    abs_tol=2e-4,
                    rel_tol=2e-5,
                    callback_elements=row.get("elements"),
                    expected_callback_elements=row.get("final_vocab"),
                )
            )
        return slots


def replay_offline_matrix(repro_dir: Path) -> dict[str, Any]:
    """Run I1-I5 checks uniformly over archived projections inside reproducibility/."""
    report: dict[str, Any] = {}

    ifeval_rows = [json.loads(line) for line in (repro_dir / "evidence/ifeval.jsonl").read_text().splitlines() if line.strip()]
    for lane, idx in (("mlx/original", 1), ("mlx/repaired", 2)):
        groups = []
        for resp in ifeval_rows:
            slots = []
            for pos, row in enumerate(resp["positions"]):
                nat_id, nat_lp = row[0]
                pub_id, pub_lp = row[idx]
                slots.append(
                    PositionSlot(
                        request_id=f"{lane}:{resp['key']}",
                        position=pos,
                        expected_extent=1,
                        seen_extent=1 if pub_lp is not None else 0,
                        claimed_id=pub_id,
                        observed_score=pub_lp,
                        candidates=dict(row[3]),
                        genuine_top_set=True,
                        reference_id=nat_id,
                        reference_score=nat_lp,
                    )
                )
            groups.append(slots)
        report[lane] = summarize_slots(groups)

    llama_records = json.loads((repro_dir / "comparison/others/llama.json").read_text())["responses"]
    for width in (1, 5):
        groups = []
        for resp in llama_records:
            if resp["width"] != width:
                continue
            slots = []
            for pos, row in enumerate(resp["positions"]):
                slots.append(
                    PositionSlot(
                        request_id=f"llama-w{width}:{resp['key']}",
                        position=pos,
                        expected_extent=1,
                        seen_extent=1 if row["returned_id"] is not None else 0,
                        claimed_id=row["returned_id"],
                        observed_score=row["returned_logp"],
                        candidates={c["id"]: c["logp"] for c in row["candidates"]},
                        reference_id=row["native_id"],
                        reference_score=row["reference_logp"],
                        abs_tol=0.002,
                        rel_tol=0.0,
                    )
                )
            groups.append(slots)
        report[f"llama.cpp/width-{width}"] = summarize_slots(groups)

    mistral_pairs = json.loads((repro_dir / "comparison/others/mistral.json").read_text())["pairs"]
    for mode in ("nonstream", "stream"):
        groups = []
        for pair in mistral_pairs:
            if pair["rank"] > 2:
                continue
            slots = []
            for pos, row in enumerate(pair["positions"]):
                score = row[f"{mode}_logp"]
                other_mode = "stream" if mode == "nonstream" else "nonstream"
                slots.append(
                    PositionSlot(
                        request_id=f"mistral-{mode}:{pair['stratum']}-{pair['source_index']}",
                        position=pos,
                        expected_extent=1,
                        seen_extent=1 if score is not None else 0,
                        observed_score=score,
                        observed_bytes=row[f"{mode}_bytes"],
                        reference_score=row["reference_logp"],
                        reference_bytes=row["canonical_bytes"],
                        abs_tol=0.02,
                        rel_tol=0.0,
                        stream_has_logprobs=True if row[f"{other_mode}_logp"] is not None else None,
                        stream_score=row["stream_logp"] if mode == "nonstream" else None,
                        stream_bytes=row["stream_bytes"] if mode == "nonstream" else None,
                    )
                )
            groups.append(slots)
        report[f"mistral.rs/{mode}"] = summarize_slots(groups)

    ct2_calls = json.loads((repro_dir / "ctranslate2/evidence.json").read_text())["calls"]
    for path in ("batch", "stream"):
        groups = []
        for call in ct2_calls:
            if call["path"] != path:
                continue
            slots = []
            for pos, row in enumerate(call["steps"]):
                vec = row["same_event"] if path == "batch" else row["forward"]
                ref = vec["selected_logit"] - vec["log_z"]
                slots.append(
                    PositionSlot(
                        request_id=f"ct2-{path}:{call['call_id']}",
                        position=pos,
                        expected_extent=1,
                        seen_extent=1 if row["log_prob"] is not None else 0,
                        claimed_id=row["token_id"],
                        observed_score=row["log_prob"],
                        reference_id=row["token_id"],
                        reference_score=ref,
                        abs_tol=2e-4,
                        rel_tol=2e-5,
                    )
                )
            groups.append(slots)
        report[f"ct2/singleton-{path}"] = summarize_slots(groups)

    ct2_distinct = repro_dir / "downstream_3b/ct2_distinct_batch.json"
    if ct2_distinct.exists():
        cd = json.loads(ct2_distinct.read_text())
        for arm in ("joint", "split", "singletons"):
            calls = [
                CTranslate2NativeAdapter.slots_from_batch_rows(call["call_id"], call["rows"])
                for call in cd["subpanel_a_audit_calls"][arm]
            ]
            report[f"ct2/{arm}-distinct-eq"] = summarize_slots(calls)

    ct2_batches = json.loads((repro_dir / "ctranslate2/batch-control.json").read_text())
    for split in (False, True):
        groups = []
        for idx, call in enumerate(ct2_batches):
            if call["copies"] != 2 or bool(call["split"]) != split:
                continue
            groups.append(
                [
                    PositionSlot(
                        request_id=f"ct2-dup-{'split' if split else 'joint'}:{idx}",
                        position=p,
                        expected_extent=1,
                        seen_extent=1 if row["log_prob"] is not None else 0,
                        claimed_id=row["token_id"],
                        observed_score=row["log_prob"],
                        reference_id=row["token_id"],
                        reference_score=row["final_chosen_logp"],
                        abs_tol=2e-4,
                        rel_tol=2e-5,
                        callback_elements=row["elements"],
                        expected_callback_elements=row["final_vocab"],
                    )
                    for p, row in enumerate(call["rows"])
                ]
            )
        report["ct2/" + ("split-dup" if split else "joint-dup")] = summarize_slots(groups)

    d24 = repro_dir / "downstream_2026_11_24"
    sg_vllm = d24 / "sglang_vllm_streaming.json"
    if sg_vllm.exists():
        sv = json.loads(sg_vllm.read_text())
        for arm_name, calls in sv.get("audit_arms", {}).items():
            groups = [
                [PositionSlot(**slot_kw) for slot_kw in call_slots]
                for call_slots in calls
            ]
            report[arm_name] = summarize_slots(groups)

    ollama_file = d24 / "ollama_live_audit.json"
    if ollama_file.exists():
        ol = json.loads(ollama_file.read_text())
        for arm_name, calls in ol.get("audit_arms", {}).items():
            groups = [
                [PositionSlot(**slot_kw) for slot_kw in call_slots]
                for call_slots in calls
            ]
            report[arm_name] = summarize_slots(groups)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    http_cmd = sub.add_parser("probe-http", help="Audit a live OpenAI-compatible HTTP server")
    http_cmd.add_argument("--base-url", required=True)
    http_cmd.add_argument("--model", default="default_model")
    http_cmd.add_argument("--top-k", type=int, default=5)
    http_cmd.add_argument("--max-tokens", type=int, default=16)
    http_cmd.add_argument("--no-stream", action="store_true")
    http_cmd.add_argument("--same-event-dir", type=Path, default=None)
    http_cmd.add_argument("--prompt", action="append", default=None)

    replay_cmd = sub.add_parser("replay-matrix", help="Run unified I1-I5 checks over reproducibility projections")
    replay_cmd.add_argument(
        "--repro-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reproducibility",
    )

    args = parser.parse_args()
    if args.cmd == "probe-http":
        prompts = args.prompt or [
            "Explain why sky appears blue in one short sentence.",
            "Write a short Python function that adds two numbers.",
            "In Vietnamese, translate 'good morning' and explain briefly.",
        ]
        adapter = OpenAICompatibleHTTPAdapter(args.base_url, model=args.model)
        all_slots = []
        details = []
        for i, p in enumerate(prompts):
            res = adapter.probe_prompt(
                p,
                study_id=f"probe-{i}",
                seed=9100 + i,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                check_stream=not args.no_stream,
                same_event_dir=args.same_event_dir,
            )
            all_slots.append(res["slots_k"])
            details.append({k: v for k, v in res.items() if k != "slots_k"})
        summary = summarize_slots(all_slots)
        print(json.dumps({"summary": summary, "details": details}, indent=2, ensure_ascii=False))
    elif args.cmd == "replay-matrix":
        print(json.dumps(replay_offline_matrix(args.repro_dir), indent=2))


if __name__ == "__main__":
    main()
