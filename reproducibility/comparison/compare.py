"""Compare frozen observations with a common set of checks, preserving unavailable references."""
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import argparse


@dataclass
class Slot:
    expected_extent: int
    seen_extent: int
    claimed_id: int | None = None
    observed_score: float | None = None
    reference_id: int | None = None
    reference_score: float | None = None
    # None means this interface has no candidate-list input for this check.
    candidates: dict | None = None
    absolute_tolerance: float = 0.0
    relative_tolerance: float = 0.0
    observed_bytes: int | None = None
    reference_bytes: int | None = None
    genuine_top_set: bool = False
    recovery_score_witness: float | None = None


def agrees(actual, reference, absolute=0.0, relative=0.0):
    if actual is None or reference is None:
        return "abstain"
    if abs(actual - reference) <= absolute + relative * abs(reference):
        return "agree"
    return "differ"


def assess(slot):
    score = agrees(slot.observed_score, slot.reference_score, slot.absolute_tolerance, slot.relative_tolerance)
    recovered = None
    if slot.candidates is not None and slot.reference_id is not None:
        recovered = slot.candidates.get(slot.reference_id)
    bound = "abstain"
    if slot.genuine_top_set and slot.candidates and slot.reference_id is not None and slot.observed_score is not None:
        if recovered is not None:
            bound = agrees(slot.observed_score, recovered, slot.absolute_tolerance, slot.relative_tolerance)
        elif slot.observed_score > min(slot.candidates.values()):
            bound = "differ"
    # Grading an estimated lookup does not expose that witness to the bound test.
    grading_score = slot.reference_score if slot.recovery_score_witness is None else slot.recovery_score_witness
    return {
        "score_consistency_given_id": bound,
        "requested_extent": agrees(slot.seen_extent, slot.expected_extent),
        "selected_id": agrees(slot.claimed_id, slot.reference_id),
        "selected_score": score,
        "selected_bytes": agrees(slot.observed_bytes, slot.reference_bytes),
        "candidate_lookup_score_accuracy": agrees(recovered, grading_score,
                                            slot.absolute_tolerance, slot.relative_tolerance),
    }


def summarize(calls):
    positions, responses = {}, {}
    for slots in calls:
        if not slots:
            raise ValueError("An empty group needs a separate availability decision")
        outcomes = [assess(slot) for slot in slots]
        for check in outcomes[0]:
            states = [o[check] for o in outcomes]
            positions.setdefault(check, Counter()).update(states)
            # One contradiction is sufficient. An incomplete witness cannot certify the draw.
            state = "differ" if "differ" in states else "abstain" if "abstain" in states else "agree"
            responses.setdefault(check, Counter()).update([state])
    return {"positions": positions, "groups": responses}


def mlx(supplement, comparison):
    rows = [json.loads(line) for line in (supplement / "evidence/ifeval.jsonl").read_text().splitlines()]
    canonical = {
        row["key"]: row["canonical"]["stop_aware"]["classes"]
        for row in (json.loads(line) for line in (comparison / "mlx/observations.jsonl").read_text().splitlines())
    }
    for lane, index in (("original", 1), ("corrected", 2)):
        calls = []
        wire_calls = []
        for response in rows:
            wire_slots = []
            slots = []
            for offset, row in enumerate(response["positions"]):
                native_id, native_score = row[0]
                public_id, public_score = row[index]
                slots.append(Slot(1, int(public_score is not None), public_id, public_score, native_id, native_score, dict(row[3]),
                                  genuine_top_set=True))
                aligned_ids = canonical[response["key"]]
                inferred_id = aligned_ids[offset] if aligned_ids is not None else None
                wire_slots.append(Slot(1, int(public_score is not None), public_id, public_score,
                                       inferred_id, candidates=dict(row[3]), genuine_top_set=True,
                                       recovery_score_witness=native_score))
            calls.append(slots)
            wire_calls.append(wire_slots)
        yield "mlx/" + lane, calls
        yield "mlx/" + lane + "-canonical", wire_calls


def llama(comparison):
    records = json.loads((comparison / "others/llama.json").read_text())["responses"]
    for width in (1, 5):
        calls = []
        for response in records:
            if response["width"] != width:
                continue
            slots = []
            for row in response["positions"]:
                slots.append(Slot(1, int(row["returned_id"] is not None), row["returned_id"],
                                  row["returned_logp"], row["native_id"], row["reference_logp"],
                                  {c["id"]: c["logp"] for c in row["candidates"]}, absolute_tolerance=0.002))
            calls.append(slots)
        yield f"llama/width-{width}", calls


def mistral(comparison):
    pairs = json.loads((comparison / "others/mistral.json").read_text())["pairs"]
    for mode in ("nonstream", "stream"):
        groups = []
        for pair in pairs:
            if pair["rank"] > 2:
                continue
            slots = []
            for row in pair["positions"]:
                score = row[mode + "_logp"]
                slots.append(Slot(1, int(score is not None), observed_score=score,
                                  reference_score=row["reference_logp"], absolute_tolerance=0.02,
                                  observed_bytes=row[mode + "_bytes"], reference_bytes=row["canonical_bytes"]))
            groups.append(slots)
        yield "mistral/" + mode, groups


def ctranslate(supplement):
    root = supplement / "ctranslate2"
    calls = json.loads((root / "evidence.json").read_text())["calls"]
    for path in ("batch", "stream"):
        groups = []
        for call in calls:
            if call["path"] != path:
                continue
            slots = []
            for row in call["steps"]:
                # The reference is conditioned on the ID reported by the interface.
                # The next-decoder ID witness is a separate, smaller experiment.
                vector = row["same_event"] if path == "batch" else row["forward"]
                ref = vector["selected_logit"] - vector["log_z"]
                slots.append(Slot(1, int(row["log_prob"] is not None), row["token_id"], row["log_prob"], reference_score=ref,
                                  absolute_tolerance=0.0002, relative_tolerance=0.00002))
            groups.append(slots)
        yield "ct2/singleton-" + path, groups

    batches = json.loads((root / "batch-control.json").read_text())
    for split in (False, True):
        groups = []
        for call in batches:
            if call["copies"] != 2 or bool(call["split"]) != split:
                continue
            slots = [Slot(row["final_vocab"], row["elements"], row["token_id"], row["log_prob"],
                          reference_score=row["final_chosen_logp"], absolute_tolerance=0.0002,
                          relative_tolerance=0.00002) for row in call["rows"]]
            groups.append(slots)
        yield "ct2/" + ("split" if split else "joint"), groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("supplement", nargs="?", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    comparison = Path(__file__).resolve().parent
    groups = [*mlx(args.supplement, comparison), *llama(comparison), *mistral(comparison), *ctranslate(args.supplement)]
    print(json.dumps({name: summarize(calls) for name, calls in groups}, indent=2))


if __name__ == "__main__":
    main()
