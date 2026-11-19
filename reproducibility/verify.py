"""Recompute the numerical checks from the public projections."""
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"


def load(path):
    return json.loads(path.read_bytes())


def records(path):
    with path.open(encoding="utf-8") as stream:
        yield from map(json.loads, stream)


def position_counts(positions):
    counts = Counter(tokens=0, wrong_id=0, wrong_logp=0, missing_from_list=0, repaired=0)
    for native, stock, patched, candidates in positions:
        candidate_by_id = dict(candidates)
        assert len(candidate_by_id) == len(candidates)
        assert stock == candidates[0]
        if native[0] in candidate_by_id:
            assert candidate_by_id[native[0]] == native[1]
        counts.update(tokens=1, wrong_id=native[0] != stock[0], wrong_logp=native[1] != stock[1],
                      missing_from_list=native[0] not in candidate_by_id, repaired=native == patched)
    return counts


def ifeval():
    total, lengths, prompts = Counter(), Counter(), []
    for row in records(EVIDENCE / "ifeval.jsonl"):
        counts = position_counts(row["positions"])
        assert all(len(x[-1]) == 5 for x in row["positions"])
        counts.update(prompts=1, affected=counts["wrong_id"] > 0)
        total.update(counts)
        lengths[str(counts["tokens"])] += 1
        prompts.append({"key": row["key"], "finish": row["finish"], **counts})
    assert dict(total=total, lengths=lengths, per_prompt=prompts) == load(EVIDENCE / "ifeval-expected.json")
    return dict(total)


def patch_http():
    expected = load(HERE / "patch/results.json")["http"]
    tally, detail = Counter(), []
    for row in records(EVIDENCE / "patch-http.jsonl"):
        tally.update(requests=1, empty_display=row["empty_display"])
        tally.update((row["route"], row["finish"]))
        spec = {key: row[key] for key in ("request", "route", "finish")}
        spec["positions"] = row["emitted"]
        if row["width"]:
            counts = position_counts(row["positions"])
            assert counts["tokens"] == row["emitted"]
            assert all(len(x[-1]) == row["width"] for x in row["positions"])
            spec.update(stock_wrong_ids=counts["wrong_id"], patched_wrong_ids=counts["tokens"] - counts["repaired"])
            tied = sum(native[0] != stock[0] and native[1] == stock[1]
                       for native, stock, _, _ in row["positions"])
            tally.update(positions=counts["tokens"], stock_wrong_ids=spec["stock_wrong_ids"],
                         patched_wrong_ids=spec["patched_wrong_ids"],
                         sampled_in_candidates=counts["tokens"] - counts["missing_from_list"],
                         sampled_outside_candidates=counts["missing_from_list"],
                         first_score_tied_with_different_id=tied)
        else:
            assert not row["positions"]
        detail.append(spec)
    assert dict(counts=dict(tally), detail=detail) == expected
    return dict(tally)


def arc():
    expected = load(EVIDENCE / "arc-expected.json")["confirmation"]
    results = Counter()
    seen = set()
    for row in records(EVIDENCE / "arc.jsonl"):
        assert row["key"] not in seen
        seen.add(row["key"])
        native, reported = ([Fraction(v) for v in row[name]] for name in ("native", "reported"))
        text, correct, parsed = (row[name] for name in ("same_text_class", "correct", "parsed"))
        native_best = {i for i, value in enumerate(native) if value == max(native)}
        reported_best = {i for i, value in enumerate(reported) if value == max(reported)}
        n, r = min(native_best), min(reported_best)
        first_parsed = next((i for i, is_parsed in enumerate(parsed) if is_parsed), 0)
        results.update(questions=1, different_text=text[n] != text[r], native_tied=len(native_best) > 1,
                       reported_tied=len(reported_best) > 1,
                       disjoint_best_text_sets={text[i] for i in native_best}.isdisjoint(text[i] for i in reported_best),
                       native_correct=correct[n], reported_correct=correct[r], first_valid_correct=correct[first_parsed],
                       native_only_correct=correct[n] and not correct[r],
                       reported_only_correct=correct[r] and not correct[n])
    rules = expected["selection_rules"]
    ranking = expected["ranking"]
    assert dict(results) == {
        "questions": expected["n_questions"], "different_text": ranking["text"],
        "native_tied": ranking["native_tied"], "reported_tied": ranking["outer_tied"],
        "disjoint_best_text_sets": ranking["disjoint_best_text_sets"],
        "native_correct": rules["native"]["correct"], "reported_correct": rules["outer"]["correct"],
        "first_valid_correct": rules["first_valid"]["correct"],
        "native_only_correct": rules["outer"]["native_only_correct"],
        "reported_only_correct": rules["outer"]["rule_only_correct"],
    }
    return dict(results)


def multilingual():
    evidence = load(EVIDENCE / "multilingual-summary.json")
    pairs = evidence["pairs"]
    tally = Counter()
    for language, counts in evidence["strata"].items():
        rows = [row for row in pairs if row["language"] == language]
        assert len(rows) == counts["aligned_pairs"] == 12
        assert len({row["index"] for row in rows}) == 12
        assert all(row["http_status"] == [200, 200] and row["target_script"] for row in rows)
        affected = sum(row["loss_positions"] > 0 for row in rows)
        loss = sum(row["loss_positions"] for row in rows)
        chars = sum(row["characters_affected"] for row in rows)
        assert (affected, loss, chars) == (
            counts["loss_affected_pairs"], counts["loss_positions"], counts["characters_affected"]
        )
        tally.update(pairs=len(rows), loss_affected_pairs=affected, loss_positions=loss, characters_affected=chars)
    assert tally["characters_affected"] == sum(evidence["completed_characters_by_script"].values())
    return dict(tally)


if __name__ == "__main__":
    print(json.dumps({"ifeval": ifeval(), "patch_http": patch_http(), "arc": arc(),
                      "multilingual": multilingual()}, indent=2))
