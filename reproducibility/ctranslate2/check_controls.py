"""Regrade the fresh normalization check, batch coverage controls and decoder trace."""
import argparse
from collections import Counter
import json
from pathlib import Path


def grade(folder):
    single = json.loads((folder / "confirmation.json").read_text())
    assert single["selected_id_matches_final"] and single["callback_vs_final_max_difference"] == 0
    assert single["positive_entries"] > 0 and single["logsumexp"] > 1
    assert single["max_raw_difference"] < 0.0002 and single["max_difference_after_normalization"] < 0.0002
    assert single["selected_difference_after_normalization"] < 0.0002
    assert abs(single["looked_up"] - single["logsumexp"] - single["selected_score"]) < 0.0002
    cases = json.loads((folder / "batch-control.json").read_text())
    totals = Counter()
    assert {(c["copies"], c["split"], c["seed"]) for c in cases} == {
        (1,0,1701),(2,0,1701),(2,1,1701),(2,0,2701),(2,1,2701)}
    for case in cases:
        group = "joint" if case["copies"] == 2 and case["split"] == 0 else "split_or_single"
        rows = case["rows"]
        assert len(rows) == len({(r["item"],r["step"]) for r in rows}) == 2 * case["copies"]
        for r in rows:
            assert r["final_count"] == 2 and r["token_id"] == r["final_id"] and r["piece"] == r["final_piece"]
            assert abs(r["log_prob"] - r["final_chosen_logp"]) < 0.0002
            assert r["final_vocab"] == 49152
            width = r["elements"] // r["final_vocab"]
            assert r["elements"] == width * r["final_vocab"]
            expected = (2 if r["item"] == 0 else 0) if group == "joint" else 1
            assert width == expected
            totals[f"{group}_vectors_of_width_{width}V"] += 1
    assert totals == {"joint_vectors_of_width_2V":4,"joint_vectors_of_width_0V":4,"split_or_single_vectors_of_width_1V":10}
    alignment = json.loads((folder / "batch-alignment.json").read_text())["runs"]
    assert len(alignment) == 2
    for case in alignment:
        assert len(case["steps"]) == 2
        for step in case["steps"]:
            first = step["first_callback"]
            second = step["second_callback"]
            assert first["batch_id"] == 0 and first["shape"] == [2, 49152]
            assert second["batch_id"] == 1 and second["elements"] == 0
            assert len(step["final_rows"]) == len(first["rows"]) == 2
            for idx, (row, own) in enumerate(zip(first["rows"], step["final_rows"])):
                assert row["sha256"] == own["sha256"] and row["index"] == own["batch_id"] == idx
                assert row["by_final_item"][idx] == {"equal": True, "max_abs_error": 0.0}
                totals["joint_callback_rows_equal_to_final_rows"] += 1
    native = json.loads((folder / "witness/decoder-inputs.json").read_text())
    response_path = folder / "witness/run-001/response.json"
    resp = json.loads(response_path.read_text())
    ids = [step["token_id"] for step in native]
    callbacks = resp["callback_rows"]
    assert len(ids) == len(callbacks) == 12
    assert ids[0] == resp["prompt_ids"][-1]
    assert ids[1:] == resp["final_sequence_ids"][:-1] == [r["token_id"] for r in callbacks[:-1]]
    assert [r["absolute_step"] for r in native] == list(range(len(resp["prompt_ids"])-1, len(resp["prompt_ids"])-1+len(ids)))
    totals["witnessed_next_ids"] = len(ids)-1
    totals["witnessed_off_first_ids"] = sum(r["rank"] > 1 for r in callbacks[:-1])
    return totals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path(__file__).resolve().parent)
    print(json.dumps(grade(parser.parse_args().directory),indent=2))
