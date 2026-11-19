"""Read the sampled pair from each release's native serializer output schema."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent


def project(method, fixture, candidates):
    state = SimpleNamespace(request_id="test", system_fingerprint="test", object_type="chat.completion",
                            requested_model="test", created=0, stream=False)
    result = method(state, fixture["text"], "length", prompt_token_count=0,
                    completion_token_count=1, tokens=[fixture["native_id"]],
                    token_logprobs=[fixture["native_logprob"]], top_tokens=candidates)
    fields = result["choices"][0]["logprobs"]
    if "content" in fields:
        item, = fields["content"]
        pair = [item["id"], item["logprob"]]
        returned = item.get("top_logprobs", [])
        return "per-position", pair, returned == (candidates[0] if candidates else [])
    ids = fields["tokens"]
    scores = fields["token_logprobs"]
    token_id, = ids
    logp, = scores
    return "parallel-arrays", [token_id, logp], fields["top_logprobs"] == candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("helper", type=Path, help="Path to the accompanying minimal-regression/check.py")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("serializer_helper", args.helper)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    releases = json.loads((HERE / "releases.json").read_text())
    fixtures = json.loads((HERE / "fixtures.json").read_text())
    results = []
    for release in releases:
        record = {"version": release["version"], "cases": []}
        results.append(record)
        if "unavailable" in release:
            record["unavailable"] = release["unavailable"]
            continue
        digest = release["method_sha256"]
        fragment = (HERE / "methods" / (digest + ".txt")).read_bytes()
        if hashlib.sha256(fragment).hexdigest() != digest:
            raise ValueError("Source fragment digest differs")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "serializer.py"
            path.write_bytes(b"class APIHandler:\n" + fragment)
            method = helper.load_serializer(path)
        for fixture in fixtures:
            native = [fixture["native_id"], fixture["native_logprob"]]
            try:
                schema, plain, no_alternatives = project(method, fixture, [])
                schema_five, outer, preserved = project(method, fixture, [fixture["alternatives"]])
            except (TypeError, AttributeError, KeyError, ValueError) as exc:
                record["cases"].append({"label": fixture["label"], "unclassified": str(exc)})
                continue
            assert schema == schema_five
            record["cases"].append({"label":fixture["label"],"schema":schema,"native":native,
                                    "omitted":plain,"five":outer,"id_error":outer[0]!=native[0],
                                    "score_error":outer[1]!=native[1],"control_fields_ok":plain==native and no_alternatives and preserved})
    print(json.dumps(results, indent=2))
if __name__ == "__main__":
    main()
