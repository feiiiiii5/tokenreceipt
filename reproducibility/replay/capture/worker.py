"""Start the release with an ephemeral loopback port and optional observation hooks."""
import inspect
import json
import os
from pathlib import Path
import re
import sys

from mlx_lm import server
from transformers import AutoTokenizer


def observe(handler, directory, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    parse_chat = handler.handle_chat_completions
    generate = server.ResponseGenerator.generate
    serialize = handler.generate_response
    signature = inspect.signature(serialize)

    def identify(self):
        label = self.headers.get("X-Study-Id", "")
        if not re.fullmatch(r"[0-9]+-0\.8-[0-3]", label):
            raise ValueError("Missing or invalid local capture label")
        request = parse_chat(self)
        request.study_id = label
        return request

    def sample(self, request, args, progress_callback=None):
        context, stream = generate(self, request, args, progress_callback)

        def recorded():
            rows = []
            try:
                for row in stream:
                    rows.append({"id": int(row.token), "logprob": float(row.logprob),
                                 "finish_reason": row.finish_reason})
                    yield row
            finally:
                (directory / f"{request.study_id}.native.json").write_text(json.dumps(rows))

        return context, recorded()

    def emit(self, *args, **kwargs):
        original = serialize(self, *args, **kwargs)
        if self.stream:
            return original
        bound = signature.bind(self, *args, **kwargs)
        params = {key: value for key, value in bound.arguments.items() if key != "self"}
        tokens, scores = params["tokens"], params["token_logprobs"]
        witness = [{"id": token, "logprob": score, "token": piece}
                   for token, score, piece in zip(tokens, scores, tokenizer.convert_ids_to_tokens(tokens), strict=True)]
        receipt = {"upstream": original,
                   "same_call_no_candidates": serialize(self, **{**params, "top_tokens": []}),
                   "pre_serializer": witness}
        label = self.headers["X-Study-Id"]
        (directory / f"{label}.serializer.json").write_text(json.dumps(receipt))
        return original

    handler.handle_chat_completions = identify
    handler.generate_response = emit
    server.ResponseGenerator.generate = sample


def main():
    directory = Path(os.environ["CAPTURE_OUTPUT"])
    activate = server.ThreadingHTTPServer.server_activate

    def publish_bound_port(self):
        activate(self)
        partial = directory / "port.tmp"
        partial.write_text(str(self.server_address[1]))
        partial.replace(directory / "port")

    server.ThreadingHTTPServer.server_activate = publish_bound_port
    if os.environ["CAPTURE_NATIVE"] == "1":
        observe(server.APIHandler, directory, sys.argv[sys.argv.index("--model") + 1])
    server.main()


if __name__ == "__main__":
    main()
