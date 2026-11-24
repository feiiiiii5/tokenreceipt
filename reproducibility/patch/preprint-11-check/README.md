# Preprint patch revision (11 checks)

`mlx-lm-0.32.2.patch` is the 11-check revision evaluated in the November 23 preprint and in the November 23 live audits in [`../../downstream_3b/`](../../downstream_3b/). It corrects non-streaming selected-token attribution (`I1`/`I2`) but does not change streaming responses.

- Patch SHA-256: `b30b3433280b4cc7f00a9288f6f3af61c75cd88f5319ed6aa6da7b86ba0de6b4`
- Patched `server.py` SHA-256: `b6770cd13dc4791a7fa0002325e64481fde80627a1679501d6b740111003bd24`

`python3 -B ../../verify_2026_11_24.py` applies this file to the pinned stock module and checks that the result matches the patched-server hash recorded by those audits. The current patch in the parent directory supersedes it: it adds four streaming checks and passes all 15 offline checks.
