"""B4 LLM citizens: each player is a fresh headless `claude -p` call per round.

`prompts` holds the neutral rules text, the K = 3 type paragraphs of B4-calibrated, the
observation rendering and the strict answer parser (pure). `backend` runs one call
(the real CLI in an empty temporary directory, or any object with the same `complete`
coroutine). `runner` drives whole games, archives every call and writes runs.
"""
