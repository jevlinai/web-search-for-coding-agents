# Jevlin experiments on the coding-search harness

Everything here is benchmark-side. Jevlin itself is unchanged; the arms are in
`scripts/coding_search/jevlin.py`, the public ticket tree is built by
`scripts/jevlin_bench/build_public_catalog.py`, and the per-arm summary by
`scripts/jevlin_bench/analyze.py`. The design, fairness controls and
limitations are written up in the design repository
(`docs/implementation/jevlin-openbenchmarks-2026-10-15/`).

## Arms

| arm | experiment | what the model gets |
|---|---|---|
| `perplexity_low`, `exa_fast`, `parallel_fast` | A: direct vendors | the harness's own runners, unchanged |
| `jevlin_fast` | A | tier `fast`, the account's default provider set, Jevlin's merged order |
| `jevlin_balanced` | A, and D over HTTP | tier `balanced`, default set, merged order |
| `jevlin_max` | A, only if `/statusz` shows it differs from balanced | tier `max` |
| `jevlin_single` | B-A | one provider (`JEVLIN_SINGLE_PROVIDER`, default `exa`) through Jevlin, its own order |
| `jevlin_fanout_raw` | B-B | pinned set (`JEVLIN_PROVIDERS`), round-robin, exact-URL dedup |
| `jevlin_fanout_merged` | B-C | pinned set, Jevlin's merged order |
| `jevlin_cli` | B-D | `jevlin search --stdin`: the client's own merge, account defaults |

Every arm shows the model at most `--max-results` hits (default 8) of
url / title / snippet. `--shuffle-seed` randomises the (ticket, arm) order so
no arm runs in one block of time; use a different seed per pass.

## Host preparation (Linux run server, not the router's host)

```bash
git clone https://github.com/jevlinai/web-search-for-coding-agents.git
cd web-search-for-coding-agents && git checkout jevlin-backend
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
cp .env.example .env    # fill OPENAI_API_KEY, EXA_API_KEY, PARALLEL_API_KEY, PERPLEXITY_API_KEY, JEVLIN_API_KEY
python3 scripts/jevlin_bench/build_public_catalog.py --download --out datasets
DATASETS_ROOT=datasets PYTHONPATH=scripts python -m coding_search selftest
PYTHONPATH=scripts python -m unittest coding_search.test_jevlin coding_search.test_jevlin_e2e
```

For `jevlin_cli`, the client gets its own benchmark config and the released
binary; nothing under `~/.jevlin` is used or changed:

```bash
experiments/jevlin-cli/setup.sh build ~/Github/jevlin-go v0.1.0   # -> runs/bin/jevlin
JEVLIN_API_URL=https://api.jevlin.ai experiments/jevlin-cli/setup.sh config
# append the two printed lines (JEVLIN_CLI_CONFIG=..., JEVLIN_BIN=...) to .env
```

The arm passes that config with `-config` on every call, so a `JEVLIN_CONFIG`
exported by the operator's shell cannot redirect it (python-dotenv never
overrides a variable the shell already set, which is why the arm has its own
variable). The CLI reads the key from `JEVLIN_API_KEY` in the environment, so it
searches as the same bench account as the HTTP arms; mining is disabled
through the client's own `state/mining_decision.json`.

Before any paid run, record what the server resolves each tier to (four tiny
searches):

```bash
experiments/preflight.sh
```

Set `JEVLIN_PROVIDERS` in `.env` to exactly the `balanced` list it prints, so
the B arms stay pinned to what production ran on the day. If `max` resolved to
the same set as `balanced`, drop `jevlin_max` from `ARMS`.

## Catalog corrections and freeze

The public rows carry no catalog, so the gold tokens are derived
(`build_public_catalog.py`). Reviewed corrections live in
`experiments/catalog-corrections.json` with a reason per entry and are applied
by the builder; `review_catalog.py` is the (mostly page-blocked) documentation
check that informed them. Before the pilot, freeze the tree:

```bash
.venv/bin/python scripts/jevlin_bench/build_public_catalog.py --download --out datasets
DATASETS_ROOT=datasets PYTHONPATH=scripts .venv/bin/python -m coding_search selftest
.venv/bin/python experiments/catalog_freeze.py freeze     # writes experiments/FROZEN.sha256
```

`pilot.sh` refuses to start unless the tree matches `FROZEN.sha256`.

## Smoke (paid, small): 2 tickets × every arm × 1 pass

```bash
./experiments/smoke.sh
```

## Pilot: 30 tickets × arms × 3 passes

```bash
./experiments/pilot.sh            # three passes, seeds 1 2 3, one harness invocation each
python3 scripts/jevlin_bench/analyze.py --out runs/pilot/pass-*.json \
  --catalog datasets/web-search/coding/tasks/catalog.json --rate-card experiments/rates.json --md runs/pilot/report.md
```

Passes run back to back; a ticket's repeats are therefore a full pass apart,
well beyond Jevlin's 15-minute cache. Cache hits are still recorded per search
and reported.
