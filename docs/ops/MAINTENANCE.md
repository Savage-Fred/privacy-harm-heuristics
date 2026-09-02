# Scheduled maintenance Routine (issue #6)

A Claude Code Remote Routine opens a fresh cloud session once a week, checks
this repository's health from GitHub, fixes what it safely can, and reports on
the rolling `maintenance-log` issue. This file is the contract the routine
follows. The routine's stored prompt is one paragraph that points here plus
the §Hard rules, so behaviour changes by PR to this file — except the hard
rules, which must be changed in both places.

This routine replaced the automation budget that used to target
`Savage-Fred/privacy-heuristics`, which is wrapped up and maintenance-only
since 2026-09-02 (privacy-heuristics#294). This repository is a **frozen
practicum artifact**, so the routine's job is to keep it reproducible and
green, not to develop it.

| | |
|---|---|
| Routine | `privacy-harm-heuristics weekly maintenance` — <https://claude.ai/code/routines/trig_01TpPuGYm1paWdNLbXdDKU4C> |
| Schedule | cron `0 14 * * 1` (UTC — Monday 07:00 PDT / 06:00 PST), fresh session per fire, push notification on completion |
| Manage | `/schedule` skill: `RemoteTrigger {action: "get", trigger_id: "trig_01TpPuGYm1paWdNLbXdDKU4C"}`; `update` with `{"enabled": false}` pauses; `run` fires now; `list_runs` then `get_run_log` to debug. Deleting is UI-only at <https://claude.ai/code/routines>. |
| Reach | GitHub only. The repo is public, so `git clone` needs no credentials; pushing and commenting need either an authenticated `gh` or the session's GitHub MCP tools (`mcp__github__*`). If neither is available the run is read-only and reports in its final message. |

## Run procedure

Bounded, in order. Small and well-scoped beats complete.

0. **Preflight.** Confirm a GitHub read path: `gh auth status` and
   `gh api repos/Savage-Fred/privacy-harm-heuristics --jq .full_name`, or
   `mcp__github__get_me` plus a read of this repo. Record which path
   worked. If neither works, stop; nothing below can post.
1. **Orient.** Read this file, then only the last comment on the open issue
   labeled `maintenance-log`. If none is open, create
   `Maintenance log (YYYY-MM)` with that label and continue. Read
   `README.md` §"Reproducing these numbers" before touching anything under
   `data/`, `heuristics/`, or `trained_models/`.
2. **Repo health**, one call each:
   - `ci.yml` on `main`: last 3 runs must be green. A red run caused by a
     dependency resolving to a newer release is fixed by tightening the
     bound in `pyproject.toml` (the bounds are the contract; see the `dev`
     extra's comment), never by loosening a gate.
   - Release asset: `gh release view v1.0.0 --json assets` still lists
     `with_features.jsonl`; `data/CHECKSUMS.txt` still present.
   - Open PRs: a PR idle more than 7 days gets one comment, never a second.
   - Open issues: skip anything labeled `needs-owner`; do not post "stale"
     nags.
3. **Triage.** Search open issues first. One issue per finding: title as a
   sentence, evidence, a concrete acceptance line, exactly one of
   `P1`/`P2`/`P3` and one of `size:S`/`size:M`/`size:L`, plus `needs-owner`
   when only the owner can act. Comment on an existing issue rather than
   opening a twin.
4. **Fix.** `size:S` issues without `needs-owner`, highest priority first,
   following `.claude/skills/git-workflow/SKILL.md` (cloud variant: skip the
   coord lease, record the claim on the issue). One PR per issue; merge only
   on a green `ci` check; `verified` is another agent's label, never your
   own. Review tier: `/code-review max` for anything under
   `.github/workflows/`, `data/`, `trained_models/`, or `heuristics/`;
   `/code-review medium` for everything else.
5. **Report.** One comment on the open log issue: what was checked, what
   was found, what was filed or fixed (links), what is owner-blocking, and
   what the next run should look at first. A no-finding run is one short
   comment.

## Hard rules

Stored in the routine's prompt as well, so they survive a broken or missing
copy of this file. Change them in both places.

- This repository is a frozen practicum artifact: never change recorded
  results, data samples, checksums, trained models, or the headline numbers
  in `README.md`.
- Never add a paid API or spend API credits.
- Never touch `Savage-Fred/privacy-heuristics`; it is wrapped up and its
  leftovers are owner-gated (privacy-heuristics#294).
- Never merge with red CI, force-push, or rewrite `main`.
- Deploy-grade review scope as listed in step 4, without exception.
- Commit bodies end with `Bug: #N` (`Fixes #N` on the closing commit); one
  PR per issue; stop at anything labeled `needs-owner` and say so in the
  report.

## The routine's prompt (snapshot 2026-09-02; the live copy is `RemoteTrigger get`)

```
Weekly maintenance for Savage-Fred/privacy-harm-heuristics (a public repository). Fresh cloud session, no memory. If the repo is not already checked out, run `git clone https://github.com/Savage-Fred/privacy-harm-heuristics` and work inside it; then read docs/ops/MAINTENANCE.md FIRST and follow its "Run procedure" exactly — it is the contract (preflight, orient from the last maintenance-log comment, repo health, triage, fix, report). If that file is missing: preflight, a read-only health check, one comment on the open issue labeled maintenance-log (create "Maintenance log (YYYY-MM)" with that label if none is open) saying the contract doc is missing, and change nothing else. If no GitHub write path works (no gh auth, no GitHub MCP tools, push rejected), do the read-only health check and end with a one-paragraph report in your final message instead of a comment.

Hard rules that hold even if the doc changes: this repository is a frozen practicum artifact — never change recorded results, data samples, checksums, trained models, or the headline numbers in README.md; never add a paid API or spend API credits; never touch Savage-Fred/privacy-heuristics (wrapped up, maintenance-only, owner-gated). Never merge with red CI, force-push, or rewrite main. Any change under .github/workflows/, data/, trained_models/, or heuristics/ gets a /code-review max pass with findings fixed or rebutted before merge. Commit bodies end with `Bug: #N`; one PR per issue; stop at anything labeled needs-owner and say so in the report. Keep the run small: a no-finding run is one short comment on the log issue.
```
