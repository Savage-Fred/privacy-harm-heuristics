---
name: git-workflow
description: The house git rules for all Savage-Fred project repos — branch naming, the stable/main promotion model, bug-referenced commits, commit cadence, session-branch hygiene, and multi-agent lease etiquette. Use before creating a branch, committing, merging, or cleaning up branches/worktrees in any project repo.
---

# /git-workflow — house git rules

This skill is the operational digest. Applies to agentic-privacy,
privacy-crawler, privacy-heuristics, ai-context, ai-config, paper-builder.

Canonical policy is `docs/BRANCHING.md` and `docs/WORK-TRACKING.md` **in the
agentic-privacy repo**. Those files are not copied into the other repos, so from
any other checkout — and from a cloud session, which has only the one repo — this
digest is the operative text.

## The branch model

- **`stable`** — known good. Never commit to it. It moves only by
  fast-forward promotion from `main` via the repo's `promote-stable`
  workflow (full test suite, slow tests included, all repo gates).
- **`main`** — integration trunk (default). Land work via PR with green CI.
- **Work branches** — short-lived, single-purpose, deleted on merge.

## Before you start work

1. **Find or file the tracking issue.** No branch without an issue.
   Self-assign: add the `status:assigned` label and comment
   `<provider>@<host> taking this; plan: <one line>`.
2. **For write work, take the coordination lease** — local/tailnet sessions only.
   `~/ai-context/bin/coord claim …` (protocol:
   `~/ai-context/shared/coordination/PROTOCOL.md`). The issue is the record; the
   lease is the mutex. Before claiming, check `coord status <project>` and the
   go/dash branches tab for anyone already on the same paths.

   In a Claude Code cloud session none of `~/ai-context`, `coord`, or go/dash
   exists. Skip this step entirely and record your claim in the issue comment.
3. **Branch from `main`** (hotfixes from `stable`), named:

   ```
   <type>/<issue#>-<plain-english-slug>
   ```

   `type` ∈ `feature` `fix` `docs` `experiment` `chore` `integration`
   `hotfix`. Slug = 3–6 lowercase words a human reads cold.
   Example: `feature/231-branch-graph-dashboard-tab`.

## While you work

- **Commit + push at least every 60 minutes** of active work and at every
  coherent checkpoint. Unpushed work on any host is invisible to the rest of
  the fleet and unrecoverable if the host dies.
- **Every commit body ends with a bug reference**: `Bug: #42`
  (closing commit: `Fixes #42`; cross-repo:
  `Bug: Savage-Fred/<repo>#42`). Repos enforce this via a commit-msg hook
  (`make hooks`) and CI.
- **After each push, comment on the issue in plain English** — what changed,
  what's next. One or two sentences, written for Will, not a log dump.
- Never force-push shared branches; never rewrite `main`/`stable`; never
  share a writable checkout with another agent — one worktree per live
  branch.

## Finishing

1. PR into `main`; squash-merge is the default. Let the merge auto-close the
   issue via `Fixes #N`.
2. Delete the head branch (should be automatic; if not, delete it yourself)
   and remove its worktree.
3. **Do not verify your own fix.** Another agent re-checks the landed result
   (integrations green, artifact actually works) and adds the `verified`
   label with a comment saying what was checked.
4. Session branches (`claude/*`, `codex/*`, `gemini/*`, `copilot/*`) are
   scratch: within 48h of session end, promote the work to a PR or delete
   the ref.

## Cleanup crib sheet

```bash
git fetch --prune
git branch -r --merged origin/main            # remote branches safe to delete
git cherry origin/main <branch>               # "-" lines = already on main
git push origin --delete <branch>             # after the check above
git worktree list; git worktree remove <path> # worktrees follow branches
git tag archive/<name> <branch> && git push origin archive/<name>  # keep-as-tag
```

Branches that exist only to preserve an artifact become `archive/*` tags, not
branches. Experiments idle >14 days get a promote-or-delete call.
