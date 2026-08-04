#!/usr/bin/env bash
#
# Dependency install for Claude Code cloud sessions (claude.ai/code,
# `claude --cloud`, routines).
#
# Cloud sessions start from a bare clone on an Anthropic-managed Ubuntu VM.
# Nothing from ~/.claude, the tailnet, or this machine is present — only what
# is committed here. Without this, every cloud session spends its first turns
# working out how to install the project.
#
# Wired up as a SessionStart hook in .claude/settings.json, so it also fires
# locally; the CLAUDE_CODE_REMOTE guard makes local sessions a no-op. Never
# exits non-zero: a failing hook should not take the session down with it.
set -uo pipefail

[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0

# The hook re-runs on resume within the same VM; the stamp keeps that cheap.
stamp="${TMPDIR:-/tmp}/.cloud-setup-done"
[ -f "$stamp" ] && exit 0

cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/..}" || exit 0

pip install -e ".[dev]" --quiet || true

touch "$stamp"
exit 0
