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

root="$(cd "$(dirname "$0")/.." && pwd)" || exit 0
cd "$root" || exit 0

# One stamp per checkout, not one per VM. A single shared filename would let
# whichever repo ran first permanently silence every sibling checkout's hook on
# the same machine — agents here routinely clone more than one repo per session.
stamp="${TMPDIR:-/tmp}/.cloud-setup-$(printf '%s' "$root" | sha256sum | cut -c1-16)"
[ -f "$stamp" ] && exit 0

install_deps() {
  # Ubuntu 24.04 marks the system Python externally-managed (PEP 668), so a bare
  # `pip install` dies with "externally-managed-environment". Try the clean path
  # first so a venv-based image is unaffected, then fall back.
  pip install -e ".[dev]" --quiet 2>/dev/null \
    || pip install -e ".[dev]" --quiet --break-system-packages
}

# Stamp only on success. Stamping unconditionally would turn one transient
# registry failure into a permanently dependency-less VM, with the resume run
# skipping the retry and nothing anywhere saying why.
if install_deps; then
  touch "$stamp"
else
  echo "cloud-setup: dependency install FAILED in $root — dependencies are missing, install them by hand before trusting a test run." >&2
fi
exit 0
