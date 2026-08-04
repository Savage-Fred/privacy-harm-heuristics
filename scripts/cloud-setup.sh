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
# Keep stdout silent — a SessionStart hook's stdout is injected into the
# model's context.
set -uo pipefail

[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0

root="$(cd "$(dirname "$0")/.." && pwd)" || exit 0
cd "$root" || exit 0

# One stamp per checkout, not one per VM. A single shared filename would let
# whichever repo ran first permanently silence every sibling checkout's hook on
# the same machine — agents here routinely clone more than one repo per session.
# cksum is the fallback because a missing hasher would collapse every repo back
# onto one shared stamp, silently reintroducing exactly that bug.
if command -v sha256sum >/dev/null 2>&1; then
  key="$(printf '%s' "$root" | sha256sum | cut -c1-16)"
else
  key="$(printf '%s' "$root" | cksum | tr -cd '0-9')"
fi
stamp="${TMPDIR:-/tmp}/.cloud-setup-${key:-fallback}"
[ -f "$stamp" ] && exit 0

install_deps() {
  # Ubuntu 24.04 marks the system Python externally-managed (PEP 668), so a bare
  # `pip install` dies with "externally-managed-environment". Try the clean path
  # first so a venv-based image is unaffected, then fall back. Only the second
  # attempt keeps its stderr, so a non-PEP-668 failure still reports itself.
  pip install -e ".[dev]" --quiet 2>/dev/null \
    || pip install -e ".[dev]" --quiet --break-system-packages
}

install_tools() { :; }

# Stamp only on a successful dependency install. Stamping unconditionally would
# turn one transient registry failure into a permanently dependency-less VM,
# with the resume run skipping the retry and nothing anywhere saying why.
# Optional CI tooling is best-effort: it must not force a full reinstall of the
# dependencies on every resume just because a linter download failed.
if install_deps; then
  install_tools || echo "cloud-setup: optional CI tooling did not install; repo gates that need it will fail until it does." >&2
  touch "$stamp"
else
  echo "cloud-setup: dependency install FAILED in $root — dependencies are missing, install them by hand before trusting a test run." >&2
fi
exit 0
