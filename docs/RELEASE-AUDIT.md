# Release audit — v1.0.0

Before this repository was made public and tagged `v1.0.0`, it went through a
pre-release audit (internal reference: "P4"). The auditor's scope was
read-only; the notes below are a sanitized public record of what was checked,
what was found, and what was fixed. No handles, permalinks, or row contents
from the underlying corpora are reproduced here.

## Scope

Seven checks against the working tree and the full git history:

| # | Check | Initial verdict | Status now |
|---|-------|-----------------|------------|
| 1 | Secrets in working tree and git history | Pass | Pass |
| 2 | Blob sizes / Git LFS | Pass | Pass |
| 3 | Content sensitivity (privacy over-disclosure) | **Fail** | **Remediated** |
| 4 | Licensing / provenance / no vendored code | Pass | Pass |
| 5 | Stale references / infrastructure leaks | Flag | **Remediated** |
| 6 | Fresh-clone verification + CI green | **Fail** | **Remediated** |
| 7 | Release-asset readiness (checksums / fetch URL) | Pass | Pass (re-checksummed) |

The audit was a **NO-GO** on the strength of items 3 and 6; both blockers, plus
the item-5 hygiene flag, have since been remediated (see below).

## What was found, in category terms

**Item 3 — content sensitivity (the headline blocker).** Institutional
breach-reporting content (regulator filings, company disclosures, security
news) is appropriate to publish. But two of the corpora also republished
*verbatim, re-identifiable posts from private individuals* drawn from social
sources (a discussion-forum sample, a news-aggregator comment sample, and a
small number of fediverse posts). Each such row carried the poster's handle,
the full post body, and a live link back to the original — including posts
where individuals narrated sensitive personal legal, medical, or financial
situations under throwaway pseudonyms. Aggregating and redistributing that
material is precisely the secondary-use / context-collapse harm this project
studies, so it could not ship. The committed samples were a preview of a much
larger full-corpus release asset carrying the same content at scale.

**Item 5 — hygiene flags.** No infrastructure leak (no personal paths,
hostnames, IPs, or tokens). Two cosmetic issues: an internal orchestration/roadmap
file was committed, and several documentation pages still referenced the
predecessor project's Python package name in copy-pasteable import snippets
(which would raise `ImportError` against the current package).

**Item 6 — fresh-clone / CI.** On a clean clone, the test suite and CI failed
because a core runtime dependency (`pydantic`, imported at module load in the
schema layer) was only present transitively, and two optional model trainers'
tests raised hard import errors instead of skipping when their heavy optional
dependencies were absent.

## What was remediated

**Item 3 — PII / content scrub.** A pure, unit-tested scrub stage was added to
`scripts/make_samples.py` (`scrub_row`). Every published row — both committed
samples **and** the full release asset — is now run through it. The scrub
removes the entire raw source-record subobject, all identity handles and
account fields, all URLs / permalinks that re-identify a social post, all
verbatim free-text content fields, and fine-grained geography; a regex backstop
additionally drops any residual value matching an email, phone, or fediverse
handle pattern. It **keeps** exactly the non-identifiable engineered features,
harm-category labels, numeric/boolean flags, and source-type metadata that the
interpretable models actually consume. The full kept/dropped field inventory is
documented in the script's module docstring. A committed test
(`tests/test_scrub.py`) asserts the forbidden fields are absent from every row
of both committed samples (a standing CI guard), and unit-tests the scrub
function's field removal and regex backstop. Both samples and the release asset
were regenerated and re-checksummed (`data/CHECKSUMS.txt`).

**Item 5 — hygiene.** The internal roadmap file was removed and its dangling
references cleaned up. The stale package-name import snippets in the affected
documentation pages were corrected to the current package name.

**Item 6 — CI / dependencies.** `pydantic` was promoted to a core dependency;
the two optional trainers' dependencies were declared in the `models` extra;
and their tests now use `pytest.importorskip` so the suite degrades cleanly
(skips) rather than hard-failing when those optional dependencies are absent.

## Verification

- Scrub verified by exhaustive grep over both committed samples and the full
  release asset: zero social permalinks, zero user handles, zero fediverse
  handles, zero email/phone matches.
- Full test suite, `ruff`, `black --check`, and `mypy` all green.
- Release asset re-checksummed; `data/CHECKSUMS.txt` updated for the asset and
  both samples.
