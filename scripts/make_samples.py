#!/usr/bin/env python3
"""Deterministic sampling + PII scrub of large source JSONL into committable samples.

Sources (too large to commit; live as v1.0.0 release assets or on the source
machine — see data/README.md and data/CHECKSUMS.txt):
  data/with_features.jsonl        (~211 MB, ~33.5k rows) -> data/with_features.sample.jsonl
  data/labeled_gliner2_full.jsonl (~159 MB, ~30.3k rows) -> data/labeled_gliner2.sample.jsonl

Sampling is stratified by `harm_category` so rare harm labels survive
downsampling instead of being drowned out by the dominant "unknown" bucket.
The seed is fixed so re-running this script reproduces byte-identical output
against the same source file.

--------------------------------------------------------------------------
PII / content scrub (see `scrub_row`)
--------------------------------------------------------------------------
Both corpora mix institutional breach reporting (fine to publish) with
*verbatim, re-identifiable posts from private individuals* (Reddit / Hacker
News / fediverse). Publishing the latter is the exact secondary-use /
context-collapse harm this project studies, so EVERY row written by this
script (samples AND the full release asset) is run through `scrub_row`, which
removes identity and raw-content fields and keeps only the engineered
features + labels the interpretable models actually consume.

Decision rule: the trainers (see `models/data.py`) only ever read columns with
prefixes `kw_` / `pf_` / `rc_` / `geo_us_state_` / `f_`, a handful of numeric
engineered fields, and the `harm_category` target. Free text and identity are
never features. So "when in doubt, drop" costs training nothing.

DROPPED (identity / raw content / re-identifying handles & URLs):
  - `raw`               entire nested source-record subobject (author handles,
                        selftext_html, permalinks, account objects, etc.)
  - identity handles    author, username, account_username, account_display_name,
                        account_id, account_bot, reporter (threat-feed submitter alias)
  - social record ids   submission_id, link_id, parent_id  (reconstruct URLs)
  - URLs / permalinks    url, source_url, article_url, permalink, account_url,
                        uri, xbrl_archive, logo_path, disclosureurl
  - verbatim free text  selftext, content_html, body, title, description,
                        web_description, summary, harm_summary, content_excerpt,
                        company_response, search_query
  - fine-grained geo    location, zip_code, domain, instance
  - regex backstop      any REMAINING field whose (stringified) value matches an
                        email, phone, or fediverse `@user@host` handle pattern is
                        dropped from that row (defence in depth for stragglers).

KEPT (non-identifiable features / labels / institutional + source metadata):
  - engineered features  every `kw_*`, `pf_*`, `rc_*`, `geo_us_state_*`, `f_*`
  - labels               harm_category, harm_category_source, harm_categories
                        (the free-text `harm_summary` is DROPPED — it can echo a
                        post; only the category label itself is kept)
  - structured lists     product_features, root_cause_features, causal_indicators,
                        privacy_data_exposed, privacy_keywords_matched,
                        data_classes, gliner2_labels, entities_company_candidates
  - institutional facts  entity_name, entity_type, company, has_company_mention,
                        breach_type, breach_name, filing_type, regulatory_body,
                        jurisdiction, penalty_amount, individuals_affected,
                        pwn_count, business_associate, state (2-letter, coarse),
                        issue / sub_issue / product / sub_product (CFPB fixed
                        taxonomy enums), submitted_via, timely_response,
                        consumer_disputed
  - numerics / meta      score, num_comments, created_utc, content_length,
                        content_hash, search_rank, favourites_count,
                        reblogs_count, replies_count, sentiment_score,
                        sentiment_context (categorical), sentiment_meta
  - source-type metadata source, source_category, source_name, source_type,
                        source_reliability, type, id, language, platform,
                        subreddit (topic/community, not a person), subreddit_category,
                        and the date fields (added_date, breach_date, ...)

Usage (from repo root, once the source files are present locally):
    python scripts/make_samples.py
    python scripts/make_samples.py --with-features-src /path/to/with_features.jsonl \\
        --gliner2-src /path/to/labeled_gliner2_full.jsonl
    # Scrub a full corpus (e.g. to regenerate the release asset) without
    # sampling — reads every row, scrubs it, writes all rows:
    python scripts/make_samples.py --scrub-file SRC --scrub-out OUT
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

SEED = 20260721  # fixed -- changing this changes every committed sample's contents
REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_WITH_FEATURES_SRC = REPO_ROOT / "data" / "with_features.jsonl"
WITH_FEATURES_OUT = REPO_ROOT / "data" / "with_features.sample.jsonl"
WITH_FEATURES_TARGET_ROWS = 2500  # ~10-20 MB at this corpus's ~6.3 KB/row average

DEFAULT_GLINER2_SRC = REPO_ROOT / "data" / "labeled_gliner2_full.jsonl"
GLINER2_OUT = REPO_ROOT / "data" / "labeled_gliner2.sample.jsonl"
GLINER2_FIRST_N = 150  # unstratified head, for anyone eyeballing "the first N records"
GLINER2_STRATIFIED_TARGET = 750  # + GLINER2_FIRST_N ~= 900 rows, kept under 2 MB
# A handful of gliner2 rows carry huge embedded HTML/text (p90 ~6.8 KB, max ~300 KB)
# that would blow the <2 MB budget for a ~1000-row sample; drop those outliers from
# the sampling pool rather than shrinking the row count to accommodate them.
# NB: measured on the *pre-scrub* row so the sampled set is stable regardless of
# how much the scrub later shrinks each row.
GLINER2_MAX_ROW_BYTES = 5000

STRATUM_FLOOR = 10  # minimum rows kept per non-empty harm_category, so rare labels survive

# ----------------------------------------------------------------------------
# PII / content scrub
# ----------------------------------------------------------------------------

# Fields removed from every row, unconditionally. See module docstring for the
# rationale (identity handles, raw source records, verbatim free text, URLs that
# re-identify a social post, and fine-grained geography). Trainers never read
# any of these, so removing them costs the models nothing.
DROP_FIELDS: frozenset[str] = frozenset(
    {
        # nested raw source record (carries handles, selftext_html, permalinks, ...)
        "raw",
        # identity handles / account objects
        "author",
        "username",
        "account_username",
        "account_display_name",
        "account_id",
        "account_bot",
        "reporter",  # threat-feed submitter alias (e.g. urlhaus contributor handle)
        # social record ids that reconstruct permalinks
        "submission_id",
        "link_id",
        "parent_id",
        # URLs / permalinks
        "url",
        "source_url",
        "article_url",
        "permalink",
        "account_url",
        "uri",
        "xbrl_archive",
        "logo_path",
        "disclosureurl",
        # verbatim free text (may quote a private individual's post)
        "selftext",
        "content_html",
        "body",
        "title",
        "description",
        "web_description",
        "summary",
        "harm_summary",
        "content_excerpt",
        "company_response",
        "search_query",
        # fine-grained geography / host that narrows to an individual
        "location",
        "zip_code",
        "domain",
        "instance",
    }
)

# Backstop: after DROP_FIELDS, any remaining field whose stringified value matches
# one of these is dropped from that row. Applied only to string-bearing values so
# numeric feature columns are never touched. Conservative on purpose (phone pattern
# requires phone-like grouping) to avoid nuking legitimate structured fields.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# fediverse handle: @user@instance.tld  (lemmy / mastodon style)
_FEDI_HANDLE_RE = re.compile(r"@[A-Za-z0-9_.\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# US-style phone number with separators/grouping (not a bare id/DOI run of digits)
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d{1,2}[\s.\-]?)?\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}(?!\w)")

_PII_VALUE_RES = (_EMAIL_RE, _FEDI_HANDLE_RE, _PHONE_RE)


def _value_has_pii(value: Any) -> bool:
    """True if any string anywhere inside `value` matches an email/phone/handle."""
    if isinstance(value, str):
        return any(rx.search(value) for rx in _PII_VALUE_RES)
    if isinstance(value, dict):
        return any(_value_has_pii(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_value_has_pii(v) for v in value)
    return False


def scrub_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return a new row with identity / raw-content fields removed.

    Pure function (no I/O, no mutation of the input): safe to unit-test and to
    apply to every row of every published artifact. See module docstring for the
    full kept/dropped field inventory and the decision rule.
    """
    scrubbed: dict[str, Any] = {}
    for key, value in row.items():
        if key in DROP_FIELDS:
            continue
        if _value_has_pii(value):
            # A kept field's value looks like an email/phone/fediverse handle —
            # drop it defensively rather than publish the straggler.
            continue
        scrubbed[key] = value
    return scrubbed


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_scrubbed(rows: list[dict[str, Any]], out: Path) -> None:
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(scrub_row(r), ensure_ascii=False) + "\n")


def _stratified_sample(
    rows: list[dict[str, Any]], label_key: str, target_total: int, seed: int
) -> list[dict[str, Any]]:
    """Proportional-by-label sample with a floor, deterministic given `seed`.

    Each stratum is shuffled with its own Random(seed, label) instance so the
    result doesn't depend on dict/stratum iteration order, only on `seed` and
    the source data itself.
    """
    strata: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        strata[r.get(label_key)].append(r)

    total = len(rows)
    alloc: dict[Any, int] = {}
    for label, items in strata.items():
        weight = len(items) / total
        alloc[label] = min(len(items), max(STRATUM_FLOOR, round(weight * target_total)))

    sampled: list[dict[str, Any]] = []
    for label, items in sorted(strata.items(), key=lambda kv: str(kv[0])):
        rng = random.Random(f"{seed}:{label}")
        chosen = sorted(items, key=lambda r: str(r.get("id", "")))  # stable pre-shuffle order
        rng.shuffle(chosen)
        sampled.extend(chosen[: alloc[label]])
    return sampled


def make_with_features_sample(src: Path) -> None:
    print(f"[with_features] reading {src} ...")
    rows = _load_rows(src)
    print(f"[with_features] {len(rows)} source rows")
    sample = _stratified_sample(rows, "harm_category", WITH_FEATURES_TARGET_ROWS, SEED)
    rng = random.Random(SEED)
    rng.shuffle(sample)  # de-correlate stratum grouping in the output file order
    _write_scrubbed(sample, WITH_FEATURES_OUT)  # scrub happens here
    size_mb = WITH_FEATURES_OUT.stat().st_size / 1e6
    print(
        f"[with_features] wrote {len(sample)} scrubbed rows, {size_mb:.2f} MB -> {WITH_FEATURES_OUT}"
    )


def make_gliner2_sample(src: Path) -> None:
    print(f"[gliner2] reading {src} ...")
    rows = _load_rows(src)
    print(f"[gliner2] {len(rows)} source rows")
    small_rows = [
        r for r in rows if len(json.dumps(r, ensure_ascii=False)) <= GLINER2_MAX_ROW_BYTES
    ]
    print(f"[gliner2] {len(small_rows)} rows <= {GLINER2_MAX_ROW_BYTES}B (outliers dropped)")
    first = small_rows[:GLINER2_FIRST_N]
    first_ids = {id(r) for r in first}
    remainder = [r for r in small_rows if id(r) not in first_ids]
    stratified = _stratified_sample(remainder, "harm_category", GLINER2_STRATIFIED_TARGET, SEED)
    sample = first + stratified
    _write_scrubbed(sample, GLINER2_OUT)  # scrub happens here
    size_mb = GLINER2_OUT.stat().st_size / 1e6
    print(f"[gliner2] wrote {len(sample)} scrubbed rows, {size_mb:.2f} MB -> {GLINER2_OUT}")


def scrub_file(src: Path, out: Path) -> None:
    """Scrub every row of `src` (no sampling) and stream to `out`.

    Used to regenerate the full v1.0.0 release asset (`with_features.jsonl`)
    with identity/content removed. Streams line-by-line to stay memory-flat on
    the ~200 MB corpus.
    """
    print(f"[scrub-file] {src} -> {out}")
    n = 0
    with src.open() as fin, out.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            fout.write(json.dumps(scrub_row(json.loads(line)), ensure_ascii=False) + "\n")
            n += 1
    size_mb = out.stat().st_size / 1e6
    print(f"[scrub-file] wrote {n} scrubbed rows, {size_mb:.2f} MB -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-features-src", type=Path, default=DEFAULT_WITH_FEATURES_SRC)
    parser.add_argument("--gliner2-src", type=Path, default=DEFAULT_GLINER2_SRC)
    parser.add_argument(
        "--scrub-file",
        type=Path,
        default=None,
        help="Scrub a full JSONL corpus (no sampling) and exit. Requires --scrub-out.",
    )
    parser.add_argument("--scrub-out", type=Path, default=None)
    args = parser.parse_args()

    if args.scrub_file is not None:
        if args.scrub_out is None:
            parser.error("--scrub-file requires --scrub-out")
        scrub_file(args.scrub_file, args.scrub_out)
        return

    make_with_features_sample(args.with_features_src)
    make_gliner2_sample(args.gliner2_src)


if __name__ == "__main__":
    main()
