.PHONY: setup train reproduce reproduce-full fetch-data test lint clean

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

RELEASE_URL := https://github.com/Savage-Fred/privacy-harm-heuristics/releases/download/v1.0.0/with_features.jsonl
WITH_FEATURES := data/with_features.jsonl
CHECKSUMS := data/CHECKSUMS.txt

## setup: create a venv and install the package with dev+models extras.
setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev,models]"

## train: retrain the five interpretable models on the committed feature sample.
train:
	$(PY) -m privacy_harm_heuristics.cli train-all

## reproduce: run the offline headline comparison and check rules_static against
## the recorded results in data/experiments/final_results_summary.md.
## Unsets provider API keys so the run stays on the deterministic offline
## fallback (see cli.py's reproduce docstring for why rules_static may still
## legitimately mismatch -- that's a documented, structural finding, not a bug).
reproduce:
	env -u OPENAI_API_KEY -u GEMINI_API_KEY -u GOOGLE_API_KEY \
		$(PY) -m privacy_harm_heuristics.cli reproduce --check

## reproduce-full: same as reproduce, but trains against the full corpus
## (data/with_features.jsonl) instead of the committed sample, if present.
## Run `make fetch-data` first to obtain it.
reproduce-full:
	@if [ ! -f "$(WITH_FEATURES)" ]; then \
		echo "reproduce-full: $(WITH_FEATURES) not found. Run 'make fetch-data' first."; \
		exit 1; \
	fi
	env -u OPENAI_API_KEY -u GEMINI_API_KEY -u GOOGLE_API_KEY \
		$(PY) -m privacy_harm_heuristics.cli reproduce --check
	$(PY) -m privacy_harm_heuristics.cli train-all --data $(WITH_FEATURES)

## fetch-data: download the v1.0.0 release asset (full feature corpus) and
## verify it against data/CHECKSUMS.txt. Placeholder until the v1.0.0 tag/release
## exists (P4) -- fails with a clear message until then.
fetch-data:
	@echo "Fetching $(WITH_FEATURES) from $(RELEASE_URL)..."
	@curl -fL --retry 3 -o $(WITH_FEATURES).tmp "$(RELEASE_URL)" || { \
		echo "fetch-data: download failed. The v1.0.0 release asset may not be published yet"; \
		echo "(see the v1.0.0 release notes). Falling back to the committed sample is fine for 'make reproduce';"; \
		echo "this target is only needed for 'make reproduce-full' / full-corpus training."; \
		rm -f $(WITH_FEATURES).tmp; \
		exit 1; \
	}
	@expected=$$(grep -F "$(WITH_FEATURES)" $(CHECKSUMS) | awk '{print $$1}' | head -1); \
	if [ -z "$$expected" ]; then \
		echo "fetch-data: no checksum entry for $(WITH_FEATURES) in $(CHECKSUMS); aborting."; \
		rm -f $(WITH_FEATURES).tmp; \
		exit 1; \
	fi; \
	actual=$$(sha256sum $(WITH_FEATURES).tmp | awk '{print $$1}'); \
	if [ "$$expected" != "$$actual" ]; then \
		echo "fetch-data: checksum mismatch for $(WITH_FEATURES)."; \
		echo "  expected: $$expected"; \
		echo "  actual:   $$actual"; \
		rm -f $(WITH_FEATURES).tmp; \
		exit 1; \
	fi; \
	mv $(WITH_FEATURES).tmp $(WITH_FEATURES); \
	echo "fetch-data: verified checksum, wrote $(WITH_FEATURES)."

## test: run the test suite.
test:
	$(PY) -m pytest

## lint: ruff + black --check + mypy.
lint:
	$(PY) -m ruff check src/ tests/
	$(PY) -m black --check src/ tests/
	$(PY) -m mypy src/

clean:
	rm -rf results/
