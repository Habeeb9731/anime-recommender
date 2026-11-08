# Each stage is one command; `make all` runs everything from raw data to reports.
PY ?= .venv/bin/python
export PYTHONPATH := src

.PHONY: setup data train banners evaluate report test app all clean

setup:
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -r requirements.txt

data:
	$(PY) scripts/prepare_data.py

train:
	$(PY) scripts/train.py

banners:
	$(PY) scripts/add_banners.py

evaluate:
	$(PY) scripts/evaluate.py

report:
	$(PY) scripts/report.py

test:
	$(PY) -m pytest -q

app:
	$(PY) -m streamlit run app/streamlit_app.py

all: test data train banners evaluate report

# Removes all derived files. Raw downloads in data/raw are kept.
clean:
	rm -rf data/processed artifacts reports/generated reports/figures reports/results.md reports/claims.md
