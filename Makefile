.PHONY: venv install crawl enrich test test-cov smoke lab dashboard clean

PYTHON  := python3
VENV    := .venv
PIP     := $(VENV)/bin/pip
PYTEST  := $(VENV)/bin/pytest
JUPYTER := $(VENV)/bin/jupyter
STREAMLIT := $(VENV)/bin/streamlit

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

crawl:
	$(VENV)/bin/python -m lngraph.fetch_mempool

enrich:
	$(VENV)/bin/python -m lngraph.fetch_mempool --enrich

test:
	$(PYTEST) tests/ -v

test-cov:
	$(PYTEST) tests/ -v --cov=lngraph --cov-report=term-missing

smoke:
	$(PYTEST) tests/ -v -m live

lab:
	$(JUPYTER) lab --notebook-dir=notebooks/

dashboard:
	$(STREAMLIT) run dashboard/app.py

clean:
	rm -rf $(VENV) __pycache__ lngraph/__pycache__ tests/__pycache__ \
	       .pytest_cache .coverage
