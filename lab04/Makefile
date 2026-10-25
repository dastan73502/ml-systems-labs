.PHONY: all install test clean

all:
	python run.py

install:
	pip install -r requirements.txt

test:
	python -m pytest -q tests

clean:
	rm -rf data results data_card.md
