.PHONY: setup deps

deps:
	python -m pip install -r requirements.txt

setup:
	python -m pip install -e .
