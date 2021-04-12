.PHONY: setup deps test

deps:
	python -m pip install -r requirements.txt

setup:
	python -m pip install -e .

test:
	pytest --doctest-modules toxic_comments
