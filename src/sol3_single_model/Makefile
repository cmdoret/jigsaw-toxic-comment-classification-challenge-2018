.PHONY: preprocess train

IN="../../input/"
PRE="../../build/sol3sm/pre/"
OUT="../../build/sol3sm/out/"

preprocess:
	python ./preprocess.py $(IN) $(PRE)

train:
	python ./model.py $(PRE) $(OUT)

