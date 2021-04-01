.PHONY: preprocess train

IN="../../input/"
PRE="../../build/sol3sm/pre/"
OUT="../../build/sol3sm/out/"

preprocess:
	python ./01_preprocess.py $(IN) $(PRE)

train:
	python ./01_model.py $(PRE) $(OUT)

