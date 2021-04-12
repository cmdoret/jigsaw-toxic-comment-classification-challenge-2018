.PHONY: augment preprocess train evaluate

BUILD_DIR="../../build/sol3sm/build"

augment:
	python ./00_augment.py $(BUILD_DIR)

preprocess:
	python ./01_preprocess.py $(BUILD_DIR)

train:
	python ./02_train.py $(BUILD_DIR)

evaluate:
	python ./03_evaluate.py $(BUILD_DIR)
