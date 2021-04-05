import yaml
import pathlib
import warnings
import pandas as pd

PROJECT_DIR = pathlib.Path(__file__).parents[1]
EMBEDDING_FILE_FASTTEXT = PROJECT_DIR / "input/fasttext-crawl-300d-2M.vec"
EMBEDDING_FILE_TWITTER = PROJECT_DIR / "input/glove.twitter.27B.200d.vec"

training_set = pd.read_csv(PROJECT_DIR / "input/train.csv.zip")
X_test = pd.read_csv(PROJECT_DIR / "input/test.csv.zip")
y_test = pd.read_csv(PROJECT_DIR / "input/test_labels.csv.zip")

# Load  parameters yaml file into a dictionary
try:
    with open(PROJECT_DIR / "params.yaml") as fd:
        params = yaml.safe_load(fd)
except FileNotFoundError:
    warnings.warn("No params.yaml file found.")
