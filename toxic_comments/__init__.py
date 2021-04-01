import yaml
import pathlib
import warnings

PROJECT_DIR = pathlib.Path(__file__).parents[1]

# Load  parameters yaml file into a dictionary
try:
    with open(PROJECT_DIR / "params.yaml") as fd:
        params = yaml.safe_load(fd)
except FileNotFoundError:
    warnings.warn("No params.yaml file found.")
