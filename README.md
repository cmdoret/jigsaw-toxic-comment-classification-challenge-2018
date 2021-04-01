# jigsaw-toxic-comment-classification-challenge-2018
Improving best solutions from an old competition for educational purpose.

## Project structure:

* src: Stores the code for the main pipeline, data extraction, processing, training, evaluation...
* notebooks: Exploratory analyses in the form of jupyter notebooks.
* toxic_comments: Boilerplate code and utilities meant to be imported as a python package and reusable in other projects.
* build: contains everything generated by us, be it temporary files, model weights, predictions, ...
* input: input files, such as training data or embedding vectors.


## Setup:

All dependencies can be installed using:
```bash
make deps
```

To make `toxic_comments` importable in python scripts and notebooks, you can run: `make setup`.

All input and output data are managed via dvc. They can be imported as follows:
```bash
pip install dvc[gdrive]
dvc pull
```

## Workflow

Code changes are managed via `git`. Data changes are managed via `dvc`, which is connected to a google drive folder.
When modifying or adding new datafiles, the modifications must be uploaded to the dvc server.
The updated small tracker file (`.dvc`) must be commited to git to keep track of changes.
The standard process is as follows:
```bash
dvc add data
dvc push
git add data.dvc
git commit -m 'added new file'
git push
```