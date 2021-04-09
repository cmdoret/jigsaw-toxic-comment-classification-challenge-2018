"""Use the test set to evaluate model"""
import sys
from os.path import join
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from toxic_comments.models import get_sol3sm_model, RocAucEvaluation
from toxic_comments import params, y_test


BUILD_DIR = sys.argv[1]

load_np = lambda x: np.load(join(BUILD_DIR, f"{x}.npy"))
X_test_seq = load_np("X_test")
features_test = load_np("features_test")
embedding_matrix = load_np("embedding")

model_params = params["model"]
train_params = params["training"]

model = get_sol3sm_model(
    X_test_seq, embedding_matrix, features_test, **model_params
)
pred_test = model.predict(
    [X_test_seq, features_test],
    batch_size=train_params["batch_size"],
    verbose=1,
)
test_score = roc_auc_score(y_test, pred_test)
print(f"ROC AUC for testing set: {test_score}")


sample_submission = pd.read_csv(join(BUILD_DIR, "sample_submission.csv.zip"))
class_names = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]
sample_submission[class_names] = pred_test
sample_submission.to_csv(
    join(BUILD_DIR, "model_9872_baseline_submission.csv"), index=False
)
