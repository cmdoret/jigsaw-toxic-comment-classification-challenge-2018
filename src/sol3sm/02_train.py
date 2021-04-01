# This is a reimplementation of Alexander Burmistrov's solution,
# which finished in 3rd place in the Kaggle Toxic comment
# Classification Challenge. The code is largely based on Larry Freeman's
# implementation of this solution.
# cmdoret, 202010309

import sys
import os
from os.path import join
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score

from toxic_comments.models import get_sol3sm_model, RocAucEvaluation
from toxic_comments import params


IN_DIR = sys.argv[1]
OUT_DIR = sys.argv[2]
os.makedirs(OUT_DIR, exist_ok=True)


train_params = params["training"]
model_params = params["model"]

# Load preprocessed data and embedding matrix
load_np = lambda x: np.load(join(IN_DIR, f"{x}.npy"))
embedding_matrix = load_np("embedding")
features = load_np("features")
test_features = load_np("test_features")
X_train_seq = load_np("X_train")
X_test_seq = load_np("X_test")
y_train = load_np("y_train")

# Extract test set
(
    X_train_seq,
    X_test,
    features,
    features_test,
    y_train,
    y_test,
) = train_test_split(
    X_train_seq, features, y_train, test_size=0.10, random_state=42
)
### 6. Model definition ####


model = get_sol3sm_model(
    X_train_seq, embedding_matrix, features, **model_params
)


predict = np.zeros((X_test_seq.shape[0], 6))
# Uncomment for out-of-fold predictions
# scores = []
# oof_predict = np.zeros((train.shape[0],6))
kf = KFold(n_splits=train_params["num_folds"], shuffle=True, random_state=239)

for train_index, test_index in kf.split(X_train_seq):

    kfold_y_train, kfold_y_test = y_train[train_index], y_train[test_index]
    kfold_X_train = X_train_seq[train_index]
    kfold_X_features = features[train_index]
    kfold_X_valid = X_train_seq[test_index]
    kfold_X_valid_features = features[test_index]

    # K.clear_session()
    model = get_sol3sm_model(
        X_train_seq, embedding_matrix, features, **model_params
    )

    ra_val = RocAucEvaluation(
        validation_data=(
            [kfold_X_valid, kfold_X_valid_features],
            kfold_y_test,
        ),
        weights_path=join(OUT_DIR, "best_weights.h5"),
        interval=1,
    )

    model.fit(
        [kfold_X_train, kfold_X_features],
        kfold_y_train,
        batch_size=train_params["batch_size"],
        epochs=train_params["epochs"],
        verbose=1,
        callbacks=[ra_val],
    )

    # model.load_weights(bst_model_path)
    model.load_weights(join(OUT_DIR, "best_weights.h5"))

    predict += (
        model.predict(
            [X_test_seq, test_features],
            batch_size=train_params["batch_size"],
            verbose=1,
        )
        / train_params["num_folds"]
    )

    # gc.collect()
    # uncomment for out of fold predictions
    # oof_predict[test_index] = model.predict([kfold_X_valid, kfold_X_valid_features],batch_size=batch_size, verbose=1)
    # cv_score = roc_auc_score(kfold_y_test, oof_predict[test_index])

    # scores.append(cv_score)
    # print('score: ',cv_score)

print("Done")
# print('Total CV score is {}'.format(np.mean(scores)))
