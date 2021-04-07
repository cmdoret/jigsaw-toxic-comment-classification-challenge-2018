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
from toxic_comments.models import get_sol3sm_model
from toxic_comments.callbacks import RocAucEvaluation
from toxic_comments import params


BUILD_DIR = sys.argv[1]
os.makedirs(BUILD_DIR, exist_ok=True)


train_params = params["training"]
model_params = params["model"]

# Load preprocessed data and embedding matrix
load_np = lambda x: np.load(join(BUILD_DIR, f"{x}.npy"))
embedding_matrix = load_np("embedding")
features = load_np("features")
X_train_seq = load_np("X_train")
y_train = load_np("y_train")
X_test_seq = load_np("X_test")
features_test = load_np("features_test")

model = get_sol3sm_model(
    X_train_seq, embedding_matrix, features, **model_params
)

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
        weights_path=join(BUILD_DIR, "best_weights.h5"),
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

    model.load_weights(join(BUILD_DIR, "best_weights.h5"))


# Additional fitting with test set pseudolabels
y_test_pseudo = model.predict(
    [X_test_seq, features_test],
    batch_size=train_params["batch_size"],
    verbose=1,
)
model.fit(
    [X_test_seq, features_test],
    y_test_pseudo,
    batch_size=train_params["batch_size"],
    epochs=train_params["epochs"],
    verbose=1,
    callbacks=[ra_val],
)
model.save_weights(join(BUILD_DIR, "best_weights.h5"))

print("Done")
