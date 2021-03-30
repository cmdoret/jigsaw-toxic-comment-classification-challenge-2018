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
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import (
    Bidirectional,
    concatenate,
    Dense,
    Embedding,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    GRU,
    Input,
    LSTM,
    SpatialDropout1D,
)

IN_DIR = sys.argv[1]
OUT_DIR = sys.argv[2]
os.makedirs(OUT_DIR, exist_ok=True)

batch_size = 32
# Used epochs=100 with early exiting for best score.
epochs = 1
# Change to 10
num_folds = 3  # number of folds

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


# Summary:
# 1.The concatenated ft and tw embeddings acting as pretrained weights in an embedding layer
# 2.Spatial dropout of 50% which should reduce overfitting (?)
# 3. Bidirectional LSTM in "all to all" mode with output space of 40
# 4. Bidirectional GRU , returning both last state and all outputs
# 5. concatenation of [avg_pool(gru_output_seq), gru_last_state, max_pool(gru_output_seq), original_input]
# 6. Dense output layer, dimension of 6 for the 6 output labels


def get_model(features, clipvalue=1.0, num_filters=40, dropout=0.5):
    features_input = Input(shape=(features.shape[1],))
    inp = Input(shape=(X_train_seq.shape[1],))

    # Layer 1: concatenated fasttext and glove twitter embeddings.
    x = Embedding(
        embedding_matrix.shape[0],
        embedding_matrix.shape[1],
        weights=[embedding_matrix],
        trainable=False,
    )(inp)

    # Uncomment for best result
    # Layer 2: SpatialDropout1D(0.5)
    x = SpatialDropout1D(dropout)(x)

    # Uncomment for best result
    # Layer 3: Bidirectional CuDNNLSTM
    x = Bidirectional(LSTM(num_filters, return_sequences=True))(x)

    # Layer 4: Bidirectional CuDNNGRU
    x, x_h, x_c = Bidirectional(
        GRU(num_filters, return_sequences=True, return_state=True)
    )(x)

    # Layer 5: A concatenation of the last state, maximum pool, average pool
    # and two features: "Unique words rate" and "Rate of all-caps words"
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)

    x = concatenate([avg_pool, x_h, max_pool, features_input])

    # Layer 6: output dense layer.
    outp = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=[inp, features_input], outputs=outp)
    adam = optimizers.Adam(clipvalue=clipvalue)
    model.compile(
        loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"]
    )
    return model


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.max_score = 0
        self.not_better_count = 0

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=1)
            score = roc_auc_score(self.y_val, y_pred)
            print(
                "\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch + 1, score)
            )
            if score > self.max_score:
                print(
                    "*** New High Score (previous: %.6f) \n" % self.max_score
                )
                model.save_weights(join(OUT_DIR, "best_weights.h5"))
                self.max_score = score
                self.not_better_count = 0
            else:
                self.not_better_count += 1
                if self.not_better_count > 3:
                    print(
                        "Epoch %05d: early stopping, high score = %.6f"
                        % (epoch, self.max_score)
                    )
                    self.model.stop_training = True


model = get_model(features)


predict = np.zeros((X_test_seq.shape[0], 6))
# Uncomment for out-of-fold predictions
# scores = []
# oof_predict = np.zeros((train.shape[0],6))
kf = KFold(n_splits=num_folds, shuffle=True, random_state=239)

for train_index, test_index in kf.split(X_train_seq):

    kfold_y_train, kfold_y_test = y_train[train_index], y_train[test_index]
    kfold_X_train = X_train_seq[train_index]
    kfold_X_features = features[train_index]
    kfold_X_valid = X_train_seq[test_index]
    kfold_X_valid_features = features[test_index]

    # K.clear_session()
    model = get_model(features)

    ra_val = RocAucEvaluation(
        validation_data=(
            [kfold_X_valid, kfold_X_valid_features],
            kfold_y_test,
        ),
        interval=1,
    )

    model.fit(
        [kfold_X_train, kfold_X_features],
        kfold_y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=[ra_val],
    )

    # model.load_weights(bst_model_path)
    model.load_weights(join(OUT_DIR, "best_weights.h5"))

    predict += (
        model.predict(
            [X_test_seq, test_features], batch_size=batch_size, verbose=1
        )
        / num_folds
    )

    # gc.collect()
    # uncomment for out of fold predictions
    # oof_predict[test_index] = model.predict([kfold_X_valid, kfold_X_valid_features],batch_size=batch_size, verbose=1)
    # cv_score = roc_auc_score(kfold_y_test, oof_predict[test_index])

    # scores.append(cv_score)
    # print('score: ',cv_score)

print("Done")
# print('Total CV score is {}'.format(np.mean(scores)))
print("Validation set score")
pred_test = model.predict(
    [X_test, features_test], batch_size=batch_size, verbose=1
)
test_score = roc_auc_score(y_test, pred_test)
print(f"ROC AUC for testing set: {test_score}")


sample_submission = pd.read_csv(join(OUT_DIR, "sample_submission.csv.zip"))
class_names = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]
sample_submission[class_names] = predict
sample_submission.to_csv(
    join(OUT_DIR, "model_9872_baseline_submission.csv"), index=False
)

# uncomment for out of fold predictions
# oof = pd.DataFrame.from_dict({'id': train['id']})
# for c in class_names:
#    oof[c] = np.zeros(len(train))
#
# oof[class_names] = oof_predict
# for c in class_names:
#    oof['prediction_' +c] = oof[c]
# oof.to_csv('oof-model_9872_baseline_submission.csv', index=False)
