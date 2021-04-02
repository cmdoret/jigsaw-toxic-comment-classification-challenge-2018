import numpy as np
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


def get_sol3sm_model(
    X: np.ndarray,
    embedding_matrix: np.ndarray,
    features: np.ndarray,
    clipvalue: float = 1.0,
    num_filters: int = 40,
    dropout: float = 0.5,
):
    """Bidirectional LSTM based single model from solution 3.
    Summary:
    1.The concatenated ft and tw embeddings acting as pretrained weights in an embedding layer
    2.Spatial dropout of 50% which should reduce overfitting (?)
    3. Bidirectional LSTM in "all to all" mode with output space of 40
    4. Bidirectional GRU , returning both last state and all outputs
    5. concatenation of [avg_pool(gru_output_seq), gru_last_state, max_pool(gru_output_seq), original_input]
    6. Dense output layer, dimension of 6 for the 6 output labels
    """
    features_input = Input(shape=(features.shape[1],))
    inp = Input(shape=(X.shape[1],))

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
    """Model callback used to compute ROC AUC during training. This
    callback also implements early stopping and saves weights periodically
    when model performance have improved.
    """

    def __init__(
        self, validation_data=(), interval=1, weights_path="best_weights.h5"
    ):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.max_score = 0
        self.not_better_count = 0
        self.weights_path = weights_path

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
                self.model.save_weights(self.weights_path)
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
