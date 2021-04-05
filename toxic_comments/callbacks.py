from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import Callback


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
