"""Use the test set to evaluate model"""

# TODO: Write code to load model weights, run prediction on test set
# and simulate submission

pred_test = model.predict(
    [X_test, features_test], batch_size=train_params["batch_size"], verbose=1
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