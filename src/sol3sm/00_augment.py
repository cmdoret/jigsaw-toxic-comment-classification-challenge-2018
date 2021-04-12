import sys
import numpy as np
import toxic_comments as toxic
import toxic_comments.augment as aug

BUILD_DIR = sys.argv[1]

# 1. Load and preprocess inputs
train = toxic.training_set
test = toxic.X_test

# Data augmentation: Translate each comment back and forth
# in 4 languages to generate new samples

def augment_df(df):
    """Augment input comments by translating into multiple
    languages. Original dataframe is replicated for each
    language.
    """
    df_aug = []
    # Multiway translation of input text
    aug_text = aug.translate_pavel(df.comment_text)

    # Copy input dataframe for each language, replacing text column
    for lang, text in train_aug.items():
        df_aug.append(df.copy())
        df_aug[-1].comment_text = text
    df_aug = pd.concat(df_aug)

    return df_aug

# train time augmentation
train = augment_df(train)
# test time augmentation
test = augment_df(test)

# Save augmented sets
os.makedirs(BUILD_DIR, exist_ok=True)
np.save(join(BUILD_DIR, "train_augmented.npy"), embedding_matrix)
np.save(join(BUILD_DIR, "test_augmented.npy"), embedding_matrix)
