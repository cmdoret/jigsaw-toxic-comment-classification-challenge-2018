# This is a reimplementation of Alexander Burmistrov's solution,
# which finished in 3rd place in the Kaggle Toxic comment
# Classification Challenge. The code is largely based on Larry Freeman's
# implementation of this solution.
# cmdoret, 202010309

import sys
import os
import re
from os.path import join
import tqdm
import numpy as np
import pandas as pd
import gensim
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing import text, sequence

import toxic_comments.spellcheck as spck
from toxic_comments.data_loaders import load_w2v_to_dict
from toxic_comments import params

eng_stopwords = set(stopwords.words("english"))


IN_DIR = sys.argv[1]
OUT_DIR = sys.argv[2]

params = params["preprocessing"]

# 1. Load and preprocess inputs
train = pd.read_csv(join(IN_DIR, "train.csv.zip"))
test = pd.read_csv(join(IN_DIR, "test.csv.zip"))

EMBEDDING_FILE_FASTTEXT = join(IN_DIR, "fasttext-crawl-300d-2M.vec")
EMBEDDING_FILE_TWITTER = join(IN_DIR, "glove.twitter.27B.200d.txt")

print("Data loaded")

labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

# Remove special characters
special_character_removal = re.compile(
    r"[^A-Za-z\.\-\?\!\,\#\@\% ]", re.IGNORECASE
)


def get_clean_text(text: str) -> str:
    cleaned = special_character_removal.sub("", text)
    return cleaned


X_train = train["comment_text"].apply(get_clean_text)
X_test = test["comment_text"].apply(get_clean_text)

y_train = train.loc[:, labels]

### 2. Feature engineering ###

# Add prop unique words and prop capital as features
def add_features(df):
    words = df.comment_text.str.split("\S+")
    df["num_words"] = words.apply(len)
    df["caps"] = words.apply(lambda s: sum([1 for w in s if w.isupper()]))
    df["uniq"] = words.apply(lambda s: len(np.unique(s)))
    df["prop_caps"] = df["caps"] / df["num_words"]
    df["prop_uniq"] = df["uniq"] / df["num_words"]
    return df


features = add_features(train)[["prop_caps", "prop_uniq"]]
test_features = add_features(test)[["prop_caps", "prop_uniq"]]
# Standardize numeric features
ss = StandardScaler()
ss.fit(np.vstack((features, test_features)))
features = ss.transform(features)
test_features = ss.transform(test_features)
# Tokenization, each sentence becomes sequence of ints
tokenizer = text.Tokenizer(params["max_features"])
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
# Pad sequences with 0s on the left
X_train_seq = sequence.pad_sequences(
    X_train_seq, maxlen=params["max_comment_len"]
)
X_test_seq = sequence.pad_sequences(
    X_test_seq, maxlen=params["max_comment_len"]
)

print("Comments tokenized and padded")

### 3. Load pretrained embeddings ###

# Get dictionaries to map words to vectors
embeddings_index_ft = load_w2v_to_dict(EMBEDDING_FILE_FASTTEXT)
embeddings_index_tw = load_w2v_to_dict(EMBEDDING_FILE_TWITTER)

# Also use gensim to get the ordered list of words
spell_model = gensim.models.KeyedVectors.load_word2vec_format(
    EMBEDDING_FILE_FASTTEXT
)
# Words are sorted from most frequent to least frequent
words = spell_model.index_to_key

print("Embeddings loaded")


### 4. Spelling correction ###

# Use fast text as vocabulary

# TODO: Use distance levenstein function to generate edits
# Correct oov words by generating all words within edit distance of 1.

# 5. Combine embeddings #
ft_dim, tw_dim = 300, 200
word_index = tokenizer.word_index
nb_words = min(params["max_features"], len(word_index))
embedding_matrix = np.zeros((nb_words, ft_dim + tw_dim + 1))

something = np.zeros((ft_dim + tw_dim + 1,))
something[
    :ft_dim,
] = embeddings_index_ft["something"]
something[
    ft_dim:-1,
] = embeddings_index_tw["something"]


# Fasttext vector is used by itself if there is no glove vector but
# not the other way around.
# Summary:
# * If the word has a fasttext + glove embeddings, we use their concatenation
# * If it only has a fasttext embedding, we use it
# * Otherwise, we attempt to correct the word with edits
# * If no word with embedding is found withing 2 edits or the word is too
#   long to correct it in reasonable time, the embeddings of word "something" is used
def all_caps(word):
    return len(word) > 1 and word.isupper()


def embed_word(embedding_matrix, i, word):
    embedding_vector_ft = embeddings_index_ft.get(word)
    if embedding_vector_ft is not None:
        if all_caps(word):
            last_value = np.array([1])
        else:
            last_value = np.array([0])
        embedding_matrix[i, :ft_dim] = embedding_vector_ft
        embedding_matrix[i, -1] = last_value
        embedding_vector_tw = embeddings_index_tw.get(word)
        if embedding_vector_tw is not None:
            embedding_matrix[i, ft_dim:-1] = embedding_vector_tw


for word, i in tqdm.tqdm(
    word_index.items(), total=params["max_features"], unit="words"
):

    if i >= params["max_features"]:
        continue

    if embeddings_index_ft.get(word) is not None:
        embed_word(embedding_matrix, i, word)
    else:
        # change to > 20 for better score.
        if len(word) > params["max_spellcheck_len"]:
            embedding_matrix[i] = something
        else:
            word2 = spck.edit_correct(word, words, max_dist=2)
            if word2 is None:
                word2 = spck.edit_correct(
                    spck.singlify(word, min_rep=2), words, max_dist=2
                )
                if word2 is None:
                    embedding_matrix[i] = something
                else:
                    embed_word(embedding_matrix, i, word2)
            else:
                embed_word(embedding_matrix, i, word2)

print("Embedding matrix complete")

# Save preprocessed data
os.makedirs(OUT_DIR, exist_ok=True)
np.save(join(OUT_DIR, "embedding.npy"), embedding_matrix)
np.save(join(OUT_DIR, "X_train.npy"), X_train_seq)
np.save(join(OUT_DIR, "X_test.npy"), X_test_seq)
np.save(join(OUT_DIR, "features.npy"), features)
np.save(join(OUT_DIR, "test_features.npy"), test_features)
np.save(join(OUT_DIR, "y_train.npy"), y_train)
