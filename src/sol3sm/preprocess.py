# This is a reimplementation of Alexander Burmistrov's solution,
# which finished in 3rd place in the Kaggle Toxic comment
# Classification Challenge. The code is largely based on Larry Freeman's
# implementation of this solution.
# cmdoret, 202010309

import sys
import os
import re
from os.path import join
import numpy as np
import pandas as pd
import gensim
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing import text, sequence

eng_stopwords = set(stopwords.words("english"))


IN_DIR = sys.argv[1]
OUT_DIR = sys.argv[2]

# Maximum word length on which to perform spellchecking
spellcheck_len = 20  # Fast: 0, Opti: 20
# Maximum number of words to consider
max_features = 293759  # Fast: 100k, Opti: 293759
# Maximum comment length
maxlen = 900  # Fast: 50, Opti: 900


# 1. Load and preprocess inputs
train = pd.read_csv(join(IN_DIR, "train.csv.zip"))
test = pd.read_csv(join(IN_DIR, "test.csv.zip"))

EMBEDDING_FILE_FASTTEXT = join(IN_DIR, "fasttext-crawl-300d-2M.vec")
EMBEDDING_FILE_TWITTER = join(IN_DIR, "glove.twitter.27B.200d.txt")

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
tokenizer = text.Tokenizer(max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
# Pad sequences with 0s on the left
X_train_seq = sequence.pad_sequences(X_train_seq, maxlen=maxlen)
X_test_seq = sequence.pad_sequences(X_test_seq, maxlen=maxlen)

### 3. Load pretrained embeddings ###


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype="float32")


embeddings_index_ft = dict(
    get_coefs(*o.rstrip().rsplit(" "))
    for o in open(EMBEDDING_FILE_FASTTEXT, encoding="utf-8")
)

embeddings_index_tw = dict(
    get_coefs(*o.strip().split())
    for o in open(EMBEDDING_FILE_TWITTER, encoding="utf-8")
)


spell_model = gensim.models.KeyedVectors.load_word2vec_format(
    EMBEDDING_FILE_FASTTEXT
)

# This code is  based on: Spellchecker using Word2vec by CPMP
# https://www.kaggle.com/cpmpml/spell-checker-using-word2vec

words = spell_model.index2word

### 4. Spelling correction ###

# This code is  based on: Spellchecker using Word2vec by CPMP
# https://www.kaggle.com/cpmpml/spell-checker-using-word2vec
w_rank = {}
for i, word in enumerate(words):
    w_rank[word] = i

WORDS = w_rank

# Use fast text as vocabulary
def words(text):
    return re.findall(r"\w+", text.lower())


def P(word):
    "Probability of `word`."
    # use inverse of rank as proxy
    # returns 0 if the word isn't in the dictionary
    return -WORDS.get(word, 0)


def correction(word):
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)


def candidates(word):
    "Generate possible spelling corrections for word."
    return (
        known([word]) or known(edits1(word)) or known(edits2(word)) or [word]
    )


def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)


# TODO: Use distance levenstein function to generate edits
def edits1(word):
    "All edits that are one edit away from `word`."
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


def singlify(word):
    return "".join(
        [
            letter
            for i, letter in enumerate(word)
            if i == 0 or letter != word[i - 1]
        ]
    )


# Use fast text as vocabulary

# TODO: Use distance levenstein function to generate edits
# Correct oov words by generating all words within edit distance of 1.

### 5. Combine embeddings ###
ft_dim, tw_dim = 300, 200
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
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


for word, i in word_index.items():

    if i >= max_features:
        continue

    if embeddings_index_ft.get(word) is not None:
        embed_word(embedding_matrix, i, word)
    else:
        # change to > 20 for better score.
        if len(word) > spellcheck_len:
            embedding_matrix[i] = something
        else:
            word2 = correction(word)
            if embeddings_index_ft.get(word2) is not None:
                embed_word(embedding_matrix, i, word2)
            else:
                word2 = correction(singlify(word))
                if embeddings_index_ft.get(word2) is not None:
                    embed_word(embedding_matrix, i, word2)
                else:
                    embedding_matrix[i] = something


# Save preprocessed data
os.makedirs(OUT_DIR, exist_ok=True)
np.save(join(OUT_DIR, "embedding.npy"), embedding_matrix)
np.save(join(OUT_DIR, "X_train.npy"), X_train_seq)
np.save(join(OUT_DIR, "X_test.npy"), X_test_seq)
np.save(join(OUT_DIR, "features.npy"), features)
np.save(join(OUT_DIR, "test_features.npy"), test_features)
np.save(join(OUT_DIR, "y_train.npy"), y_train)
