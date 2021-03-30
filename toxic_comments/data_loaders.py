from typing import Dict
import numpy as np


def load_w2v_to_dict(path: str) -> Dict[str, np.ndarray]:
    """Load input file in word2vec format into a dictionary"""

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype="float32")

    embeddings_idx = dict(
        get_coefs(*o.rstrip().rsplit(" "))
        for o in open(path, encoding="utf-8")
    )
    return embeddings_idx