from typing import Iterable, Optional
import numpy as np
from rapidfuzz import process, fuzz, string_metric


def edit_correct(
    word: str, wordlist: Iterable[str], max_dist: str = 2
) -> Optional[str]:
    """Fix spelling mistakes in input word by
    computing the levenstein distance to a list of valid words
    sorted by decreasing priority. The closest word within
    maximum edit distance is returned. Ties are resolved by
    picking the highest priority word.

    Examples
    --------
    >>> edit_correct("kug", ("mug", "bug", "but"))
    'mug'
    >>> edit_correct("bug", ("mug", "bug", "but"))
    'bug'
    >>> edit_correct("friend", ("mug", "bug", "but"))
    """
    if word in wordlist:
        return word
    # extractOne returns the most similar word, in case of
    # ties, the first word is returned. Since words are in
    # decreasing order of priority, this will automatically return
    # the most relevant word
    res = process.extractOne(word, wordlist, scorer=string_metric.levenshtein, score_cutoff=max_dist)
    # Extract string from the output (if a match was found)
    if res is not None:
        res = res[0]
    return res


def singlify(word: str, min_rep: int = 2) -> str:
    """Merge identical letters repeated at least
    min_rep times in input word.

    Examples
    --------
    >>> singlify("abba", min_rep=2)
    'aba'
    >>> singlify("abba", min_rep=3)
    'abba'
    >>> singlify("nooooooooooooooo")
    'no'
    """
    # Preallocating + setting 10% faster than append()
    singles = [""] * len(word)
    for i, letter in enumerate(word):
        if i == 0 or (letter * min_rep) != word[i - (min_rep - 1) : i + 1]:
            singles[i] = letter
    return "".join(singles)
