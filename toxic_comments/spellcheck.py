from typing import Iterable, Optional
import numpy as np
from numba import types
import numba

@numba.jit(nopython=True)
def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute the Levenshtein distance between two strings.
    Code adapted from the jellyfish (0.8.2) library:
    https://github.com/jamesturk/jellyfish

    Examples
    --------
    >>> levenshtein_distance('ABC', 'ABD')
    1
    """

    if s1 == s2:
        return 0
    rows = len(s1) + 1
    cols = len(s2) + 1

    if not s1:
        return cols - 1
    if not s2:
        return rows - 1

    prev = None
    cur = list(range(cols))
    for r in range(1, rows):
        prev, cur = cur, [r] + [0] * (cols - 1)
        for c in range(1, cols):
            deletion = prev[c] + 1
            insertion = cur[c - 1] + 1
            edit = prev[c - 1] + (0 if s1[r - 1] == s2[c - 1] else 1)
            cur[c] = min(edit, deletion, insertion)

    return cur[-1]

@numba.jit(nopython=True)
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
    # Since words are in decreasing order of priority,
    # we can stop as soon as we find a best (d=1) match
    # distances = [0] * len(wordlist)
    distances = np.zeros(len(wordlist), dtype=np.int64)
    for i, ref in enumerate(wordlist):
        lev = levenshtein_distance(word, ref)
        if lev == 1:
            return ref
        distances[i] = lev

    # Review distances and pick the closest available,
    # of best priority (first index)
    for dist in range(2, max_dist+1):
        dist_idx = np.flatnonzero(distances == dist)
        if len(dist_idx) > 0:
            return wordlist[dist_idx[0]]

    # No good match found, we return nothing
    return None


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
