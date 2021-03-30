# Use fast text as vocabulary
from typing import Iterable, Optional
from jellyfish import levenshtein_distance  # Super fast distance functions


def edit_correct(
    word: str, wordlist: Iterable[str], max_dist: str = 2
) -> Optional[str]:
    """Fix spelling mistakes in input word by
    computing the levenstein distance to a list of valid words
    sorted by decreasing priority. The highest priority word within
    maximum edit distance is returned.

    Examples
    --------
    >>> edit_correct("kug", ["mug", "bug", "but"])
    'mug'
    >>> edit_correct("bug", ["mug", "bug", "but"])
    'bug'
    >>> edit_correct("friend", ["mug", "bug", "but"])
    """
    if word in wordlist:
        return word
    # Since words are in decreasing order of priority,
    # we can stop as soon as we find a good enough match
    for ref in wordlist:
        if levenshtein_distance(word, ref) <= max_dist:
            return ref
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
