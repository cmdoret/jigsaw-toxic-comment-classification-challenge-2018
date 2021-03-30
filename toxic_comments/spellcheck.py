# Use fast text as vocabulary
from typing import Iterable, Optional
from jellyfish import levenshtein_distance  # Super fast distance functions


def edit_correct(
    word: str, wordlist: Iterable[str], max_dist: str = 2
) -> Optional[str]:
    """Fix spelling mistakes in input word by
    computing the levenstein distance to a list of valid words
    sorted by decreasing priority. The most frequent word within
    maximum edit distance is returned.
    """
    # No need to correct if the word is valid
    if word in wordlist:
        return word
    # Since words are in decreasing order of priority,
    # we can stop as soon as we find a good enough match
    for ref in wordlist:
        if levenshtein_distance(word, ref) <= max_dist:
            return ref
    # No good match found, we return nothing
    return None


def singlify(word: str) -> str:
    """Removes duplicate letters in input word"""
    return "".join(
        [
            letter
            for i, letter in enumerate(word)
            if i == 0 or letter != word[i - 1]
        ]
    )
