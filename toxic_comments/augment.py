import os
import pandas as pd
from typing import Optional, Iterable, List, Dict
from joblib import Parallel, delayed
from textblob import TextBlob
from textblob.translate import NotTranslated
from google_trans_new import google_translator


def translate_two_way(
    text: str, language: str, backend: str = "google"
) -> Optional[str]:
    """
    Translate input text to target language and back to english.
    Supports backends "google" and "textblob".

    Examples
    --------
    >>> translate_two_way('table', 'zh')
    'table'
    """
    if hasattr(text, "decode"):
        text = text.decode("utf-8")

    try:
        if backend == "google":
            translator = google_translator()
            trans = translator.translate(
                lang_src="en", lang_tgt=language, text=text
            )
            result = translator.translate(
                lang_tgt="en", lang_src=language, text=trans
            )

        elif backend == "textblob":
            trans = TextBlob(text)
            trans = trans.translate(to=language)
            result = str(trans.translate(to="en"))
        else:
            raise ValueError("Supported backends are: 'google', 'textblob'")
    # In case no valid translation exists, return None
    except NotTranslated:
        return None
    # In case multiple results are returned, take the first one
    if isinstance(result, list):
        result = result[0]

    return result.strip()


def translate_pavel(
    sentences: Iterable[str],
    languages: Iterable[str]=["es", "de", "fr"],
    backend: str="google",
    threads: int=1,
    verbose: bool=False
) -> Dict[str, List[str]]:
    """Translate a list of sentences to another language and  back to
    english for data augmentation. Returns a dictionary mapping language
    to the list of translated comments, in the same order as the input.

    Examples
    --------
    >>> translate_pavel(['grow old', 'play chess'])
    {'es': ['get older', 'play chess'], 'de': ['to become old', 'play chess'], 'fr': ['to get old', 'play chess']}
    """
    results = {}

    # For each languages, words are processed with multithreading
    # this helps because most of the time is spent waiting on the
    # translation server.
    parallel = Parallel(threads, backend="threading", verbose=0)
    for lang in languages:
        if verbose:
            print('Translate comments using "{0}" language'.format(lang))
        translated_data = parallel(
            delayed(translate_two_way)(text, lang, backend)
            for text in sentences
        )
        results[lang] = translated_data

    return results
