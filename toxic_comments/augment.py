import os
import time
import tqdm
import pandas as pd
from typing import Optional, Iterable, List, Dict
from textblob import TextBlob
from textblob.translate import NotTranslated
from google_trans_new import google_translator
from google_trans_new.google_trans_new import google_new_transError
import urllib

def safe_request(fun):
    """
    Wraps function sending http requests to allow safe errors and retry.
    
    Parameters
    ----------
    fun : python function
        The python function that queries a server
    
    Returns
    -------
    wrapped_f : python function
        A wrapped version of the input function which will call itself recursively
        every 5 seconds if the server is overloaded.
    """

    def wrapped_f(*args, **kwargs):

        try:
            a = fun(*args, **kwargs)
            return a
        # google-new-trans defines their own exception over urllib...
        except (urllib.error.HTTPError, google_new_transError) as e:
            if isinstance(e, google_new_transError):
                code = e.rsp.status_code
            else:
                code = e.code
            if code in (429, 502):
                time.sleep(10)
                print("Sending too many requests, sleeping 10sec and retrying...")
                wrapped_f(*args, **kwargs)
            else:
                breakpoint()
                raise e
                
    return wrapped_f


@safe_request
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

    for lang in languages:
        translated_data = []
        if verbose:
            print('Translate comments using "{0}" language'.format(lang))
        for text in tqdm.tqdm(sentences, total=len(sentences), unit=f'texts[{lang}]'):
            translated_data.append(translate_two_way(text, lang, backend))
            # Don't spam the server to avoid IP ban
            time.sleep(0.5)
        results[lang] = translated_data

    return results
