"""Single source of truth for text -> IPA phonemization.

Imported by BOTH the inference server (serve.py) and the training-list builder, so training
and inference feed the model an identical token alphabet. (Feeding graphemes at train time
but IPA at inference makes the model only approximate — mispronunciations, wrong pitch.)

Optional per-deployment overrides are read from PhonemeSubstitutions.json in this directory
(see PhonemeSubstitutions.example.json for the format). If the file is absent, plain gruut
phonemization is used.
"""

import re as _re
import json as _json
import os as _os
import gruut

_SUBS_FILE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'PhonemeSubstitutions.json')


def _load_overrides(path=_SUBS_FILE):
    """Returns (substitutions, phoneme_overrides). Empty if the file is missing/invalid."""
    subs, overrides = [], {}
    try:
        with open(path, encoding='utf-8') as f:
            data = _json.load(f)
        subs = [(s['pattern'], s['replacement']) for s in data.get('substitutions', [])]
        overrides = {k.lower(): v for k, v in data.get('phoneme_overrides', {}).items()}
    except (FileNotFoundError, ValueError, KeyError):
        pass
    return subs, overrides


_SUBS, _OVERRIDES = _load_overrides()


def preprocess(text):
    """Normalise text before phonemization (spelling substitutions + trailing punctuation)."""
    for pattern, replacement in _SUBS:
        text = _re.sub(pattern, replacement, text, flags=_re.IGNORECASE)
    # Ensure trailing punctuation so the duration predictor doesn't clip the last word.
    text = text.strip()
    if text and text[-1] not in '.!?,;:':
        text += '.'
    return text


def phonemize(text):
    """Convert normalised text to a space-separated IPA phoneme string via gruut.

    A word whose lowercased text is in phoneme_overrides emits that IPA string verbatim.
    """
    words = []
    for sentence in gruut.sentences(text, lang='en-us'):
        for word in sentence:
            wt = (word.text or '').lower()
            if wt in _OVERRIDES:
                words.append(_OVERRIDES[wt])
                continue
            if hasattr(word, 'phonemes') and word.phonemes:
                words.append(''.join(word.phonemes))
    return ' '.join(words)
