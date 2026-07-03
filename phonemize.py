"""Single source of truth for text -> IPA phonemization.

Imported by BOTH the inference server (serve.py) and the training-list builder, so training
and inference feed the model an identical token alphabet. (Feeding graphemes at train time
but IPA at inference makes the model only approximate — mispronunciations, wrong pitch.)

Backend: espeak-ng via the `phonemizer` library — the SAME phonemizer the LibriTTS base
model and PL-BERT were pretrained on. Convention parity with the base model matters as much
as train/inference parity: espeak produces connected-speech forms (function words destressed,
"hello" -> həlˈoʊ), whereas dictionary phonemizers like gruut emit citation forms with a
stress mark on every word, which the model renders as a pitch accent per word (robotic).

Optional per-deployment overrides are read from PhonemeSubstitutions.json in this directory
(see PhonemeSubstitutions.example.json for the format). If the file is absent, plain espeak
phonemization is used.
"""

import re as _re
import json as _json
import os as _os

# phonemizer locates libespeak-ng via ctypes; on Homebrew macOS the dylib isn't on the
# default search path, so point at it explicitly unless the caller already has.
if 'PHONEMIZER_ESPEAK_LIBRARY' not in _os.environ:
    for _cand in ('/opt/homebrew/lib/libespeak-ng.dylib',
                  '/usr/local/lib/libespeak-ng.dylib',
                  '/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1'):
        if _os.path.exists(_cand):
            _os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = _cand
            break

from phonemizer.backend import EspeakBackend

_backend = EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)

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


def configure(path):
    """Reload substitutions/overrides from an explicit path (e.g. one supplied by the caller
    that lives outside this repo). Call before preprocess()/phonemize()."""
    global _SUBS, _OVERRIDES
    _SUBS, _OVERRIDES = _load_overrides(path)


def preprocess(text):
    """Normalise text before phonemization (spelling substitutions + trailing punctuation)."""
    for pattern, replacement in _SUBS:
        text = _re.sub(pattern, replacement, text, flags=_re.IGNORECASE)
    # Ensure trailing punctuation so the duration predictor doesn't clip the last word.
    text = text.strip()
    if text and text[-1] not in '.!?,;:':
        text += '.'
    return text


def _espeak(text):
    """Raw espeak phonemization of one text segment. Returns '' for empty/punct-only input."""
    if not text.strip():
        return ''
    return _backend.phonemize([text], strip=True)[0]


def phonemize(text):
    """Convert normalised text to an IPA phoneme string via espeak-ng.

    A word whose lowercased text is in phoneme_overrides emits that IPA string verbatim;
    the segments around it are phonemized in one espeak pass each so connected-speech
    context is preserved everywhere else.
    """
    if _OVERRIDES:
        splitter = _re.compile(r'\b(' + '|'.join(_re.escape(w) for w in _OVERRIDES) + r')\b',
                               _re.IGNORECASE)
        parts = [(_OVERRIDES[p.lower()] if p.lower() in _OVERRIDES else _espeak(p))
                 for p in splitter.split(text)]
        result = ' '.join(p for p in parts if p)
    else:
        result = _espeak(text)
    # Space punctuation out into standalone tokens (matches the training-list format) and
    # keep '|' out of the output — it is the train_list column delimiter.
    result = result.replace('|', ' ')
    result = _re.sub(r'\s*([,.;:!?…—])', r' \1', result)
    result = ' '.join(result.split())
    return result
