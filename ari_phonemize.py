"""Single source of truth for ARI's text -> IPA phonemization.

Imported by BOTH the inference server (serve.py / the ARI-embedded server script) and the
training-list builder, so training and inference feed the model the *same* token alphabet.
Previously training fed raw graphemes while inference fed gruut IPA, which the model could
only approximate (mispronunciations, wrong pitch). Keep this the only place phonemization
is defined.
"""

import re as _re
import gruut

# Normalise the assistant's name to a single token so gruut tokenises it as one word.
# The actual phonemes are overridden in phonemize() below.
_WORD_SUBS = [
    (r'\bA\.R\.I\.?\b', 'Ari'),
    (r'\bARI\b',        'Ari'),
]

# Target pronunciation for "Ari" — "ah-ree" as one word. Every symbol here exists in
# text_utils.py's symbol table (ˈ ɑ ː ɹ i).
_ARI_IPA = 'ˈɑːɹi'
_ARI_WORDS = {'ari', 'a.r.i', 'a.r.i.'}


def preprocess(text):
    """Normalise text before phonemization (name substitution + trailing punctuation)."""
    for pattern, replacement in _WORD_SUBS:
        text = _re.sub(pattern, replacement, text, flags=_re.IGNORECASE)
    # Ensure trailing punctuation so the duration predictor doesn't clip the last word.
    text = text.strip()
    if text and text[-1] not in '.!?,;:':
        text += '.'
    return text


def phonemize(text):
    """Convert normalised text to a space-separated IPA phoneme string via gruut."""
    words = []
    for sentence in gruut.sentences(text, lang='en-us'):
        for word in sentence:
            if (word.text or '').lower() in _ARI_WORDS:
                words.append(_ARI_IPA)
                continue
            if hasattr(word, 'phonemes') and word.phonemes:
                words.append(''.join(word.phonemes))
    return ' '.join(words)
