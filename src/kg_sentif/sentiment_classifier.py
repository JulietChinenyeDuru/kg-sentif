"""Aspect-level sentiment classification module.

Assigns a sentiment polarity (positive, negative, neutral) to each extracted
aspect individually, based on sentiment-bearing words found in its local
context window within the sentence.

This module uses a transparent lexicon-based approach by default so the
pipeline runs without any external API or GPU dependency. It is designed to
be swapped out for a transformer-based classifier (e.g. a fine-tuned
DeBERTa or RoBERTa ABSA model) via the `SentimentModel` interface below.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
import re


# Words that typically introduce a contrasting clause. Sentiment scoring is
# scoped to the clause containing the aspect so that a positive word in one
# clause doesn't leak into the score of an aspect in another clause (e.g.
# "camera is great but battery is bad").
CLAUSE_BOUNDARIES = {"but", "however", "although", "though", "while", "yet"}


# A compact, illustrative sentiment lexicon. For production use, this should
# be replaced with a larger validated lexicon (e.g. SentiWordNet, VADER) or
# a fine-tuned transformer classifier via the SentimentModel interface.
POSITIVE_WORDS = {
    "excellent", "great", "good", "amazing", "love", "loved", "fantastic",
    "impressive", "smooth", "fast", "reliable", "beautiful", "comfortable",
    "responsive", "outstanding", "superb", "solid", "crisp", "sharp",
}

NEGATIVE_WORDS = {
    "disappointing", "poor", "bad", "terrible", "slow", "awful", "broken",
    "unreliable", "frustrating", "weak", "faulty", "laggy", "cheap",
    "flimsy", "annoying", "worst", "horrible", "buggy",
}

NEGATIONS = {"not", "no", "never", "hardly", "barely", "n't"}


@dataclass
class AspectSentiment:
    """Represents the sentiment result for a single aspect.

    Attributes:
        aspect (str): The aspect term being scored.
        sentiment (str): One of "positive", "negative", or "neutral".
        confidence (float): A heuristic confidence score in [0.0, 1.0].
    """

    aspect: str
    sentiment: str
    confidence: float


class SentimentModel:
    """Interface for pluggable sentiment scoring backends.

    Implement `score_window` to integrate a transformer-based or other
    machine-learned sentiment classifier in place of the default lexicon
    approach used by `LexiconSentimentClassifier`.
    """

    def score_window(self, context_window: str) -> Tuple[str, float]:
        """Scores a local text window for sentiment polarity.

        Args:
            context_window (str): A short span of text surrounding an
                aspect mention.

        Returns:
            Tuple[str, float]: A (sentiment_label, confidence) pair, where
                sentiment_label is one of "positive", "negative", "neutral".
        """
        raise NotImplementedError


class LexiconSentimentClassifier(SentimentModel):
    """Default lexicon-based aspect sentiment classifier.

    Scores each aspect by inspecting a fixed-size word window around its
    mention in the source text, counting positive/negative lexicon hits,
    and applying simple negation handling.

    Attributes:
        window_size (int): Number of words to look at on either side of the
            aspect mention when scoring sentiment.
    """

    def __init__(self, window_size: int = 6):
        """Initializes the classifier.

        Args:
            window_size (int): Number of words of context to consider on
                each side of an aspect mention.
        """
        self.window_size = window_size

    def classify(self, text: str, aspects: List[str]) -> List[AspectSentiment]:
        """Classifies sentiment for each aspect found in the text.

        Args:
            text (str): The original source text.
            aspects (List[str]): Aspect terms previously extracted from
                `text` (see `AspectExtractor.extract`).

        Returns:
            List[AspectSentiment]: One sentiment result per aspect, in the
                same order as the input `aspects` list.
        """
        tokens = self._tokenize(text)
        results: List[AspectSentiment] = []

        for aspect in aspects:
            window = self._context_window(tokens, aspect)
            sentiment, confidence = self.score_window(window)
            results.append(AspectSentiment(aspect=aspect, sentiment=sentiment, confidence=confidence))

        return results

    def score_window(self, context_window: str) -> Tuple[str, float]:
        """Scores a context window using lexicon lookups and negation handling.

        Negation is detected by looking back up to 3 words from a sentiment
        word (to catch patterns like "not a great experience"), rather than
        only the immediately preceding word.

        Args:
            context_window (str): Words surrounding an aspect mention.

        Returns:
            Tuple[str, float]: (sentiment_label, confidence).
        """
        words = context_window.lower().split()
        pos_hits, neg_hits = 0, 0
        negation_lookback = 3

        for i, word in enumerate(words):
            lookback_start = max(0, i - negation_lookback)
            negated = any(w in NEGATIONS for w in words[lookback_start:i])

            if word in POSITIVE_WORDS:
                neg_hits += 1 if negated else 0
                pos_hits += 0 if negated else 1
            elif word in NEGATIVE_WORDS:
                pos_hits += 1 if negated else 0
                neg_hits += 0 if negated else 1

        total_hits = pos_hits + neg_hits
        if total_hits == 0:
            return "neutral", 0.5

        if pos_hits > neg_hits:
            confidence = 0.5 + 0.5 * (pos_hits / total_hits)
            return "positive", round(min(confidence, 0.99), 2)
        elif neg_hits > pos_hits:
            confidence = 0.5 + 0.5 * (neg_hits / total_hits)
            return "negative", round(min(confidence, 0.99), 2)
        else:
            return "neutral", 0.5

    def _tokenize(self, text: str) -> List[str]:
        """Splits text into lowercase word tokens, preserving contractions.

        Args:
            text (str): Raw input text.

        Returns:
            List[str]: List of word tokens, with clause-boundary punctuation
                (periods, commas, semicolons) preserved as separate "|"
                markers so clause boundaries survive tokenization.
        """
        marked = re.sub(r"[.,;]", " | ", text.lower())
        return re.findall(r"[a-zA-Z']+|\|", marked)

    def _context_window(self, tokens: List[str], aspect: str) -> str:
        """Builds a clause-scoped window of context around an aspect's mention.

        The window extends outward from the aspect mention up to
        `self.window_size` words on each side, but stops early at clause
        boundaries (contrast conjunctions like "but", or punctuation) so
        that sentiment words from a different clause are not attributed to
        this aspect.

        Args:
            tokens (List[str]): Tokenized source text (may include "|"
                clause-break markers from `_tokenize`).
            aspect (str): The aspect phrase to locate within `tokens`.

        Returns:
            str: The clause-scoped context window as a space-joined string.
                Falls back to the full token sequence if the aspect cannot
                be located.
        """
        aspect_tokens = aspect.lower().split()
        n = len(aspect_tokens)

        for i in range(len(tokens) - n + 1):
            if tokens[i:i + n] == aspect_tokens:
                start = self._scan_to_boundary(tokens, i - 1, step=-1)
                end = self._scan_to_boundary(tokens, i + n, step=1)
                window_tokens = [t for t in tokens[start:end] if t != "|"]
                return " ".join(window_tokens)

        # Aspect not found verbatim (e.g. due to tokenization differences);
        # fall back to scoring the whole sentence.
        return " ".join(t for t in tokens if t != "|")

    def _scan_to_boundary(self, tokens: List[str], start_index: int, step: int) -> int:
        """Scans outward from a position until a clause boundary or window limit.

        Args:
            tokens (List[str]): Tokenized source text.
            start_index (int): Index to begin scanning from.
            step (int): +1 to scan forward, -1 to scan backward.

        Returns:
            int: The index just past the boundary that was hit, clamped to
                a valid slice bound within `self.window_size`.
        """
        i = start_index
        words_seen = 0

        while 0 <= i < len(tokens) and words_seen < self.window_size:
            token = tokens[i]
            if token == "|" or token in CLAUSE_BOUNDARIES:
                break
            words_seen += 1
            i += step

        return i + 1 if step == -1 else i
