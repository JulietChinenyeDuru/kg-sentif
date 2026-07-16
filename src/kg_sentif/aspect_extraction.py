"""Aspect extraction module.

Identifies candidate aspect terms (product/service features being discussed)
within a piece of text, using noun-phrase extraction and lightweight
dependency-based filtering.
"""

from typing import List, Set
import spacy


# Generic stopword-like terms that are grammatically noun phrases but
# rarely useful as sentiment aspects.
DEFAULT_ASPECT_STOPWORDS: Set[str] = {
    "it", "this", "that", "these", "those", "i", "we", "you", "he", "she",
    "they", "everything", "something", "anything", "nothing", "one",
}


class AspectExtractor:
    """Extracts candidate aspect terms from text using noun-phrase parsing.

    Attributes:
        nlp (spacy.language.Language): Loaded spaCy pipeline used for
            tokenization, POS tagging, and dependency parsing.
        min_aspect_length (int): Minimum character length for a candidate
            aspect term to be retained.
    """

    def __init__(self, model_name: str = "en_core_web_sm", min_aspect_length: int = 3):
        """Initializes the extractor with a spaCy language model.

        Args:
            model_name (str): Name of the spaCy model to load. Defaults to
                the small English pipeline.
            min_aspect_length (int): Minimum character length for a
                candidate aspect to be kept. Filters out noise like single
                letters or very short tokens.

        Raises:
            OSError: If the specified spaCy model is not installed. Run
                `python -m spacy download en_core_web_sm` to install it.
        """
        self.nlp = spacy.load(model_name)
        self.min_aspect_length = min_aspect_length

    def extract(self, text: str) -> List[str]:
        """Extracts candidate aspect terms from a piece of text.

        Args:
            text (str): Raw input text (e.g. a product review or social
                media post).

        Returns:
            List[str]: A list of unique, lowercased candidate aspect terms,
                in order of first appearance.
        """
        doc = self.nlp(text)
        aspects: List[str] = []
        seen = set()

        for chunk in doc.noun_chunks:
            candidate = self._clean_chunk(chunk.text)
            if not candidate:
                continue
            if candidate in seen:
                continue
            if candidate.lower() in DEFAULT_ASPECT_STOPWORDS:
                continue
            if len(candidate) < self.min_aspect_length:
                continue

            seen.add(candidate)
            aspects.append(candidate)

        return aspects

    @staticmethod
    def _clean_chunk(chunk_text: str) -> str:
        """Normalizes a noun chunk by stripping leading determiners.

        Args:
            chunk_text (str): Raw noun chunk text from spaCy.

        Returns:
            str: Cleaned, lowercased aspect candidate string.
        """
        words = chunk_text.strip().split()
        determiners = {"a", "an", "the", "my", "his", "her", "their", "our", "your"}
        while words and words[0].lower() in determiners:
            words = words[1:]
        return " ".join(words).lower().strip()
