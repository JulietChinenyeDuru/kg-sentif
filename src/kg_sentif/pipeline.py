"""End-to-end KG-SentIF pipeline.

Orchestrates aspect extraction, aspect-level sentiment classification, and
knowledge graph construction into a single, easy-to-call interface.
"""

from typing import Dict, Any, List

from .aspect_extraction import AspectExtractor
from .sentiment_classifier import LexiconSentimentClassifier, AspectSentiment
from .knowledge_graph import AspectKnowledgeGraph


class KGSentIFPipeline:
    """Runs the full aspect extraction -> sentiment -> knowledge graph pipeline.

    Attributes:
        extractor (AspectExtractor): Component used to identify aspect terms.
        classifier (LexiconSentimentClassifier): Component used to classify
            aspect-level sentiment.
        graph (AspectKnowledgeGraph): Persistent knowledge graph accumulated
            across all documents passed to `analyze`.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Initializes the pipeline and its component modules.

        Args:
            spacy_model (str): Name of the spaCy model used for aspect
                extraction.
        """
        self.extractor = AspectExtractor(model_name=spacy_model)
        self.classifier = LexiconSentimentClassifier()
        self.graph = AspectKnowledgeGraph()

    def analyze(self, text: str) -> Dict[str, Any]:
        """Runs the full pipeline on a single piece of text.

        Args:
            text (str): Raw review or social media text to analyze.

        Returns:
            Dict[str, Any]: A structured result containing the original
                text, per-aspect sentiment results, and a summary of the
                knowledge graph state after incorporating this document.
        """
        aspects = self.extractor.extract(text)
        aspect_sentiments: List[AspectSentiment] = self.classifier.classify(text, aspects)
        self.graph.add_document(aspect_sentiments)

        return {
            "text": text,
            "aspects": [
                {
                    "aspect": item.aspect,
                    "sentiment": item.sentiment,
                    "confidence": item.confidence,
                }
                for item in aspect_sentiments
            ],
            "graph_summary": self.graph.summary(),
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Runs the pipeline over multiple documents, accumulating one shared graph.

        Args:
            texts (List[str]): A list of review or social media texts.

        Returns:
            List[Dict[str, Any]]: One result dict per input text, in order.
        """
        return [self.analyze(text) for text in texts]
