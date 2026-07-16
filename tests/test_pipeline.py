"""Unit tests for the KG-SentIF pipeline and its components."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402
from kg_sentif.pipeline import KGSentIFPipeline  # noqa: E402
from kg_sentif.sentiment_classifier import LexiconSentimentClassifier  # noqa: E402
from kg_sentif.knowledge_graph import AspectKnowledgeGraph  # noqa: E402
from kg_sentif.sentiment_classifier import AspectSentiment  # noqa: E402


@pytest.fixture(scope="module")
def pipeline() -> KGSentIFPipeline:
    """Provides a shared pipeline instance for tests in this module."""
    return KGSentIFPipeline()


def test_analyze_returns_expected_keys(pipeline: KGSentIFPipeline) -> None:
    """The result of analyze() should contain the expected top-level keys."""
    result = pipeline.analyze("The camera is great but the battery is bad.")
    assert "text" in result
    assert "aspects" in result
    assert "graph_summary" in result


def test_aspect_sentiment_polarity(pipeline: KGSentIFPipeline) -> None:
    """Aspects with clearly polarized context should be scored accordingly."""
    result = pipeline.analyze("The camera quality is excellent but the battery life is disappointing.")
    aspects_by_name = {a["aspect"]: a["sentiment"] for a in result["aspects"]}

    assert any("camera" in name for name in aspects_by_name)
    assert any("battery" in name for name in aspects_by_name)

    for name, sentiment in aspects_by_name.items():
        if "camera" in name:
            assert sentiment == "positive"
        if "battery" in name:
            assert sentiment == "negative"


def test_negation_flips_sentiment() -> None:
    """A negated positive word should be classified as negative, not positive."""
    classifier = LexiconSentimentClassifier()
    sentiment, confidence = classifier.score_window("not a great experience")
    assert sentiment == "negative"


def test_knowledge_graph_tracks_cooccurrence() -> None:
    """Aspects appearing in the same document should be linked by an edge."""
    graph = AspectKnowledgeGraph()
    graph.add_document([
        AspectSentiment(aspect="battery", sentiment="negative", confidence=0.8),
        AspectSentiment(aspect="camera", sentiment="positive", confidence=0.9),
    ])

    related = graph.related_aspects("battery")
    assert ("camera", 1) in related


def test_batch_analysis_accumulates_shared_graph(pipeline: KGSentIFPipeline) -> None:
    """analyze_batch should build one shared graph across all documents."""
    texts = [
        "The battery life is poor.",
        "The battery life is poor but the camera is good.",
    ]
    results = pipeline.analyze_batch(texts)
    assert len(results) == 2
    # Graph should have accumulated mentions across both documents.
    assert results[-1]["graph_summary"]["nodes"] >= 1
