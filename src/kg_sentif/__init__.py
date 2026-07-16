"""KG-SentIF: Knowledge Graph-Enhanced Sentiment Intelligence Framework.

A framework for aspect-based sentiment analysis that represents extracted
aspects and their relationships as a knowledge graph, enabling aspect-level
sentiment reasoning rather than flat, document-level sentiment scoring.
"""

from .pipeline import KGSentIFPipeline

__all__ = ["KGSentIFPipeline"]
__version__ = "0.1.0"
