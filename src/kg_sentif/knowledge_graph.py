"""Knowledge graph construction module.

Builds a graph representation of aspects and their relationships, both
within a single document (co-occurrence) and across a corpus of documents
(shared aspect nodes accumulate cross-document edges).

Using a graph rather than a flat list lets aspect relationships be queried
directly — for example, finding which aspects most frequently co-occur with
a negatively-scored aspect across an entire review corpus.
"""

from typing import Dict, List, Tuple, Any
import itertools
import networkx as nx

from .sentiment_classifier import AspectSentiment


class AspectKnowledgeGraph:
    """Maintains a knowledge graph of aspects, sentiments, and co-occurrence.

    Nodes represent aspect terms (deduplicated across all processed
    documents). Node attributes track cumulative sentiment counts. Edges
    represent co-occurrence within the same document, weighted by how often
    two aspects appear together.

    Attributes:
        graph (networkx.Graph): The underlying graph structure.
    """

    def __init__(self):
        """Initializes an empty aspect knowledge graph."""
        self.graph = nx.Graph()

    def add_document(self, aspect_sentiments: List[AspectSentiment]) -> None:
        """Adds a single document's aspects and sentiments to the graph.

        Args:
            aspect_sentiments (List[AspectSentiment]): Aspect-sentiment
                results for one document, as produced by
                `LexiconSentimentClassifier.classify`.
        """
        for item in aspect_sentiments:
            self._add_or_update_node(item)

        # Co-occurring aspects within the same document get an edge.
        aspect_names = [item.aspect for item in aspect_sentiments]
        for a, b in itertools.combinations(sorted(set(aspect_names)), 2):
            self._add_or_update_edge(a, b)

    def _add_or_update_node(self, item: AspectSentiment) -> None:
        """Adds a new aspect node or updates sentiment counts on an existing one.

        Args:
            item (AspectSentiment): A single aspect's sentiment result.
        """
        if item.aspect not in self.graph:
            self.graph.add_node(
                item.aspect,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                mentions=0,
            )

        self.graph.nodes[item.aspect]["mentions"] += 1
        self.graph.nodes[item.aspect][f"{item.sentiment}_count"] += 1

    def _add_or_update_edge(self, aspect_a: str, aspect_b: str) -> None:
        """Adds a co-occurrence edge between two aspects, or increments its weight.

        Args:
            aspect_a (str): First aspect term.
            aspect_b (str): Second aspect term.
        """
        if self.graph.has_edge(aspect_a, aspect_b):
            self.graph[aspect_a][aspect_b]["weight"] += 1
        else:
            self.graph.add_edge(aspect_a, aspect_b, weight=1)

    def related_aspects(self, aspect: str, top_k: int = 5) -> List[Tuple[str, int]]:
        """Finds the most frequently co-occurring aspects for a given aspect.

        Args:
            aspect (str): The aspect to find relationships for.
            top_k (int): Maximum number of related aspects to return.

        Returns:
            List[Tuple[str, int]]: (related_aspect, co_occurrence_weight)
                pairs, sorted by descending weight. Empty list if the aspect
                is not present in the graph.
        """
        if aspect not in self.graph:
            return []

        neighbors = [
            (neighbor, self.graph[aspect][neighbor]["weight"])
            for neighbor in self.graph.neighbors(aspect)
        ]
        neighbors.sort(key=lambda pair: pair[1], reverse=True)
        return neighbors[:top_k]

    def summary(self) -> Dict[str, Any]:
        """Produces a summary snapshot of the current graph state.

        Returns:
            Dict[str, Any]: Node count, edge count, and the highest-weight
                related aspect pairs currently in the graph.
        """
        edges_by_weight = sorted(
            self.graph.edges(data=True),
            key=lambda e: e[2].get("weight", 0),
            reverse=True,
        )
        top_pairs = [(u, v) for u, v, _ in edges_by_weight[:5]]

        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "related_aspect_pairs": top_pairs,
        }
