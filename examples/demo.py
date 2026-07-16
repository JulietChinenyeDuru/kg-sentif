"""Demo script: runs the KG-SentIF pipeline over a sample set of reviews.

Usage:
    python examples/demo.py
"""

import json
import os
import sys

# Allow running this script directly without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kg_sentif.pipeline import KGSentIFPipeline  # noqa: E402


def main() -> None:
    """Loads sample reviews, runs the pipeline, and prints results."""
    sample_path = os.path.join(os.path.dirname(__file__), "sample_reviews.json")
    with open(sample_path, "r", encoding="utf-8") as f:
        reviews = json.load(f)

    pipeline = KGSentIFPipeline()

    for i, review in enumerate(reviews, start=1):
        result = pipeline.analyze(review)
        print(f"\n--- Review {i} ---")
        print(f"Text: {result['text']}")
        print("Aspects:")
        for aspect in result["aspects"]:
            print(f"  - {aspect['aspect']}: {aspect['sentiment']} "
                  f"(confidence={aspect['confidence']})")
        print(f"Graph summary so far: {result['graph_summary']}")


if __name__ == "__main__":
    main()
