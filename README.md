# KG-SentIF: Knowledge Graph-Enhanced Sentiment Intelligence Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**KG-SentIF** is an open-source implementation of a knowledge graph-enhanced
approach to Aspect-Based Sentiment Analysis (ABSA). It combines transformer-based
NLP with graph-structured aspect relationships to produce sentiment intelligence
that is more interpretable and context-aware than flat sentiment classification.

This repository accompanies ongoing research on knowledge graph-enhanced
sentiment intelligence for mobile social media and product review analysis.

---

## What this framework does

Given raw review or social media text, KG-SentIF:

1. **Extracts aspects** identifies the specific entities/features being
   discussed (e.g. "battery life", "camera quality", "customer service").
2. **Builds a knowledge graph** links extracted aspects to a structured
   graph, capturing relationships between aspects, products, and categories.
3. **Classifies sentiment per aspect** assigns polarity (positive / negative
   / neutral) to each aspect individually, rather than one score for the
   whole text.
4. **Produces an aspect-sentiment intelligence report** a structured,
   queryable output combining (1)-(3), suitable for downstream analytics.

### Why knowledge graphs?

Flat sentiment analysis treats a review as one blob of text with one sentiment
score. This loses critical nuance a review can praise the camera and
criticize the battery in the same sentence. By representing aspects and their
relationships as a graph, KG-SentIF:

- Preserves aspect-level nuance instead of collapsing it into an average
- Allows sentiment patterns to be queried and aggregated across a product line
- Enables reasoning over related aspects (e.g. "battery" ↔ "charging speed" ↔
  "power management")

---

## Architecture

```
Raw Text
   │
   ▼
┌─────────────────────┐
│  Aspect Extraction    │  → identifies aspect terms in text
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│ Knowledge Graph       │  → links aspects into a structured graph
│ Construction          │
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│ Aspect-Level          │  → classifies sentiment per aspect
│ Sentiment Classifier  │
└─────────┬────────────┘
          ▼
Aspect-Sentiment Intelligence Report
```

See [`docs/architecture.md`](docs/architecture.md) for a detailed breakdown of
each module.

---

## Installation

```bash
git clone https://github.com/JulietChinenyeDuru/kg-sentif.git
cd kg-sentif
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Quick start

```python
from kg_sentif.pipeline import KGSentIFPipeline

pipeline = KGSentIFPipeline()

review = "The camera quality is excellent but the battery life is disappointing."
result = pipeline.analyze(review)

print(result)
```

Expected output (structure):

```python
{
    "text": "The camera quality is excellent but the battery life is disappointing.",
    "aspects": [
        {"aspect": "camera quality", "sentiment": "positive", "confidence": 0.91},
        {"aspect": "battery life", "sentiment": "negative", "confidence": 0.88}
    ],
    "graph_summary": {
        "nodes": 2,
        "edges": 1,
        "related_aspect_pairs": [("camera quality", "battery life")]
    }
}
```

## Running the demo

```bash
python examples/demo.py
```

This runs the pipeline over a small sample set of product reviews
(`examples/sample_reviews.json`) and prints the aspect-sentiment graph output
for each.

## Running tests

```bash
pytest tests/
```

---

## Repository structure

```
kg-sentif/
├── src/kg_sentif/
│   ├── aspect_extraction.py     # Aspect term identification
│   ├── knowledge_graph.py       # Graph construction over aspects
│   ├── sentiment_classifier.py  # Aspect-level sentiment scoring
│   └── pipeline.py              # End-to-end orchestration
├── examples/
│   ├── demo.py
│   └── sample_reviews.json
├── tests/
│   └── test_pipeline.py
├── docs/
│   └── architecture.md
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Roadmap

- [ ] Multilingual aspect extraction (Nigerian Pidgin, Yoruba, Igbo, Hausa)
- [ ] Graph neural network layer for aspect relationship learning
- [ ] Integration with real-time social media streams
- [ ] Benchmark against SemEval-2014 Task 4 ABSA dataset

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{duru2026kgsentif,
  author       = {Duru, Juliet Chinenye},
  title        = {KG-SentIF: Knowledge Graph-Enhanced Sentiment Intelligence Framework},
  year         = {2026},
  howpublished = {\url{https://github.com/JulietChinenyeDuru/kg-sentif}},
}
```

## License

This project is licensed under the MIT License see [LICENSE](LICENSE) for details.

## Author

**Juliet Chinenye Duru**
ICT Lecturer, Abia State University · Cloud & AI DevOps Engineer
[ORCID: 0009-0002-0530-8082](https://orcid.org/0009-0002-0530-8082)
