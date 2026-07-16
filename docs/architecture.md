# KG-SentIF Architecture

## Pipeline stages

### 1. Aspect Extraction (`aspect_extraction.py`)
Uses spaCy's dependency parser to extract noun chunks as candidate aspect
terms, filtering out pronouns, determiners, and very short tokens. This is a
standard, interpretable baseline for aspect identification; more advanced
setups can substitute a fine-tuned sequence-labeling model (e.g. BERT-CRF)
here without changing the rest of the pipeline.

### 2. Aspect-Level Sentiment Classification (`sentiment_classifier.py`)
The default `LexiconSentimentClassifier`:
- Locates each aspect's mention in the source text
- Builds a **clause-scoped context window** around it — the window stops
  early at contrast conjunctions ("but", "however", "although") and
  sentence punctuation, so sentiment words from a different clause aren't
  wrongly attributed to an aspect in another clause
- Applies a 3-word negation lookback (handles "not a great experience")
- Counts positive/negative lexicon hits within the window to assign
  polarity and a heuristic confidence score

The `SentimentModel` interface allows this to be swapped for a
transformer-based classifier (e.g. a fine-tuned RoBERTa ABSA model) without
changing the pipeline or knowledge graph code.

### 3. Knowledge Graph Construction (`knowledge_graph.py`)
Builds a `networkx.Graph` where:
- **Nodes** = unique aspect terms, with cumulative sentiment counts as
  attributes
- **Edges** = co-occurrence relationships, weighted by how often two
  aspects appear together across the processed documents

This enables graph-level queries such as "which aspects most often
co-occur with a negatively-scored aspect across the whole corpus" — a
capability a flat sentiment classifier does not provide.

---

## Known limitations (baseline lexicon classifier)

This is an honest account of the current limitations, since the default
classifier is a transparent lexicon-based baseline rather than a trained
model:

1. **Context-dependent polarity words.** Some words are positive in one
   context and negative in another — e.g. "fast" is positive for "the app
   is fast" but negative for "the battery drains fast." The lexicon
   approach cannot currently distinguish these cases. This is a known
   limitation of rule-based ABSA and a strong motivation for swapping in a
   trained `SentimentModel` implementation for production use.
2. **Compound noun chunks.** spaCy's noun chunker occasionally merges
   multiple coordinated aspects into a single chunk (e.g. "great sound
   quality and comfortable fit" as one chunk instead of two). A dependency-
   based splitting rule on coordinating conjunctions ("and") would improve
   this — tracked in the roadmap.
3. **Lexicon coverage.** The illustrative lexicon in
   `sentiment_classifier.py` is intentionally small and meant to be
   replaced with a validated resource (SentiWordNet, VADER) or a trained
   classifier for anything beyond demonstration purposes.

These limitations are the expected profile of a transparent, dependency-free
baseline, and are documented here rather than hidden so the framework's
current scope is clear to anyone building on it.
