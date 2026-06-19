# docdistance

[![CI](https://github.com/stellarshenson/docdistance/actions/workflows/ci.yml/badge.svg)](https://github.com/stellarshenson/docdistance/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/docdistance.svg)](https://pypi.org/project/docdistance/)
[![Total PyPI downloads](https://static.pepy.tech/badge/docdistance)](https://pepy.tech/project/docdistance)

Semantic distance between two documents via Statement Mover's Distance - optimal transport over mmBERT statement embeddings, after Kusner et al. 2015 ([*From Word Embeddings To Document Distances*](references/papers/%5Bpaper%5D%20From%20Word%20Embeddings%20To%20Document%20Distances.pdf)). A thin frontend to the library; the SOTA docs carry the mechanics, benchmarks, and validation.

<p align="center">
  <img src=".resources/docdistance-banner.svg" alt="docdistance" width="100%">
</p>

- **Input** - two documents, raw text or a file path
- **Output** - an SMD distance, a 0..1 closeness, a verdict, and the statement alignment
- **Use** - agentic document conversion and extraction pipelines, where token logits are unavailable and KL divergence cannot be computed
- **Unit** - statement-level and position-invariant, with an interpretable transport plan

## Theory

A document distance grounded in embeddings and optimal transport, not surface overlap.

- **WMD** - Word Mover's Distance (Kusner et al. 2015) casts document similarity as optimal transport between embedded tokens
- **SMD** - this project lifts it to statements: segment, embed, transport between the two statement clouds
- **Beyond cosine** - whole-document cosine collapses when the same claims sit in a different place or order; statement-level transport is position-invariant
- **Metric** - the ground cost `√(2 − 2cos)` on L2-normalized embeddings is a metric, so the document distance is one too
- **Logit-free** - an embedding-grounded alternative where token probabilities (KL divergence) are unavailable, as in frontier-model pipelines

## Method

Three stages; the transport plan is the interpretable by-product.

1. **Segment** - split each document into atomic statements with the SAT (Segment Any Text) segmenter
2. **Embed** - encode each statement with the mmBERT contextual encoder (mean-pooled, L2-normalized)
3. **Compare** - optimal transport between the two statement clouds (Statement Mover's Distance), optionally unbalanced so added or missing statements are scored, not force-matched

- **Closeness** - `1 − SMD/√2`, on a 0..1 scale
- **Source-conditioned** - a variant `d(A, B | S)` re-bases the transport onto a shared source `S` and reads off a selection axis and a grounding axis

## Which distance

Both compare two documents; they differ in what the answer tells you and what you must supply.

- **Method 1 - symmetric distance (robust, fast)** - answers *how far apart are A and B?* as one number (a 0..1 closeness plus a similar / not-similar verdict). Sub-millisecond, needs only the two documents, and is a true metric - the distance is symmetric and obeys the triangle inequality, so the numbers are consistent enough to threshold, rank and cache. The production default; use it whenever you need a reliable similarity score - dedup, drift detection, "did this conversion change the meaning?"
- **Method 2 - source-conditioned `d(A, B | S)` (slower, experimental)** - answers *why do A and B differ, given a shared source S?* You supply `S`, and instead of one number it returns two axes: a selection axis (did A and B pick different parts of the source?) and a grounding axis (did one drift from the source - dropped content vs unsupported or fabricated content?). It runs a cross-encoder × NLI pass (~seconds on GPU, far slower on CPU). Use it to audit a summary or an extraction against its source, when "how far" is not enough and you need to name the failure
- **Which to pick** - default to Method 1 for a similarity number; reach for Method 2 only when you hold the shared source and need to know *why* two documents derived from it diverge. Method 2's value is interpretation and ordering the failure modes correctly, not a higher pass rate, and it is validated on a single fixture so far - validate on your own sources first

## Usage

The quickest way to a result is the CLI - install once, then run it.

```bash
make install                                   # environment, package, Jupyter kernel
docdistance install                            # download + cache the models (once)

# method 1 - symmetric distance (robust, fast, the default)
docdistance distance a.md b.md                 # rich, coloured verdict
docdistance distance a.md b.md --json          # machine-readable JSON
docdistance distance a.md b.md --result-only   # bare SMD scalar, for scripts

# method 2 - source-conditioned d(A,B|S) (slower, experimental)
docdistance distance-wrt-source a.md b.md --source s.md          # rich, two-axis verdict
docdistance distance-wrt-source a.md b.md -s s.md --json         # machine-readable JSON
```

### CLI reference

| Command | Does | Key flags |
|---|---|---|
| `docdistance install` | download + cache the models, once | `--backend openvino\|torch\|both` |
| `docdistance distance A B` | method 1 - symmetric SMD distance + verdict | `--backend`, `--gpu`, `--anisotropy`, `--threshold`, `--json`, `--result-only` |
| `docdistance distance-wrt-source A B -s S` | method 2 - source-conditioned `d(A,B\|S)` | `--source/-s` (required), `--backend`, `--gpu`, `--json`, `--result-only` |

`A` / `B` / `S` are file paths or raw text. Run `docdistance --help` or `docdistance <command> --help` for the full flag list; the [API reference](docs/api-reference.md) covers the library.

The same thing is one function from Python:

```python
from docdistance import document_distance

result = document_distance("report_v1.md", "report_v2.md")
print(result.closeness)  # 0..1 similarity, 1 - SMD/sqrt(2)
print(result.verdict)    # "similar" | "not similar"
```

### Source-conditioned - why two documents of one source diverge

When A and B share a known source `S`, the symmetric distance tells you *how far* apart they are but not *why*. The source-conditioned distance `d(A, B | S)` re-bases both onto `S` and splits the difference into a selection axis (what each picked from the source) and a grounding axis (how far each drifts from it) - so dropped content reads differently from unsupported content. Reach for it to audit a summary or extraction against its source; it is slower and experimental, so validate on your own sources.

```bash
docdistance distance-wrt-source summary_a.md summary_b.md --source article.md
```

```python
from docdistance import source_conditioned_distance

r = source_conditioned_distance("summary_a.md", "summary_b.md", source="article.md")
print(r.d_sel)                     # how differently A and B select from the source
print(r.residual_a, r.residual_b)  # each summary's distance to the source
```

- **Offline after install** - distance calls run fully offline once the models are cached
- **Backend** - `--backend openvino|torch`, default `openvino` (CPU INT8)
- **Full API and flags** - `docdistance --help` and the SOTA docs

## Documentation

The SOTA documents explain how it works in detail; this README only introduces it.

- `docs/wmd-docdistance-solution-sota.md` - source-free distance: design, mechanism, performance, validation
- `docs/wmd-source-conditioned-docdistance-solution-sota.md` - source-conditioned distance `d(A,B|S)`: two axes (selection + grounding), design, performance, limitations
- `docs/mmbert-quantization-solution.md` - the INT8 / FP8 statement encoder
- [*From Word Embeddings To Document Distances*](references/papers/%5Bpaper%5D%20From%20Word%20Embeddings%20To%20Document%20Distances.pdf) - Kusner et al. 2015, the WMD theory ([digest](references/papers/from-word-embeddings-to-document-distances.md))
- [*All-but-the-Top: Simple and Effective Postprocessing for Word Representations*](references/papers/%5Bpaper%5D%20All-but-the-Top%3A%20Simple%20and%20Effective%20Postprocessing%20for%20Word%20Representations.pdf) - Mu & Viswanath, ICLR 2018, the anisotropy postprocessing ([digest](references/papers/all-but-the-top-simple-and-effective-postprocessing-for-word-representations.md))

> **Note**: Scaffolded with the [copier-data-science](https://github.com/stellarshenson/copier-data-science) template.
