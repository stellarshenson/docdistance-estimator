# docdistance

Compute a meaningful semantic distance between two documents using Word Mover's Distance (WMD) and Optimal Transport, following Kusner et al. 2015 (*From Word Embeddings To Document Distances*). The intended use is agentic document conversion and extraction pipelines that run through frontier models, where token-level logits are unavailable and KL divergence cannot be computed directly - WMD provides an embedding-grounded distance instead.

## Why this exists

Whole-document cosine similarity is too coarse when two executive summaries carry the same claims in different places, in a different order, or with content added or dropped. Optimal transport lifts the comparison to individual statements: it matches each statement in one document to its best counterpart in the other regardless of position, and the transport plan itself reveals what moved, what was added, and what was dropped.

## Approach

The distance is computed in stages:

1. **Segment** each document into atomic statements with the SAT (Segment Any Text) sentence segmenter
2. **Embed** each statement with a contextual encoder
3. **Compare** the two statement clouds with optimal transport (statement-level Word Mover's Distance), optionally unbalanced so added or missing statements are scored explicitly rather than force-matched

Position-invariance, statement-level granularity, and an interpretable alignment are what distinguish this from a single document embedding.

## Validation

The distance is implemented in `notebooks/04-kj-wmd-document-distance.ipynb` and validated on an executive-summary fixture set built from one IBM AI-adoption article - a gold tier (faithful summaries written under shared rules) plus two adversarial tiers (information loss and information noise). Statement Mover's Distance ranks every gold summary closer to the anchor than every adversarial one with zero ordering mistakes. The design and conclusions are in `docs/wmd-docdistance-solution-sota.md`, with a source-conditioned variant (`d(A,B|S)`) in `docs/wmd-wrt-source-docdistance-solution.md`.

## Library

The package exposes a small API for dropping the distance into a pipeline. A document argument is
either raw text or a path to a text/markdown file - paths are auto-detected.

```python
from docdistance import document_distance

result = document_distance("report_v1.md", "report_v2.md")
print(result.smd)        # the distance - lower is more similar
print(result.closeness)  # 0..1 similarity (1 - SMD/sqrt(2))
print(result.verdict)    # "similar" | "not similar"
print(result.to_dict())  # full result as a plain dict (JSON-ready)
```

For repeated comparisons load the models once and reuse them:

```python
from docdistance import DocDistance

dd = DocDistance(backend="openvino")     # models load here, once
ab = dd.distance("a.md", "b.md")
ac = dd.distance("a.md", "c.md")
```

Source-conditioned distance `d(A, B | S)` - two documents derived from a common source:

```python
from docdistance import source_conditioned_distance

r = source_conditioned_distance("summary_a.md", "summary_b.md", "article.md")
print(r.d_sel)                      # selection divergence over the shared source
print(r.residual_a, r.residual_b)   # each document's distance to the source
```

Models are not downloaded on first use - run `docdistance install` (or `DocDistance(...)` after
installing) once; the distance calls then run fully offline.

## Command line

```bash
docdistance install                                  # download + cache the models (once)
docdistance distance a.md b.md                       # rich, coloured verdict
docdistance distance "first text" "second text"      # raw text works too
docdistance distance a.md b.md --json                # machine-readable JSON
docdistance distance a.md b.md --result-only         # just the distance number
docdistance distance a.md b.md --backend torch -v    # torch encoder, verbose logs
docdistance distance-wrt-source a.md b.md --source article.md
docdistance distance-wrt-source a.md b.md -s s.md --result-only   # D_sel,res_a,res_b
```

Every command has `--help` with examples. The `install` command is the only one that touches the
network; `distance` and `distance-wrt-source` run offline and raise a clear error if a model is
missing. The encoder backend is selectable (`--backend openvino|torch`, default `openvino`).

## Notebooks

- `notebooks/01-kj-document-segmentation.ipynb` - stage 1: splits a source PDF into statements with the `sat-3l-sm` model (PyTorch, GPU), writing `data/interim/01-statements.parquet`
- `notebooks/02-kj-mmbert-quantization.ipynb` - stage 0: quantizes the mmBERT statement encoder and emits the best model per target (CPU OpenVINO INT8, GPU torchao FP8)
- `notebooks/03-kj-mmbert-throughput-saturation.ipynb` - GPU batch-saturation sweep for the encoder (throughput knee, per-core CPU optimum)
- `notebooks/04-kj-wmd-document-distance.ipynb` - stage 3: the statement-level distance (WCD / RWMD / SMD), scored across the fixture set and validated against the gold anchor

## Encoder quantization performance

The mmBERT statement encoder is quantized for two deployment targets. All rows normalized to CPU FP as the 1.0x baseline (full detail in `docs/mmbert-quantization-solution.md`; shipped CPU model: [`stellars/mmBERT-base-openvino-int8`](https://huggingface.co/stellars/mmBERT-base-openvino-int8)).

| config | ms/sentence | sentences/sec | speedup |
|---|---|---|---|
| CPU FP (base, full precision) | 30.6 | 33 | 1.0x |
| CPU OpenVINO INT8 | 21.4 | 47 | 1.4x |
| GPU bf16 eager (raw base) | 0.84 | 1196 | 37x |
| GPU bf16 compiled | 0.44 | 2281 | 70x |
| **GPU FP8 + compile** | **0.39** | **2588** | **79x** |

GPU FP8 is ~2.2x over the raw GPU base and ~55x over the shipped CPU INT8 model, at near-lossless fidelity (GPU 0.999, CPU 0.98 vs FP32). GPU rows are throughput at batch 128 / seq 128; CPU rows are per-sentence latency at small batch, so the cross-device multiple is directional, not a like-for-like benchmark.

## Quick Start

```bash
make install
```

## Makefile Targets

- `make install` - Create environment and install package
- `make test` - Run tests
- `make lint` / `make format` - Check / fix code style
- `make build` - Build distributable wheel
- `make dist` - Build sdist + wheel into `dist/`
- `make publish` - Validate and upload to PyPI with twine
- `make clean` - Remove compiled files and caches
- `make .env` / `make .env.enc` - Decrypt / encrypt environment secrets
- `make help` - Show all available targets

## Best Practices

- **Notebooks**: Name with number prefix, initials, description - `01-jqp-data-exploration.ipynb`
- **Data**: Keep `raw/` immutable, use `interim/` for transforms, `processed/` for final datasets
- **Source code**: Refactor reusable notebook code into `src/docdistance/` modules
- **Models**: Store trained models in `models/` with clear naming

## References

- `references/papers/from-word-embeddings-to-document-distances.md` - digest of the WMD paper (Kusner et al. 2015)
- `docs/wmd-docdistance-solution-sota.md` - source-free distance design, implementation, and results
- `docs/wmd-wrt-source-docdistance-solution.md` - source-conditioned distance design (`d(A,B|S)`)

## Project Organization

```
├── Makefile           <- Makefile with convenience commands
├── README.md          <- The top-level README for developers
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw            <- The original, immutable data dump
│
├── models             <- Trained and serialized models
├── notebooks          <- Jupyter notebooks
├── pyproject.toml     <- Project configuration and dependencies
├── references         <- Data dictionaries, manuals, explanatory materials
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures
├── tests              <- Test files
└── src
    └── docdistance   <- Source code for this project
        ├── __init__.py
        ├── config.py      <- Configuration variables
        ├── dataset.py     <- Data download/generation scripts
        ├── features.py    <- Feature engineering code
        ├── modeling
        │   ├── predict.py <- Model inference
        │   └── train.py   <- Model training
        └── plots.py       <- Visualization code
```

> **Note**: Scaffolded with the [copier-data-science](https://github.com/stellarshenson/copier-data-science) template.
