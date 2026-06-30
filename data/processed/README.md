# data/processed - final fixture dataset

The canonical structure-distance fixture the experiments load. Written by notebook 11 (Stage 3); experiments E07-E11 and the E2E showcase read it directly.

- **Stage role** - pipeline output, the final dataset of `external → interim → processed`
- **Self-contained** - statements, permutations, pairs and metadata; consumers compute embeddings and distances from it
- **Deterministic** - regenerating from the tracked interim inputs reproduces these files byte-for-byte

## Artefacts (`structure-fixture/`)

- `statements.json` - every document segmented into statements, each carrying its block id/name, tier, kind, and article
- `reorder_pool.json` - byte-identical `k`-swap permutations per base across a displacement grid; the order-isolation upper bound
- `pairs.json` - the evaluation pair regimes: cross-summary (diffuse), tier-contrast, paraphrase (content-invariance), section-swap
- `meta.json` - seeds, swap grid, displacement bins, article map, and the paraphrase provenance
