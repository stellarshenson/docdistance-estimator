# data/interim - converted and derived intermediates

Intermediate artefacts between the raw sources and the final fixture: text converted from the source PDFs, the generated document corpus, and the structure controls. Notebook 11 reads the tracked inputs here and writes the derived ones.

- **Stage role** - the middle of `external → interim → processed`
- **Two kinds of artefact** - pre-generated **inputs** (committed, not reproducible by the notebook alone) and notebook-**produced** intermediates
- **Curated source text lives here** - the hand-reviewed `source-article.md` per article is the canonical readable text the pipeline segments; the PDF conversion that seeds it runs in `external → interim`

## Pre-generated inputs (committed)

- `exec-summaries/ibm-ai-adoption/source/source-article.md` - curated IBM article text
- `exec-summaries/ibm-ai-adoption/summaries/*.md` - 11 LLM-generated exec-summary variants (opus / sonnet / haiku, gold and adversarial tiers); the document corpus, generated offline since the notebook has no LLM API
- `ai-society-wergeland/source/source-article.md` - curated Wergeland article text, hand-reviewed from the PDF extract

## Notebook-produced

- `structure-paraphrase/*.bt.json` - opus-mt EN → DE → EN back-translation caches, one per base; the content-invariance control, cached so re-runs stay offline
- `ai-society-wergeland/source/raw-extract.txt` (ignored) - per-column pdfplumber dump of the Wergeland PDF, regenerable from the source
- `01-statements.parquet` - statements table from the data-exploration notebook (01)
