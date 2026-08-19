# Haltere campaniform sensilla → wing steering motor neurons

Do the wing steering motor neurons **b1, b2, b3, tp1, tp2 and tpn** receive similar
input from haltere campaniform sensilla (HCS)? If HCS encode different flight
variables, then a motor neuron whose input pattern resembles a well-characterised one
is plausibly recruited under similar conditions.

The same analysis is run independently on **three connectomes**, using **direct
(monosynaptic) connections only**:

| Dataset | Animal | Source | Hemispheres |
|---|---|---|---|
| `male-cns:v1.0` | male | neuPrint | L and R |
| `manc:v1.2.3` | male | neuPrint | L and R |
| `fanc:v604` | **female** | local tables, converted | **L only** |

male-CNS and MANC are two different male flies. FANC (Female Adult Nerve Cord) is a
third animal and a different sex, which is what lets the result be tested beyond the
two males.

## Quickstart

```bash
pip install -r requirements.txt
python run_hcs_similarity.py            # reproduces every figure from the cache
```

No neuPrint account is needed. The cached query results for the two neuPrint datasets
are committed under `data/` (~640 KB), so a fresh clone reproduces their figures and
tables exactly.

**FANC is not distributed with this repo** (see *FANC* below). Without it the pipeline
runs on the two neuPrint datasets; with it, every figure gains a third panel. To build
it from a local copy of the source tables:

```bash
python tools/fanc_to_cache.py --src /path/to/FANC_notebook
```

```bash
python run_hcs_similarity.py --refresh          # re-query neuPrint (needs a token)
python run_hcs_similarity.py --out /some/dir    # write elsewhere
```

`--refresh` needs a neuPrint token:

```bash
export NEUPRINT_APPLICATION_CREDENTIALS='<token>'   # neuprint.janelia.org, Account menu
```

`HCS_DATA` and `HCS_OUT` override the input and output roots if you prefer env vars.

## What the neurons are

**HCS** — everything entering on the haltere nerve (`DMetaN`) whose type is in the
`SApp` series: 339 cells in male-cns, 333 in MANC, against ~340 campaniform sensilla
described anatomically on the haltere, which is the main evidence the definition is
right. Roughly **45% carry the generic `SApp` type with no field assignment**, which
limits how far the *which sensory variable* question can be pushed.

In FANC the equivalent set is the 123 afferents typed `haltere_L`, of which 82 make
direct contact with a steering motor neuron above threshold. FANC carries no `SApp`
field subtypes — its afferents are a flat population.

**Motor neurons** — b1, b2, tp1, tp2, tpn. b3 is analysed as an addendum in its own
cache slot, so adding it cannot disturb the five-neuron result; the driver asserts
that rather than assuming it.

## Method

Every edge is divided by the **target's** total postsynaptic count, so a value is an
input *fraction*. Without that, motor neurons would rank by size (b2 has ~15.6k input
sites against tp1's ~9k). Edges below `MIN_SYN = 3` synapses are dropped.

Each motor neuron's input vector is then sqrt-compressed — these weight distributions
are heavy-tailed, and the sqrt stops one dominant partner deciding the whole score —
and scaled to unit length. Scaling is **per motor neuron**, which is why adding a
sixth column cannot move any existing pairwise value. Similarity is cosine.

Left and right are computed separately and averaged. It has to be the *scores* that
are averaged, not the input vectors: left and right HCS are disjoint feature sets, so
the vectors do not live in a common space at all.

The code is **blind by construction** — it knows the motor neurons only as an
unordered tuple of type strings and never branches on their identity. The
reference-vs-query reading (is tp1, tp2 or tpn closest to b1/b2?) happens only in
`hcs_similarity.ipynb`, after the matrices and controls exist.

## Controls

- **L/R reliability** — the correlation between the left and right similarity
  matrices, printed under every similarity panel (r = 0.98–0.99). FANC has one
  hemisphere, so its panel reads `single hemisphere` instead.
- **L/R mutual-nearest** — interneuron `group` is bilateral, so re-expressing the
  premotor matrix over `(ipsi/contra, group)` gives a space both hemispheres live in.
  Each motor neuron's left and right copy must then be mutual nearest neighbours. If
  this fails the normalisation is wrong and nothing downstream is interpretable.
  Written to `controls.csv`; currently PASS in both datasets.

## Figures

Written to `results/`, and gathered by format into `results/figures/{png,pdf,svg}/`.
Per-dataset figures are prefixed with the dataset name there.

| Figure | What it shows |
|---|---|
| `input_heatmap` | HCS cells × motor neurons for one connectome, left and right as separate column blocks. One row per sensillum. |
| `input_heatmap_with_b3` | The same, with b3 in the motor neuron set. |
| `similarity_1hop_annotated` / `_clean` | Motor neuron × motor neuron cosine similarity, L/R averaged, both connectomes side by side. `_clean` drops the per-cell numbers. |
| `similarity_1hop_with_b3_*` | The same for the six-motor-neuron set. |
| `input_map_by_connectome` | Direct input in every connectome on one shared row-ordering rule and one shared colour scale. |
| `input_map_by_connectome_with_b3` / `_narrow` | The same with b3; `_narrow` is half-width for a multi-panel layout. |
| `*_counts*` | Every input map and heatmap also exists in a **raw synapse count** version, colour bar in synapses rather than input fraction. |

### Figures carry no titles

Each panel shows only its dataset name, its axis labels and the colour bar. Everything
descriptive belongs in the manuscript caption rather than baked into the raster, so the
same file can be reused across a paper, a talk and a poster. `svg` and `pdf` keep text
editable (`svg.fonttype="none"`, `pdf.fonttype=42`).

**What a caption must state, because the figure no longer does:**

- The level is **direct (monosynaptic) HCS → motor neuron** connections, `MIN_SYN = 3`.
- Similarity is **cosine** on sqrt-compressed, per-column unit-scaled input fractions,
  **averaged over left and right** where both exist. L/R reliability is r = 0.99
  (male-CNS) and r = 0.98 (MANC); **FANC is one hemisphere and has no such check**.
- Grey `n/a` cells are **undefined, not zero** — that motor neuron has no direct input
  to compare. In male-CNS every tp2 value additionally rests on a single hemisphere; see
  `similarity_1hop_n_hemispheres.csv`.
- In the input maps, rows are ordered by a **rule fixed in advance** — side, then the
  motor neuron each sensillum most strongly contacts, then descending strength — not by
  clustering. Independent clustering gave the connectomes unrelated row orders, so the
  same bands could not be read across panels. Rows are ordered independently per
  connectome, so bands align but individual rows do not.
- The input maps use **one shared colour scale across panels**, but input denominators
  are not computed identically between connectomes (proofreading completeness differs,
  and FANC's counts unproofread fragments). **Compare band structure, not absolute
  shade.** The similarity figures are immune, since each column is unit-scaled alone.

### Fractions and raw counts

Every input figure exists twice: as an **input fraction** (the default, and what all
similarities are computed on) and as a **raw synapse count** (`*_counts*`).

Fractions are the analysis quantity, because dividing by the target's `post` is what
stops the comparison from ranking motor neurons by size — b2 has ~15.6k input sites
against tp1's ~9k. Counts are the *measured* quantity, and they answer what a fraction
hides: whether a strong-looking connection rests on 3 synapses or 40.

Read them together, and read neither alone across connectomes. The two orderings differ
because they answer different questions — FANC has the largest input *fractions* but
MANC the largest raw *counts*, since the denominators and the proofreading completeness
both differ. The count figures reintroduce exactly the size confound normalisation
removes, which is why they are supplementary rather than primary.

Every figure ships with the table that produced it (`similarity_1hop_LRmean.csv`,
`input_1hop_ipsi.csv`, `matrix_1-hop.csv`, …). Filenames retain `1hop` for
continuity with earlier drafts, though only direct connections are analysed.

## FANC

FANC results are computed locally and **deliberately not committed**: `data/fanc/` and
`results/fanc/` are gitignored. Of the source files, only `wing_mn_segIDs_to_make_public`
and `wing_premns_to_make_public` are marked public by the collaborator who supplied them;
the connectivity table this analysis needs is not, and FANC access is community-gated
under a Code of Conduct. The published figures are therefore two-connectome. Do not
commit FANC-derived tables or figures without explicit sign-off.

`tools/fanc_to_cache.py` converts the source tables into the same six-CSV cache schema
the neuPrint datasets use, so no analysis code contains a FANC branch. Provenance:
materialization **v604**, dated 2023-05-25, received from a collaborator, not re-derived
from CAVE. FANC root IDs are mutable under proofreading, so the pinned materialization is
what makes it reproducible.

Three things the converter handles, each of which would otherwise corrupt the result:

- **`haltere_L` has 130 rows but 123 cells.** One afferent appears eight times with
  identical values. Summing rather than deduplicating inflates b1 from 385 synapses to
  644 — a 67% overstatement. The converter deduplicates and asserts the counts.
- **The source adjacency matrix is already thresholded at weight ≥ 3.** Edges are
  therefore rebuilt from the per-synapse table so `MIN_SYN` means the same thing in all
  three datasets. The converter asserts that this reproduces the collaborator's matrix
  exactly at weight ≥ 3.
- **`b3_u` and `dtpmn_u`** are mapped to b3 and tp1 following the collaborator's own
  notebook. The `_u` suffix flags these two identifications as **uncertain upstream** —
  and they are precisely the two motor neurons the conclusions lean on hardest.

## Caveats

- **tp2 has essentially no direct haltere input** — one synapse in male-cns, and in
  that dataset every tp2 value rests on a single hemisphere (see
  `similarity_1hop_n_hemispheres.csv`). This analysis is therefore close to silent
  about tp2. That is a statement
  about the measurement, not about tp2.
- **~45% of HCS carry no field assignment**, so the analysis can show two motor
  neurons sample overlapping sensilla but often not *which* field.
- **n = 3 animals: two male, one female.** Left/right is a within-animal check, not a
  replicate, and FANC provides only one hemisphere.
- **Input denominators are not identical across connectomes.** MANC percentages run
  consistently higher than male-cns (denser sensory proofreading), and FANC's denominator
  counts synapses from unproofread fragments. Compare rank orders and band structure
  across datasets, not absolute values. The similarity figures are immune to this, since
  each motor neuron's column is unit-scaled independently.
- **FANC covers 123 afferents against ~340**, so its absence of a connection is weaker
  evidence than its presence.
- **Undefined ≠ zero.** Where a motor neuron has no input its similarity is `n/a`
  throughout, and those pairs are dropped from correlations rather than imputed.
  Imputing zero would manufacture agreement.
- `MIN_SYN = 3` and the sqrt compression are choices, fixed before any result was
  inspected.

## Files

```
hcs_lib.py               queries, caching, matrices, similarity
hcs_plots.py             the three heatmap figures
run_hcs_similarity.py    driver: writes every figure and table
hcs_similarity.ipynb     narrative walkthrough and the un-blinded reading
tools/fanc_to_cache.py   converts the FANC source tables into the cache schema
data/                    cached query results (neuPrint committed; FANC gitignored)
results/                 figures and tables (regenerated, gitignored)
```

`data/<dataset>/cache*/e1.csv` is retained but unread: it held HCS→interneuron edges
for a two-hop analysis this pipeline no longer computes. It is kept so the cache
schema is stable and two-hop could be restored without re-querying.

## Data sources

- **male-CNS** v1.0 — Janelia FlyEM, <https://neuprint.janelia.org>
- **MANC** v1.2.3 — Takemura et al. 2024, *eLife*; <https://neuprint.janelia.org>
- **FANC** v604 — Azevedo et al. 2024, *Nature*; community-gated, not redistributed here
