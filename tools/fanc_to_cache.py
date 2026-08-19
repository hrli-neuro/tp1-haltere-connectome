"""
Convert the FANC wing-motor-neuron tables into this project's cache schema.

    python tools/fanc_to_cache.py [--src DIR]

Writes data/fanc/cache/ and data/fanc/cache_with_b3/ with exactly the six CSVs that
hcs_lib.load() expects, so every downstream code path treats FANC like any other
dataset and no runtime FANC branch exists anywhere in the analysis.

PROVENANCE.  FANC = Female Adult Nerve Cord (Azevedo et al. 2024).  These tables are
materialization **v604**, dated 2023-05-25, received from a collaborator -- they are not
re-derived from CAVE here.  FANC root IDs are mutable under ongoing proofreading, so the
pinned materialization is what makes this reproducible at all.

NOT CLEARED FOR PUBLICATION.  Of the source files, only `wing_mn_segIDs_to_make_public`
and `wing_premns_to_make_public` are marked public by the collaborator; the connectivity
table this converter reads is not.  data/fanc/ and results/ are gitignored for that
reason.  Do not commit FANC-derived tables or figures without explicit sign-off.

TWO DATA HAZARDS this converter exists to handle
------------------------------------------------
1.  `haltere_L` has 130 ROWS but only 123 unique cells.  One afferent
    (648518346495794064) appears eight times with byte-identical values.  Summing the
    rows inflates b1 from 385 synapses to 644 -- a 67% overstatement -- and b2 from 352
    to 429.  The rows are deduplicated, not summed, and that is asserted below.

2.  `preMN_to_MN_wing_v604.pkl` is already thresholded at weight >= 3.  Rather than
    inherit an invisible threshold, edges are rebuilt from the per-synapse table so that
    MIN_SYN means the same thing in FANC as in male-cns and MANC.  The adjacency is still
    used as a regression check: at w >= 3 the two must agree exactly.

MOTOR NEURON IDENTITY.  `dtpmn_u` is taken as tp1 and `b3_u` as b3, following the
collaborator's own notebook.  The `_u` suffix marks these two identifications as
UNCERTAIN upstream -- which is exactly the pair the project's conclusions lean on
hardest.  Every FANC-bearing figure caption says so.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hcs_lib as H

DEFAULT_SRC = os.path.expanduser(
    "~/Documents/Dickinson_lab/Connectome/Connectivity/FANC/FANC_notebook")

# FANC label -> the type string this project uses.  b3_u and dtpmn_u carry the upstream
# uncertainty flag; see the module docstring.
FANC_MN_LABELS = {"b1": "b1 MN", "b2": "b2 MN", "b3_u": "b3 MN",
                  "dtpmn_u": "tp1 MN", "tp2": "tp2 MN", "tpn": "tpn MN"}
UNCERTAIN = ("b3_u", "dtpmn_u")

# FANC reconstructed one side only; every wing motor neuron in the source is side L.
SIDE = "L"
HCS_TYPE = "haltere_L"
IN_CLASSES = ("local", "local_intersegmental", "ascending", "descending")


def _load_sources(src):
    syn = pd.read_pickle(os.path.join(src, "syanpse_positions_wing_v604.pkl"))
    props = pd.read_pickle(os.path.join(src, "wingMN_properties_v604.pkl"))
    adj = pd.read_pickle(os.path.join(src, "preMN_to_MN_wing_v604.pkl"))
    segs = pd.read_csv(os.path.join(
        src, "dfs_saved", "wing_mn_segIDs_to_make_public_20230525.csv"))
    return syn, props, adj, segs


def _annotations(adj):
    """One row per presynaptic cell: cell_type, cell_class, putative_NT.

    This is where the 8x duplicate is removed.  Deduplicating on the index rather than
    on bodyId alone is deliberate: it would raise if the duplicate rows ever stopped
    being identical, instead of silently keeping an arbitrary one.
    """
    meta = adj.index.to_frame(index=False)
    n_rows = int((meta.cell_type == HCS_TYPE).sum())
    dup = adj[~adj.index.duplicated(keep="first")]
    ann = (dup.index.to_frame(index=False)
              .drop_duplicates("pre_pt_root_id")
              .set_index("pre_pt_root_id"))
    n_cells = int((ann.cell_type == HCS_TYPE).sum())
    print(f"  {HCS_TYPE}: {n_rows} rows -> {n_cells} unique cells")
    assert (n_rows, n_cells) == (130, 123), (
        f"expected 130 haltere rows collapsing to 123 cells, got {n_rows}/{n_cells}. "
        "The source tables changed -- re-check the duplicate afferent before trusting "
        "any FANC number.")
    return ann, dup


def _edges(syn, segs):
    """Per-(presynaptic cell, motor neuron) synapse counts from the per-synapse table."""
    label = dict(zip(segs.pt_root_id, segs.cell_type))
    assert set(segs.side) == {SIDE}, f"expected FANC MNs all on side {SIDE}"
    e = syn[["pre_pt_root_id", "post_pt_root_id"]].copy()
    e["mn"] = e.post_pt_root_id.map(label)
    e = e[e.mn.notna()]
    return (e.groupby(["pre_pt_root_id", "mn"]).size().rename("w").reset_index())


def _check_against_adjacency(edges, dup, ann):
    """The collaborator's matrix is already thresholded at w >= 3; we must reproduce it.

    Not a second source of truth -- a regression check, so a future edit to this
    converter cannot silently drift away from the numbers he validated.
    """
    hcs = ann.index[ann.cell_type == HCS_TYPE]
    a = (dup[dup.index.get_level_values("cell_type") == HCS_TYPE]
         .droplevel(["preferred_module", "cell_type", "cell_class", "putative_NT"])
         .rename_axis("pre_pt_root_id").stack().rename("w").reset_index()
         .rename(columns={"level_1": "mn"}))
    a = a[a.w > 0].sort_values(["pre_pt_root_id", "mn"]).reset_index(drop=True)
    b = (edges[edges.pre_pt_root_id.isin(hcs) & (edges.w >= 3)]
         .sort_values(["pre_pt_root_id", "mn"]).reset_index(drop=True))
    assert a.equals(b), (
        f"per-synapse edges at w>=3 ({len(b)}) do not reproduce the deduplicated "
        f"adjacency ({len(a)}); the source tables disagree")
    print(f"  regression check: {len(b)} haltere edges at w>=3 match the adjacency exactly")


def _write_slot(outdir, mn_labels, edges, ann, props, hcs_ids):
    """Write the six cache CSVs for one motor neuron set."""
    os.makedirs(outdir, exist_ok=True)
    prop = props.set_index("MN_label")
    mns = pd.DataFrame([{"bodyId": int(prop.loc[k, "MN_id"]), "type": v, "side": SIDE,
                         "post": int(prop.loc[k, "synapses_include_fragments"])}
                        for k, v in mn_labels.items()])
    mn_id = dict(zip(mn_labels, mns.bodyId))

    keep = edges[edges.mn.isin(mn_labels) & (edges.w >= H.MIN_SYN)].copy()
    keep["d"] = keep.mn.map(mn_id)

    is_hcs = keep.pre_pt_root_id.isin(hcs_ids)
    direct = (keep[is_hcs].rename(columns={"pre_pt_root_id": "s"})[["s", "d", "w"]]
              .sort_values(["s", "d"]).reset_index(drop=True))

    cls = ann.cell_class.reindex(keep.pre_pt_root_id).values
    e2 = (keep[pd.Series(cls, index=keep.index).isin(IN_CLASSES).values]
          .rename(columns={"pre_pt_root_id": "i"})[["i", "d", "w"]]
          .sort_values(["i", "d"]).reset_index(drop=True))

    # `post` is genuinely unavailable for FANC interneurons -- the source gives total
    # input only for motor neurons.  Nothing in this pipeline reads it (premotor_matrix
    # divides by the MOTOR neuron's post), so it is left blank rather than invented.
    ia = ann.reindex(e2.i.unique())
    ins = pd.DataFrame({"bodyId": ia.index, "type": ia.cell_type.values, "side": SIDE,
                        "post": pd.NA, "nt": ia.putative_NT.values,
                        "group": ia.index}).reset_index(drop=True)

    hcs = pd.DataFrame({"bodyId": sorted(hcs_ids), "type": "haltere", "side": SIDE})
    # e1 (HCS -> interneuron) is not measured in these tables and is not read by this
    # pipeline; written empty so the cache schema stays uniform across datasets.
    e1 = pd.DataFrame(columns=["s", "i", "w"])

    for name, frame in (("hcs", hcs), ("mns", mns), ("ins", ins),
                        ("direct", direct), ("e1", e1), ("e2", e2)):
        frame.to_csv(os.path.join(outdir, f"{name}.csv"), index=False)
    print(f"  {os.path.basename(outdir):16s} {len(hcs):3d} HCS · {len(mns)} MN · "
          f"{len(direct):3d} direct · {len(ins):4d} IN · {len(e2):4d} IN->MN")
    return direct, mns


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--src", default=DEFAULT_SRC, help="FANC_notebook directory")
    args = ap.parse_args()

    print(f"reading {args.src}")
    syn, props, adj, segs = _load_sources(args.src)
    ann, dup = _annotations(adj)
    edges = _edges(syn, segs)
    _check_against_adjacency(edges, dup, ann)

    missing = [k for k in FANC_MN_LABELS if k not in set(props.MN_label)]
    assert not missing, f"motor neurons absent from the source: {missing}"
    print(f"  uncertain identifications carried through: "
          f"{', '.join(f'{k} -> {FANC_MN_LABELS[k]}' for k in UNCERTAIN)}")

    hcs_ids = set(ann.index[ann.cell_type == HCS_TYPE])
    five = {k: v for k, v in FANC_MN_LABELS.items() if v in H.MN_TYPES}
    assert len(five) == 5, f"expected 5 default motor neurons, mapped {len(five)}"

    out = os.path.join(H.DATA_DIR, "fanc")
    d5, _ = _write_slot(os.path.join(out, "cache"), five, edges, ann, props, hcs_ids)
    d6, mns6 = _write_slot(os.path.join(out, "cache_with_b3"), FANC_MN_LABELS,
                           edges, ann, props, hcs_ids)

    # round-trip: nothing may be dropped that passed the threshold
    tot = edges[edges.pre_pt_root_id.isin(hcs_ids) & edges.mn.isin(FANC_MN_LABELS)
                & (edges.w >= H.MIN_SYN)].w.sum()
    assert d6.w.sum() == tot, f"direct.csv lost synapses: {d6.w.sum()} != {tot}"
    print(f"\n  round-trip: {int(tot)} haltere synapses at w>={H.MIN_SYN} onto "
          f"{len(mns6)} motor neurons, across {d6.s.nunique()} afferents")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
