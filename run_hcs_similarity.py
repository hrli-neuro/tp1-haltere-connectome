"""
Driver: runs the direct-connection analysis on both connectomes and writes every
figure and table.

    python run_hcs_similarity.py                 # reads the committed cache, no token
    python run_hcs_similarity.py --refresh       # re-query neuPrint (needs a token)
    python run_hcs_similarity.py --out DIR       # write somewhere other than results/

Blind to the reference/query split: this script treats all motor neurons identically
and never groups them.  The b1/b2-vs-tp reading happens afterwards, in
hcs_similarity.ipynb, against the tables written here.
"""

import os
import shutil
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import hcs_lib as H
import hcs_plots as P

OUT = H.OUT_DIR
DATASET_ORDER = ("male-cns", "manc", "fanc")
# Queried datasets, as opposed to FANC which is converted from local tables.  The
# mancBodyid cross-reference and the b1/b2/b3 body-ID table only exist for these.
NEUPRINT = tuple(n for n in DATASET_ORDER if H.DATASETS[n]["source"] == "neuprint")


def _corr(x, y):
    """Pearson r over the pairs defined in BOTH vectors.  Undefined similarities
    (a motor neuron with no input) are NaN, not zero, and dropping them is the only
    honest option -- imputing zero would invent agreement."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan"), int(m.sum())
    return round(float(pearsonr(x[m], y[m])[0]), 4), int(m.sum())


# ------------------------------------------------------------------- per dataset
def analyse(name, refresh=False):
    print(f"\n=== {name} ===")
    b = H.load(name, refresh=refresh)
    outdir = os.path.join(OUT, name)
    sides = H.DATASETS[name]["sides"]

    direct = H.direct_matrix(b)
    premotor = H.premotor_matrix(b)            # only for the control below

    sims = {sd: H.similarity(H.prep(H.side(direct, sd))) for sd in sides}

    # --- controls -------------------------------------------------------------
    ctl, rel = [], float("nan")
    if len(sides) == 2:
        L = H.offdiag(sims["L"]).sim.values
        R = H.offdiag(sims["R"]).sim.values
        rel, n = _corr(L, R)
        ctl.append({"control": "L/R reliability", "analysis": "direct",
                    "r": rel, "n_pairs": n})
        print(f"  L/R reliability r = {rel:+.3f}  (n={n})")

        # left/right positive control, interneuron `group` feature space.  Each motor
        # neuron's left and right copy must be each other's nearest neighbour; if this
        # fails the normalisation is wrong and nothing downstream is interpretable.
        Pg = H.collapse_side_relative(premotor, b.ins, "group")
        S10 = H.similarity(H.prep(Pg))
        ok = all(S10[c].drop(c).idxmax() ==
                 f"{c.rsplit('_',1)[0]}_{'R' if c.endswith('_L') else 'L'}"
                 for c in S10.columns)
        ctl.append({"control": "L/R mutual-nearest (premotor, IN group)",
                    "analysis": "premotor", "r": float(ok)})
        print(f"  L/R positive control: {'PASS' if ok else 'FAIL'}")
    else:
        ok = None
        print(f"  single hemisphere ({sides[0]}): L/R controls not applicable")

    # --- write ------------------------------------------------------------------
    os.makedirs(outdir, exist_ok=True)
    if ctl:
        pd.DataFrame(ctl).to_csv(os.path.join(outdir, "controls.csv"), index=False)
    direct.to_csv(os.path.join(outdir, "matrix_1-hop.csv"), float_format="%.6g")
    premotor.to_csv(os.path.join(outdir, "matrix_premotor.csv"), float_format="%.6g")
    # Raw synapse counts alongside the fractions.  Not used for any similarity -- the
    # size confound that `post` removes is real -- but a fraction cannot tell you
    # whether a connection rests on 3 synapses or 40, and that is worth being able to
    # look up.
    counts = H.direct_matrix(b, normalise=False)
    counts.to_csv(os.path.join(outdir, "matrix_1-hop_counts.csv"), float_format="%g")

    # --- figures ----------------------------------------------------------------
    P.fig_input_heatmap(direct, outdir, b.name, b.hcs)
    P.fig_input_heatmap(counts, outdir, b.name, b.hcs, stem="input_heatmap_counts",
                        value_label="synapses")
    return ok, sims, rel


def _write_lr_mean(sims, outdir, stem):
    """The table behind each similarity figure: L/R mean, and how many sides backed it.

    A one-hemisphere dataset writes that hemisphere's own scores with n_hemispheres = 1,
    so the count column stays the honest record of what each value rests on.
    """
    L = P._reorder(sims["L"])
    R = P._reorder(sims["R"]) if "R" in sims else None
    # positional; adding the two frames directly would align 'b1_L' against 'b1_R' as
    # distinct labels and silently produce an all-NaN union rather than a mean
    mean, n_sides = P._avg_lr(L, R)
    lab = P._labels(L)
    pd.DataFrame(mean, index=lab, columns=lab).to_csv(
        os.path.join(outdir, f"{stem}_LRmean.csv"), float_format="%.4g")
    pd.DataFrame(n_sides, index=lab, columns=lab).to_csv(
        os.path.join(outdir, f"{stem}_n_hemispheres.csv"))


def collect_figures():
    """Gather every figure into one browsable folder, grouped by format.

    The per-dataset folders keep their figures next to the tables that produced them;
    this is a copy, refreshed on each run, so it never goes stale.  Grouping by
    format matches how they get used -- png for slides, pdf/svg for a manuscript.
    """
    dest = os.path.join(OUT, "figures")
    if os.path.isdir(dest):
        shutil.rmtree(dest)                     # rebuild so renamed figures do not linger
    n = 0
    for ext in ("png", "pdf", "svg"):
        os.makedirs(os.path.join(dest, ext), exist_ok=True)
    for src_dir, prefix in [(OUT, ""), *((os.path.join(OUT, d), f"{d}_") for d in DATASET_ORDER)]:
        if not os.path.isdir(src_dir):
            continue
        for f in sorted(os.listdir(src_dir)):
            ext = f.rpartition(".")[2]
            if ext in ("png", "pdf", "svg"):
                shutil.copy2(os.path.join(src_dir, f),
                             os.path.join(dest, ext, f"{prefix}{f}"))
                n += 1
    print(f"\ncollected {n} figure files into {dest}")
    for ext in ("png", "pdf", "svg"):
        print(f"  {ext}/  {len(os.listdir(os.path.join(dest, ext)))} files")
    return dest


def b3_panel(out5):
    """Six-motor-neuron version (b3 added), L/R averaged.

    Runs on its own cache slot so the five-neuron pipeline above is untouched.  The
    five shared motor neurons must come out with identical similarities -- prep()
    normalises each column independently, so a sixth column cannot move them -- and
    that is asserted rather than assumed.
    """
    print("\n=== b3 addendum (6 motor neurons) ===")
    sims, rel = {}, {}
    for name in DATASET_ORDER:
        b = H.load(name, mn_types=H.MN_TYPES_WITH_B3, tag="with_b3")
        D = H.direct_matrix(b)
        sims[name] = {sd: H.similarity(H.prep(H.side(D, sd)))
                      for sd in H.DATASETS[name]["sides"]}
        if "R" in sims[name]:
            r, n = _corr(H.offdiag(sims[name]["L"]).sim.values,
                         H.offdiag(sims[name]["R"]).sim.values)
            rel[name] = r
            print(f"  {name:9s} {len(b.mns)} MNs, L/R reliability r = {r:+.3f} (n={n})")
        else:
            print(f"  {name:9s} {len(b.mns)} MNs, single hemisphere")

        # consistency gate against the five-neuron run
        old, new = out5[name][1]["L"], sims[name]["L"]
        shared = [c for c in new.columns if c in old.columns]
        assert np.allclose(old.loc[shared, shared].to_numpy(),
                           new.loc[shared, shared].to_numpy(), equal_nan=True), \
            f"{name}: adding b3 changed the existing five-neuron similarities"
    print("  the five shared motor neurons are unchanged by adding b3")

    common = dict(rel=rel)
    P.fig_similarity_direct(sims, OUT, DATASET_ORDER,
                            stem="similarity_1hop_with_b3_annotated",
                            annotate=True, **common)
    P.fig_similarity_direct(sims, OUT, DATASET_ORDER,
                            stem="similarity_1hop_with_b3_clean",
                            annotate=False, **common)
    for name in DATASET_ORDER:
        _write_lr_mean(sims[name], os.path.join(OUT, name), "similarity_1hop_with_b3")


def b3_bodyid_table(refresh=False):
    """Which haltere sensilla feed b1, b2 and b3, cell by cell, in each connectome.

    b3 takes as much direct HCS input as b1 yet shows ~0.00 input similarity to it.
    This table gives the reason at cell level: the populations barely overlap.  All
    edges are pulled at weight >= 1 and both groupings are reported, because whether
    the handful of male-cns crossovers count at all depends on the threshold -- that
    is the reader's call, not something to bake in.
    """
    print("\n=== HCS -> b1 / b2 / b3 body IDs ===")
    MNS = ["b1 MN", "b2 MN", "b3 MN"]
    for name in NEUPRINT:
        try:
            e = H.hcs_to_mn_edges(name, MNS, refresh=refresh)
        except (RuntimeError, ImportError):
            # RuntimeError = no token; ImportError = neuprint not installed.  Either way
            # this one table cannot be built, and it must not take the rest of the run
            # with it -- every figure is already derived from the committed cache.
            print(f"  {name:9s} SKIPPED - no cached edge table, and cannot query "
                  f"(needs neuprint-python and a token).\n"
                  f"            run once with --refresh to populate "
                  f"data/{name}/hcs_to_mn_edges.csv; it is tokenless thereafter.")
            continue
        e = e.copy()
        e["mn"] = e.mn.str.replace(" MN", "", regex=False)

        wide = (e.pivot_table(index=["bodyId", "type", "side"], columns="mn",
                              values="w", aggfunc="sum", fill_value=0)
                  .reindex(columns=["b1", "b2", "b3"], fill_value=0)
                  .rename(columns=lambda m: f"w_{m}").reset_index())
        # `targets` is derived from the edge list and the w_* columns from the pivot,
        # so the verification below compares two independent derivations
        for thr, col in ((1, "targets"), (H.MIN_SYN, "targets_min3")):
            g = (e[e.w >= thr].groupby("bodyId").mn
                 .apply(lambda x: ",".join(sorted(set(x)))).rename(col))
            wide = wide.merge(g, on="bodyId", how="left")
        wide["targets_min3"] = wide.targets_min3.fillna("")
        contra = e[e.side != e.mn_side].groupby("bodyId").size()
        wide["contra"] = wide.bodyId.isin(contra.index)

        wide["_tot"] = wide[["w_b1", "w_b2", "w_b3"]].sum(axis=1)
        wide = (wide.sort_values(["targets", "side", "_tot"],
                                 ascending=[True, True, False])
                    .drop(columns="_tot"))

        assert (wide[["w_b1", "w_b2", "w_b3"]].sum(axis=1) > 0).all()
        for _, r in wide.iterrows():
            assert set(r.targets.split(",")) == {
                m for m in ("b1", "b2", "b3") if r[f"w_{m}"] > 0}, r.bodyId

        path = os.path.join(OUT, name, "hcs_to_b1b2b3.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        wide.to_csv(path, index=False)
        counts = wide.targets.value_counts().to_dict()
        cross = {t: sum(("b3" in v) and ("b1" in v or "b2" in v)
                        for v in wide[t] if v) for t in ("targets", "targets_min3")}
        print(f"  {name:9s} {len(wide):3d} HCS · {counts}")
        print(f"            b3 together with b1/b2:  "
              f">=1 syn: {cross['targets']}   >={H.MIN_SYN} syn: {cross['targets_min3']}")
        print(f"            wrote {os.path.basename(path)}")


def b3_input_maps():
    """The input maps rebuilt with b3 in the motor neuron set.

    Same code paths and the same deterministic row rule as the five-neuron versions,
    so the pairs can be read against each other and the only difference is the extra
    column.  b3 sorts last via MN_ORDER, which puts its (large) block of sensilla at
    the bottom of each side rather than splitting the b1/b2 block in half.
    """
    print("\n=== input maps with b3 ===")
    bundles = {n: H.load(n, mn_types=H.MN_TYPES_WITH_B3, tag="with_b3")
               for n in DATASET_ORDER}
    for name, b in bundles.items():
        d = os.path.join(OUT, name)
        P.fig_input_heatmap(H.direct_matrix(b), d, b.name, b.hcs,
                            stem="input_heatmap_with_b3")
        P.fig_input_heatmap(H.direct_matrix(b, normalise=False), d, b.name, b.hcs,
                            stem="input_heatmap_counts_with_b3", value_label="synapses")
    ipsi = {n: H.ipsi_matrix(b) for n, b in bundles.items()}
    hcs = {n: b.hcs for n, b in bundles.items()}
    P.fig_input_map_by_dataset(ipsi, hcs, OUT, DATASET_ORDER,
                               stem="input_map_by_connectome_with_b3")
    cnt = {n: H.ipsi_matrix(b, normalise=False) for n, b in bundles.items()}
    P.fig_input_map_by_dataset(cnt, hcs, OUT, DATASET_ORDER,
                               stem="input_map_by_connectome_counts_with_b3",
                               value_label="synapses (shared scale)")
    # same figure at half the column width, for a narrow slot in a multi-panel layout
    P.fig_input_map_by_dataset(ipsi, hcs, OUT, DATASET_ORDER,
                               stem="input_map_by_connectome_with_b3_narrow",
                               panel_width=1.5)
    for n in DATASET_ORDER:
        ipsi[n].to_csv(os.path.join(OUT, n, "input_1hop_ipsi_with_b3.csv"),
                       float_format="%.5g")
        cnt[n].to_csv(os.path.join(OUT, n, "input_1hop_ipsi_counts_with_b3.csv"),
                      float_format="%g")


def main():
    global OUT
    refresh = "--refresh" in sys.argv
    if "--out" in sys.argv:
        OUT = os.path.abspath(sys.argv[sys.argv.index("--out") + 1])
    os.makedirs(OUT, exist_ok=True)
    print(f"data:    {H.DATA_DIR}\noutputs: {OUT}")

    out = {}
    for name in DATASET_ORDER:
        out[name] = analyse(name, refresh)

    # ------------------------------------------------- cross-dataset concordance
    # Printed as a diagnostic only: male-cns and MANC are two different flies, so
    # agreement between them speaks to whether the structure generalises.
    # Restricted to the two-hemisphere datasets: a one-sided connectome has no L/R mean,
    # and correlating its single hemisphere against a two-sided mean would compare
    # quantities of different provenance.
    print("\n=== cross-dataset concordance (two different flies) ===")
    v = {n: np.nanmean([H.offdiag(out[n][1][sd]).sim.values
                        for sd in H.DATASETS[n]["sides"]], axis=0) for n in NEUPRINT}
    r, n = _corr(v["male-cns"], v["manc"])
    print(f"  direct    r = {r:+.3f}  (n={n} pairs with both defined)")

    # ------------------------------------- five motor neurons, per-panel scale bars
    print("\n=== similarity, five motor neurons ===")
    sims_by_ds = {n: out[n][1] for n in DATASET_ORDER}
    rel = {n: out[n][2] for n in DATASET_ORDER}
    # mark_single is left off: the asterisks landed almost entirely on male-cns tp2 and
    # cluttered the panel.  The count they encoded is still written to
    # similarity_1hop_n_hemispheres.csv, so the caveat is recorded, just not drawn.
    common = dict(rel=rel)
    P.fig_similarity_direct(sims_by_ds, OUT, DATASET_ORDER,
                            stem="similarity_1hop_annotated",
                            annotate=True, **common)
    P.fig_similarity_direct(sims_by_ds, OUT, DATASET_ORDER,
                            stem="similarity_1hop_clean",
                            annotate=False, **common)
    for n in DATASET_ORDER:
        _write_lr_mean(sims_by_ds[n], os.path.join(OUT, n), "similarity_1hop")

    # ------------- direct input, every connectome, one shared row-ordering rule
    print("\n=== input map, every connectome ===")
    bundles = {n: H.load(n) for n in DATASET_ORDER}
    ipsi = {n: H.ipsi_matrix(b) for n, b in bundles.items()}
    hcs = {n: b.hcs for n, b in bundles.items()}
    for n in DATASET_ORDER:
        print(f"  {n:9s} {(ipsi[n].abs().sum(axis=1) > 0).sum()} HCS with direct input")
    P.fig_input_map_by_dataset(ipsi, hcs, OUT, DATASET_ORDER,
                               stem="input_map_by_connectome")
    cnt = {n: H.ipsi_matrix(b, normalise=False) for n, b in bundles.items()}
    P.fig_input_map_by_dataset(cnt, hcs, OUT, DATASET_ORDER,
                               stem="input_map_by_connectome_counts",
                               value_label="synapses (shared scale)")
    for n in DATASET_ORDER:
        ipsi[n].to_csv(os.path.join(OUT, n, "input_1hop_ipsi.csv"), float_format="%.5g")
        cnt[n].to_csv(os.path.join(OUT, n, "input_1hop_ipsi_counts.csv"), float_format="%g")

    # the correspondence is not used by any figure; this records why
    audit, summ = H.crossref_audit({n: bundles[n] for n in NEUPRINT})
    audit.to_csv(os.path.join(OUT, "hcs_crossref_audit.csv"), index=False)
    print(f"  mancBodyid audit: {summ['pairs']} pairs · side agrees "
          f"{summ['side_agree']}/{summ['pairs']} · of {summ['checkable']} checkable: "
          f"{summ.get('identical',0)} identical, {summ.get('compatible',0)} compatible, "
          f"{summ.get('contradictory',0)} contradictory  -> not used for row matching")

    # -------------------------------------- b3 addendum: six motor neurons
    b3_panel(out)
    b3_bodyid_table(refresh=refresh)
    b3_input_maps()

    collect_figures()
    print(f"\nall outputs under {OUT}")


if __name__ == "__main__":
    main()
