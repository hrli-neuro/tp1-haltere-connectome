"""
Figures for the HCS -> steering motor neuron analysis.  All three are heatmaps.

Two families:

  * input maps      HCS cells x motor neurons, one row per sensillum
                    (fig_input_heatmap, fig_input_map_by_dataset)
  * similarity      motor neuron x motor neuron cosine, left/right averaged
                    (fig_similarity_direct)

Two colour decisions carry the meaning, and both are deliberate:

`HEAT` is the WARM ARM of RdBu_r only, near-white -> dark red.  Input fraction and
cosine similarity are magnitudes with a true zero and no negative values, so this is a
sequential encoding.  The full diverging map would put its pale band at 0.5 -- a
midpoint that means nothing here -- and paint honest zeros in saturated blue, which
reads as "strongly opposite" rather than "no shared input".

`NA_FILL` grey means UNDEFINED, and is visually distinct from every step of the ramp
(dE 10.9).  A motor neuron with no direct input at all has no vector to compare, which
is not the same as comparing it and finding nothing in common.  tp2 is exactly this
case in every dataset, so the distinction is load-bearing rather than decorative:
across a whole panel of `n/a` cells, painting them as 0.00 would assert a measurement
that was never made.

These are print figures for a paper, committed to the light surface.

They carry NO titles and no explanatory text: every panel shows only its dataset name,
the axis labels, and the colour bar.  Everything else -- what the level is, how rows are
ordered, what the shared scale does and does not license -- belongs in the manuscript
caption, not baked into the raster.  The facts that used to be printed on the figures are
in the README and in the CSV written beside each one.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ------------------------------------------------------------------ design tokens
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
BASELINE = "#c3c2b7"

HEAT = LinearSegmentedColormap.from_list(
    "rdbu_warm", plt.get_cmap("RdBu_r")(np.linspace(0.5, 1.0, 256)))

NA_FILL = "#8e8c85"          # undefined similarity; dE 10.9 from every step of RdBu_r
NA_INK = "#ffffff"           # 3.37:1 on that fill
DISPLAY = {"male-cns": "male-CNS", "manc": "MANC", "fanc": "FANC"}   # print names

plt.rcParams.update({
    "pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 9,
    "font.family": "sans-serif", "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
})

# Motor neuron names are identifiers, not chrome: they are set in INK everywhere,
# overriding the muted rcParams tick colour, which is meant for numeric axes.
SHORT = lambda t: t.replace(" MN", "")

# Fixed display order for every matrix figure.  Alphabetical sorting would drop b3
# between b2 and tp1, implying it groups with the basalare motor neurons; it in fact
# shares almost no haltere input with any of them, so it sits at the end and the
# original five keep the order they have in all the other figures.
MN_ORDER = ("b1", "b2", "tp1", "tp2", "tpn", "b3")


def _ink_on(rgba):
    """White or dark ink, chosen from the cell's own luminance rather than a
    magic value threshold, so labels stay legible across the whole ramp."""
    r, g, b = rgba[:3]
    f = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    lum = 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)
    return "#ffffff" if (1.05 / (lum + 0.05)) >= ((lum + 0.05) / 0.05) else INK


def _row_order(M, side, order=MN_ORDER):
    """Deterministic row order, no clustering: side, then dominant target, then strength.

    Independent per-dataset clustering gave the two connectomes unrelated row orders --
    even the left/right block order flipped, which is just arbitrary dendrogram
    orientation rather than anything about the data.  This rule is fixed in advance and
    stated in one sentence, so the same band structure appears at the same height in
    both panels and they can be read against each other.
    """
    rank = {n: i for i, n in enumerate(order)}
    types = sorted({t for t, _ in M.columns}, key=lambda t: rank.get(SHORT(t), 99))
    # each cell scored against the motor neurons on its OWN side
    ipsi = pd.DataFrame(0.0, index=M.index, columns=types)
    for t in types:
        for sd in ("L", "R"):
            if (t, sd) in M.columns:
                m = (side == sd).to_numpy()
                ipsi.loc[m, t] = M.loc[m, (t, sd)].to_numpy()
    return _row_order_ipsi(ipsi, side, order)


def _row_order_ipsi(M, side, order=MN_ORDER, by_side=True):
    """The same rule, over a matrix whose columns are already the cell's own side.

    Split out so the per-dataset figures and the cross-connectome comparison order
    their rows through one implementation and cannot drift apart.

    `by_side` controls whether hemisphere is the outermost key.  It must be True when
    the figure has literal left/right COLUMN blocks (fig_input_heatmap), since rows
    have to group with the columns they fill.  It should be False once the columns are
    ipsilateral-collapsed (fig_input_map_by_dataset): there a left sensillum driving
    left-b1 and a right one driving right-b1 are doing the same thing, so splitting
    them yields two separate b1 bands and the whole sequence appears twice for no
    reason.
    """
    rank = {n: i for i, n in enumerate(order)}
    M = M[sorted(M.columns, key=lambda t: rank.get(SHORT(t), 99))]
    dom = M.to_numpy().argmax(axis=1)
    strength = M.to_numpy().max(axis=1)
    key = pd.DataFrame({"side": (side.reindex(M.index).to_numpy() != "L").astype(int),
                        "dom": dom, "neg": -strength}, index=M.index)
    return key.sort_values(["side", "dom", "neg"] if by_side else ["dom", "neg"]).index


def _reorder(S, order=MN_ORDER):
    """Reindex a similarity matrix into MN_ORDER, keeping anything unlisted at the end."""
    base = {c: (c[:-2] if str(c).endswith(("_L", "_R")) else str(c)) for c in S.columns}
    rank = {n: i for i, n in enumerate(order)}
    cols = sorted(S.columns, key=lambda c: (rank.get(base[c], len(rank)), base[c]))
    return S.loc[cols, cols]


def _labels(S):
    """Similarity frames are labelled 'b1_L'.  Inside a single-hemisphere panel the
    side suffix is already in the panel title, so drop it."""
    lab = [str(c) for c in S.columns]
    return [c[:-2] if c.endswith(("_L", "_R")) else c for c in lab]


def _save(fig, outdir, stem):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(os.path.join(outdir, f"{stem}.{ext}"), dpi=200, bbox_inches="tight",
                    facecolor=SURFACE)
    plt.close(fig)
    print(f"    wrote {stem}.{{pdf,svg,png}}")


def _cell_grid(ax, nx, ny):
    """Hairlines between cells.  The pale end of the ramp is ~1.05:1 against the
    surface, so without these a zero-valued cell dissolves into the page and the
    matrix loses its outline."""
    ax.set_xticks(np.arange(-.5, nx, 1), minor=True)
    ax.set_yticks(np.arange(-.5, ny, 1), minor=True)
    ax.grid(which="minor", color=BASELINE, linewidth=0.7)
    ax.tick_params(which="minor", length=0)
    ax.set_axisbelow(False)


def _bare(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def _avg_lr(L, R=None):
    """Mean of the available hemispheres, plus how many were defined per cell.

    It has to be the *scores* that are averaged, not the input vectors: left and
    right HCS are disjoint feature sets, so the vectors share no space.  The
    n_sides count is what lets a caller flag a cell that rests on one hemisphere
    only, which a bare nanmean would silently present as a two-sided measurement.

    `R=None` is for a connectome reconstructed in one hemisphere (FANC): the result is
    that hemisphere's own scores, and n_sides is 1 wherever a score exists, so those
    cells are marked in exactly the same way as a cell whose other side happened to be
    undefined.  The figure then makes no claim it cannot support.
    """
    frames = [L] if R is None else [L, R]
    A = np.stack([f.to_numpy(float) for f in frames])
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(A, axis=0)
    return mean, np.isfinite(A).sum(axis=0)


# ============================================================== 1. input heatmap
def fig_input_heatmap(M, outdir, dataset, hcs, stem="input_heatmap",
                      value_label="input fraction"):
    """HCS cells x motor neurons, direct connections only.

    Every row is one HCS cell, and a blank half-row is a measured zero rather than
    padding: of the ~100 cells with direct input, essentially all contact only the
    motor neurons on their own side.
    """
    fig, ax = plt.subplots(1, 1, figsize=(5.6, 6.4), dpi=170)
    M = M.loc[M.abs().sum(axis=1) > 0]
    # Group the columns by hemisphere rather than by motor neuron.  HCS are
    # side-specific -- a left sensillum contacts left motor neurons and almost
    # nothing else -- so interleaving b1_L, b1_R, b2_L, b2_R ... breaks that
    # structure into a checkerboard.  Side-major ordering lets it read as two
    # blocks, which is what the connectivity actually looks like.
    rank = {n: i for i, n in enumerate(MN_ORDER)}
    M = M[sorted(M.columns, key=lambda c: (c[1], rank.get(SHORT(c[0]), 99)))]
    sides = [c[1] for c in M.columns]
    split = sides.index("R") if "R" in sides else len(sides)

    side = (hcs.drop_duplicates("bodyId").set_index("bodyId")
               .side.reindex(M.index))
    M = M.loc[_row_order(M, side)]
    vmax = np.percentile(M.to_numpy()[M.to_numpy() > 0], 99) if (M.to_numpy() > 0).any() else 1
    im = ax.imshow(M.to_numpy(), aspect="auto", cmap=HEAT, vmin=0, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels([SHORT(t) for t, _ in M.columns], fontsize=8.5, color=INK)
    ax.set_ylabel(f"HCS cells with input  (n = {len(M)})", fontsize=8.5)
    ax.set_yticks([])
    ax.set_title(DISPLAY.get(dataset, dataset), fontsize=12, fontweight="bold", pad=22)

    # divider plus one header per block, so the side is stated once instead of
    # being repeated under every column
    if 0 < split < len(sides):
        ax.axvline(split - 0.5, color=INK, lw=1.4, zorder=5)
        for x0, x1, name in ((0, split, "left"), (split, len(sides), "right")):
            # y is in axes fractions here, so 1.0 is the TOP of the heatmap; the
            # headers sit just above it, under the panel title's 22pt pad
            ax.text((x0 + x1 - 1) / 2, 1.012, name, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=10, fontweight="bold",
                    color=INK, clip_on=False)
    _bare(ax)
    for sp in ("top", "bottom", "left", "right"):     # keep the block's outline
        ax.spines[sp].set_visible(True); ax.spines[sp].set_color(BASELINE)
        ax.spines[sp].set_linewidth(0.7)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(value_label, fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, labelsize=7.5)
    _save(fig, outdir, stem)


# ==================================================== 2. similarity, one per dataset
def fig_similarity_direct(sims_by_ds, outdir, datasets, rel=None,
                          stem="similarity_1hop", annotate=True, mark_single=False):
    """Motor neuron x motor neuron cosine similarity, L/R averaged, datasets side by side.

    The two hemispheres are averaged rather than shown separately because they are
    two measurements of the same thing (L/R reliability r = 0.88-0.99).  Note it has
    to be the *scores* that are averaged, not the input vectors: left and right HCS
    are disjoint feature sets -- no cell-level L<->R match exists in either dataset --
    so the vectors do not live in a common space and cannot be averaged at all.

    A dataset reconstructed in one hemisphere only (FANC) is labelled `single
    hemisphere` instead of carrying an L/R reliability figure, and its scores are that
    hemisphere's own rather than a mean.

    Cells resting on a single hemisphere *within a two-sided dataset* (the other being
    undefined) can be marked with an asterisk via `mark_single` -- male-cns tp2 is the
    case.  The mark is deliberately NOT applied to a wholly one-sided dataset, where it
    would land on every cell and repeat what the panel label already says.  The count is
    written to similarity_*_n_hemispheres.csv either way, so the caveat is never lost.
    `annotate=False` drops the per-cell numbers.
    """
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n + 0.4, 4.5), dpi=300, squeeze=False)
    marked = False

    for j, ds in enumerate(datasets):
        ax = axes[0][j]
        by_side = sims_by_ds[ds]
        L = _reorder(by_side["L"])
        R = _reorder(by_side["R"]) if "R" in by_side else None
        lab = _labels(L)
        mean, n_sides = _avg_lr(L, R)
        # Only meaningful where the dataset HAS two hemispheres: there a k==1 cell is an
        # exception worth flagging.  When the whole connectome is one-sided the panel
        # label states it once, and marking every cell would add noise, not information.
        one_sided_cell = R is not None

        cmap = HEAT.copy(); cmap.set_bad(NA_FILL)
        im = ax.imshow(np.ma.masked_invalid(mean), cmap=cmap, vmin=0, vmax=1,
                       interpolation="nearest")
        ax.set_xticks(range(len(lab))); ax.set_yticks(range(len(lab)))
        ax.set_xticklabels(lab, fontsize=10, color=INK)
        ax.set_yticklabels(lab, fontsize=10, color=INK)
        for a in range(len(lab)):
            for b in range(len(lab)):
                v, k = mean[a, b], n_sides[a, b]
                if k == 0:
                    ax.text(b, a, "n/a", ha="center", va="center", fontsize=7.5,
                            style="italic", color=NA_INK)
                    continue
                star = "*" if (mark_single and one_sided_cell and k == 1) else ""
                marked |= bool(star)
                txt = (f"{v:.2f}" + star) if annotate else star
                if txt:
                    ax.text(b, a, txt, ha="center", va="center",
                            fontsize=8.5 if annotate else 13,
                            color=_ink_on(HEAT(float(v))))
        ax.set_title(DISPLAY.get(ds, ds), fontsize=12.5, fontweight="bold", pad=8)
        _cell_grid(ax, len(lab), len(lab))
        _bare(ax)
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.10)
        cb = fig.colorbar(im, cax=cax, ticks=[0, 0.25, 0.5, 0.75, 1.0])
        cb.set_label("cosine similarity", fontsize=9)
        cb.outline.set_visible(False); cb.ax.tick_params(length=2.5, labelsize=8.5)

    fig.subplots_adjust(wspace=0.42)
    if marked:
        fig.text(0.5, -0.02, "* one hemisphere only; the other is undefined "
                 "(no direct input)", ha="center", fontsize=8.5, color=MUTED,
                 style="italic")
    _save(fig, outdir, stem)


# ================================ 3. input map across datasets on one shared row rule
def fig_input_map_by_dataset(mats, hcs_by_ds, outdir, datasets,
                             stem="input_map_by_connectome", panel_width=3.0,
                             value_label="input fraction (shared scale)"):
    """Direct HCS input in each connectome, each analysed on its own.

    No cross-dataset cell correspondence is used.  `mancBodyid` would supply one, but
    it is undocumented and disagrees with MANC's own sensilla-field labels for about
    half the pairs where both datasets give a specific label -- see
    hcs_crossref_audit.csv.  Matching also discarded every unmatched cell.

    Comparability instead comes from the ordering RULE, which is fixed in advance and
    identical for every panel: side, then the motor neuron each sensillum most
    strongly contacts, then descending strength.  Because the rule is stated rather
    than fitted, the same bands land in the same sequence in either connectome.  Row
    counts differ, so bands align but individual rows do not -- that is the honest
    reading, and the claim it supports is population-level, not cell-level.

    Left and right sensilla are pooled into one population here.  The columns are each
    cell's own side, so hemisphere carries no information at this point; keeping it as
    a sort key would only draw every band twice.  A one-hemisphere connectome therefore
    needs no special handling: its cells are all on one side already.

    `datasets` is required rather than defaulted -- an earlier two-connectome default
    silently omitted a third dataset that had been loaded and computed.
    """
    n = len(datasets)
    fig_w = panel_width * n + 1.4
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 7.0), dpi=300, squeeze=False)
    # chrome has to track the panel width: at half width the default 10pt horizontal
    # tick labels are wider than the columns they sit under and would collide
    scale = min(1.0, panel_width / 3.0)
    tick_fs = max(6.5, 10.0 * scale ** 0.5)
    title_fs = max(9.0, 12.5 * scale ** 0.4)
    rot = 90 if panel_width / max(len(mats[datasets[0]].columns), 1) < 0.34 else 0

    panels = {}
    for ds in datasets:
        M = mats[ds]
        M = M.loc[M.abs().sum(axis=1) > 0]
        side = (hcs_by_ds[ds].drop_duplicates("bodyId").set_index("bodyId")
                             .side.reindex(M.index))
        panels[ds] = M.loc[_row_order_ipsi(M, side, by_side=False)]

    # ONE scale across all panels: same quantity, same units, and the figure exists
    # to be read across panels.
    allv = np.concatenate([panels[d].to_numpy().ravel() for d in datasets])
    vmax = np.percentile(allv[allv > 0], 99) if (allv > 0).any() else 1.0

    rank = {m: i for i, m in enumerate(MN_ORDER)}
    for j, ds in enumerate(datasets):
        ax = axes[0][j]
        M = panels[ds][sorted(panels[ds].columns, key=lambda t: rank.get(SHORT(t), 99))]
        im = ax.imshow(M.to_numpy(), aspect="auto", cmap=HEAT, vmin=0, vmax=vmax,
                       interpolation="nearest")
        ax.set_xticks(range(M.shape[1]))
        ax.set_xticklabels([SHORT(t) for t in M.columns], fontsize=tick_fs, color=INK,
                           rotation=rot, ha="center" if rot == 0 else "right")
        ax.set_yticks([])
        ax.set_title(f"{DISPLAY.get(ds, ds)}   (n = {len(M)})", fontsize=title_fs,
                     fontweight="bold", pad=8)
        if j == 0:
            ax.set_ylabel("haltere campaniform sensilla with direct input",
                          fontsize=max(7.0, 9.0 * scale ** 0.4))
        _bare(ax)
        for sp in ("top", "bottom", "left", "right"):
            ax.spines[sp].set_visible(True); ax.spines[sp].set_color(BASELINE)
            ax.spines[sp].set_linewidth(0.7)
        if j == n - 1:
            cax = make_axes_locatable(ax).append_axes("right", size="6%", pad=0.12)
            cb = fig.colorbar(im, cax=cax)
            cb.set_label(value_label, fontsize=max(7.0, 9.0 * scale ** 0.4))
            cb.outline.set_visible(False)
            cb.ax.tick_params(length=2.5, labelsize=max(6.5, 8.0 * scale ** 0.4))

    fig.subplots_adjust(wspace=0.18)
    # No suptitle and no ordering-rule caption: both belong in the manuscript text.  Two
    # things the caption MUST still state, since the figure no longer says them -- rows
    # are ordered independently per connectome by the motor neuron each sensillum most
    # strongly contacts then by strength, and the one shared colour scale spans
    # connectomes whose input denominators are not computed identically, so band
    # structure is comparable while absolute shade is not.
    _save(fig, outdir, stem)
