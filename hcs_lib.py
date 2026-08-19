"""
Haltere campaniform sensilla (HCS) -> wing steering motor neuron input similarity,
computed identically on two connectomes: male-cns:v1.0 and manc:v1.2.3.

The analysis uses DIRECT (monosynaptic) HCS -> motor neuron connections only.

Blind by construction.  This module knows the motor neurons only as an unordered
tuple of type strings.  It carries no notion of which are well studied and which are
not, and every parameter below was fixed before any result was inspected.  Nothing in
here branches on motor neuron identity.

Typical use:

    import hcs_lib as H
    b = H.load("male-cns")             # read the cached tables -> Bundle
    D = H.direct_matrix(b)             # HCS x MN, input fraction
    S = H.similarity(H.prep(H.side(D, "L")))    # 5 x 5 cosine similarity

Data lives in two separate roots so that inputs and outputs never mix:

    DATA_DIR   read-only cached query results, committed with the code
    OUT_DIR    figures and tables, regenerated on every run

Both can be overridden by environment variable (HCS_DATA / HCS_OUT); the driver also
takes --out.  neuPrint credentials are needed only to refresh the cache.
"""

import os
import numpy as np
import pandas as pd

# ============================================================================= config
# Fixed up front.  Do not tune these against a result; record any change in the notebook.

MN_TYPES = ("b1 MN", "b2 MN", "tp1 MN", "tp2 MN", "tpn MN")   # unordered, unlabelled
# An alternate set is passed explicitly to load(); it never changes the default run.
# Still unordered and unlabelled -- adding b3 does not alter any existing pairwise
# value, because prep() scales each motor neuron's column independently.
MN_TYPES_WITH_B3 = ("b1 MN", "b2 MN", "b3 MN", "tp1 MN", "tp2 MN", "tpn MN")
MIN_SYN = 3                      # minimum synapses for an edge to count
METRICS = ("cosine", "pearson", "jaccard")

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("HCS_DATA", os.path.join(_HERE, "data"))
OUT_DIR = os.environ.get("HCS_OUT", os.path.join(_HERE, "results"))
SERVER = "neuprint.janelia.org"

# HCS definition, identical in both datasets: everything on the haltere nerve whose type
# is in the SApp series.  339 cells in male-cns, 333 in MANC, against ~340 campaniform
# sensilla described anatomically on the haltere.
HCS_PRED = "{n}.entryNerve STARTS WITH 'DMetaN' AND {n}.type CONTAINS 'SApp'"

# --- dataset adapter --------------------------------------------------------------
# The two connectomes disagree on schema in ways that are silent rather than loud:
# MANC sides are 'LHS'/'RHS' where male-cns uses 'L'/'R', and MANC's neuron category
# lives in `class` where male-cns puts it in `superclass`.  Both are normalised here so
# that no analysis code below ever sees a dataset name.
#
# `sides` is read by the driver rather than a literal ("L", "R"), so that a dataset
# reconstructed in one hemisphere only can be added without special-casing it.
DATASETS = {
    "male-cns": dict(
        dataset="male-cns:v1.0",
        sides=("L", "R"),
        source="neuprint",
        # explicit whitelist rather than "not motor and not sensory": that phrasing
        # would also sweep in efferent, endocrine, glia and ENS bodies
        interneuron="{n}.superclass IN ['vnc_intrinsic','cb_intrinsic',"
                    "'ascending_neuron','descending_neuron']",
        nt="coalesce({n}.consensusNt, {n}.predictedNt, 'unclear')",
        status="",
    ),
    "manc": dict(
        dataset="manc:v1.2.3",
        sides=("L", "R"),
        source="neuprint",
        interneuron="{n}.class IN ['intrinsic neuron','ascending neuron',"
                    "'descending neuron']",
        nt="coalesce({n}.predictedNt, 'unclear')",
        # MANC carries 78,079 class=NULL bodies (unproofread fragments).  A
        # `NOT class IN [...]` filter would drop them through Cypher's three-valued
        # logic without ever deciding about them; the whitelist above plus this status
        # filter makes the exclusion explicit.
        status="AND {n}.status = 'Traced'",
    ),
    # FANC (Female Adult Nerve Cord), materialization v604.  Not a neuPrint dataset: it
    # is converted from local tables by tools/fanc_to_cache.py, which is why it carries
    # no Cypher fragments.  Only ONE hemisphere was reconstructed, so `sides` is what
    # keeps the driver from assuming a right-side counterpart exists; nothing else in the
    # analysis needs to know which dataset it is looking at.
    "fanc": dict(
        dataset="fanc:v604",
        sides=("L",),
        source="local",
    ),
}

# Side as a single letter, for both 'L'/'R' and 'LHS'/'RHS'.  Sensory neurons have no
# soma in the CNS and so carry rootSide rather than somaSide.
SIDE = "left(coalesce({n}.somaSide, {n}.rootSide), 1)"


def _q(template, **names):
    """Fill {n}-style node placeholders in a Cypher fragment."""
    return template.format(**names)


# ============================================================================== fetch
class Bundle:
    """Everything cached for one dataset, with matrices built lazily by the callers."""

    def __init__(self, name, hcs, mns, ins, direct, e1, e2):
        self.name = name
        self.dataset = DATASETS[name]["dataset"]
        self.hcs = hcs          # bodyId, type, side
        self.mns = mns          # bodyId, type, side, post
        self.ins = ins          # bodyId, type, side, post, nt, group
        self.direct = direct    # s, d, w          (HCS -> MN)
        self.e1 = e1            # s, i, w          (HCS -> IN)   see note in load()
        self.e2 = e2            # i, d, w          (IN  -> MN)

    def __repr__(self):
        return (f"<Bundle {self.name} ({self.dataset}): {len(self.hcs)} HCS, "
                f"{len(self.mns)} MN, {len(self.ins)} IN, "
                f"{len(self.direct)} direct edges, {len(self.e2)} IN->MN edges>")


def connect(name):
    """A neuPrint client.  Only reached when refreshing the cache."""
    from neuprint import Client          # imported lazily: a cached run needs no token
    token = os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
    if not token:
        raise RuntimeError(
            "NEUPRINT_APPLICATION_CREDENTIALS is not set.  Get a token from "
            "https://neuprint.janelia.org (Account menu) and export it:\n"
            "    export NEUPRINT_APPLICATION_CREDENTIALS='<token>'\n"
            "Only --refresh needs this; the committed cache runs without it.")
    return Client(SERVER, dataset=DATASETS[name]["dataset"], token=token)


def _fetch_all(name, mn_types=MN_TYPES):
    """Run every query for one dataset.  Called only on a cache miss."""
    from neuprint import fetch_custom
    spec = DATASETS[name]
    mn_list = list(mn_types)
    c = connect(name)
    hcs_s = _q(HCS_PRED, n="s")
    side_s, side_d, side_i = (_q(SIDE, n=x) for x in "sdi")
    in_pred = _q(spec["interneuron"], n="i") + " " + _q(spec["status"], n="i")

    hcs = fetch_custom(f"""
        MATCH (s:Neuron) WHERE {hcs_s}
        RETURN s.bodyId AS bodyId, s.type AS type, {side_s} AS side
        """, client=c, format="pandas")

    mns = fetch_custom(f"""
        MATCH (d:Neuron) WHERE d.type IN {mn_list}
        RETURN d.bodyId AS bodyId, d.type AS type, {side_d} AS side, d.post AS post
        """, client=c, format="pandas")

    direct = fetch_custom(f"""
        MATCH (s:Neuron)-[r:ConnectsTo]->(d:Neuron)
        WHERE {hcs_s} AND d.type IN {mn_list} AND r.weight >= {MIN_SYN}
        RETURN s.bodyId AS s, d.bodyId AS d, r.weight AS w
        """, client=c, format="pandas")

    # One path query, then split into the two edge tables.  This supplies `ins` and
    # `e2`, which the left/right positive control needs.  Restricting HCS->IN to
    # interneurons that also reach a target motor neuron keeps the pull small.
    paths = fetch_custom(f"""
        MATCH (s:Neuron)-[r1:ConnectsTo]->(i:Neuron)-[r2:ConnectsTo]->(d:Neuron)
        WHERE {hcs_s} AND d.type IN {mn_list}
          AND r1.weight >= {MIN_SYN} AND r2.weight >= {MIN_SYN}
          AND {in_pred}
        RETURN s.bodyId AS s, i.bodyId AS i, d.bodyId AS d,
               r1.weight AS w1, r2.weight AS w2,
               i.type AS i_type, {side_i} AS i_side, i.post AS i_post,
               {_q(spec['nt'], n='i')} AS i_nt, i.group AS i_group
        """, client=c, format="pandas")

    ins = (paths[["i", "i_type", "i_side", "i_post", "i_nt", "i_group"]]
           .drop_duplicates("i")
           .rename(columns={"i": "bodyId", "i_type": "type", "i_side": "side",
                            "i_post": "post", "i_nt": "nt", "i_group": "group"})
           .reset_index(drop=True))
    # interneurons without a bilateral `group` fall back to their own bodyId, so they
    # simply never match across hemispheres rather than silently merging into NaN
    ins["group"] = ins["group"].fillna(ins.bodyId).astype("int64")
    e1 = paths[["s", "i", "w1"]].drop_duplicates().rename(columns={"w1": "w"})
    e2 = paths[["i", "d", "w2"]].drop_duplicates().rename(columns={"w2": "w"})

    return Bundle(name, hcs, mns, ins, direct,
                  e1.reset_index(drop=True), e2.reset_index(drop=True))


def load(name, refresh=False, mn_types=MN_TYPES, tag=None):
    """Read one dataset's cached tables, fetching them first if they are absent.

    `mn_types` + `tag` allow a second motor-neuron set alongside the default one:
    the tag names a separate cache slot, so the two never overwrite each other and
    the default call keeps reading the cache it already built.

    `e1` (HCS -> interneuron) is cached but no longer read by anything: it existed for
    the two-hop analysis, which this pipeline no longer computes.  It is kept so the
    cache schema is stable and so two-hop could be restored without a re-fetch.
    """
    d = os.path.join(DATA_DIR, name, "cache" if tag is None else f"cache_{tag}")
    parts = ("hcs", "mns", "ins", "direct", "e1", "e2")
    paths = {p: os.path.join(d, f"{p}.csv") for p in parts}

    if not refresh and all(os.path.exists(p) for p in paths.values()):
        t = {p: pd.read_csv(paths[p]) for p in parts}
        b = Bundle(name, t["hcs"], t["mns"], t["ins"], t["direct"], t["e1"], t["e2"])
        print(f"  loaded {name} from cache: {b}")
        return b

    if DATASETS[name].get("source") == "local":
        raise RuntimeError(
            f"no cached tables for {name} at {d}.  This dataset is converted from local "
            f"files rather than queried, so build the cache first:\n"
            f"    python tools/fanc_to_cache.py")

    print(f"  fetching {name} ({DATASETS[name]['dataset']})"
          f"{'' if tag is None else ' [' + tag + ']'} ...")
    b = _fetch_all(name, mn_types)
    os.makedirs(d, exist_ok=True)
    for p in parts:
        getattr(b, p).to_csv(paths[p], index=False)
    print(f"  {b}")
    return b


def hcs_to_mn_edges(name, mn_types, refresh=False):
    """HCS -> motor neuron edges at weight >= 1, cached like every other table.

    Unthresholded on purpose: the caller reports both the >= 1 and the >= MIN_SYN
    grouping, because whether the handful of weak crossovers count is the reader's
    call rather than something to bake in.  Cached so that a re-run needs no token.
    """
    path = os.path.join(DATA_DIR, name, "hcs_to_mn_edges.csv")
    if not refresh and os.path.exists(path):
        return pd.read_csv(path)
    from neuprint import fetch_custom          # imported only on the query path
    c = connect(name)
    hcs = _q(HCS_PRED, n="s")
    ss, sd = _q(SIDE, n="s"), _q(SIDE, n="d")
    e = fetch_custom(f"""
        MATCH (s:Neuron)-[r:ConnectsTo]->(d:Neuron)
        WHERE {hcs} AND d.type IN {list(mn_types)}
        RETURN s.bodyId AS bodyId, s.type AS type, {ss} AS side,
               d.type AS mn, {sd} AS mn_side, r.weight AS w
        """, client=c, format="pandas")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    e.to_csv(path, index=False)
    return e


# =========================================================================== matrices
# Every matrix is (features x motor neurons).  Motor neuron columns carry a
# MultiIndex (type, side) so that left and right stay distinguishable throughout.

def _mn_cols(b):
    m = b.mns.sort_values(["type", "side"])
    return m, pd.MultiIndex.from_arrays([m.type, m.side], names=["type", "side"])


def _pivot(edges, index, mn, col_of_d, value):
    """Sparse edge list -> dense (features x MN) frame on a fixed feature index."""
    M = pd.DataFrame(0.0, index=index, columns=col_of_d)
    if len(edges):
        p = edges.pivot_table(index=edges.columns[0], columns="d",
                              values=value, aggfunc="sum", fill_value=0.0)
        lookup = dict(zip(mn.bodyId, zip(mn.type, mn.side)))
        for d in p.columns:
            M.loc[p.index, lookup[d]] += p[d].values
    return M


def direct_matrix(b, normalise=True):
    """HCS x MN, each edge as a fraction of the target's total input.

    Normalising by the target's `post` is what stops the comparison from simply
    ranking motor neurons by size (b2 has ~15.6k input sites against tp1's ~9k).
    Every similarity in this project is computed on the normalised form.

    `normalise=False` returns raw synapse counts instead.  That view is worth having --
    it is the measured quantity, and a fraction hides whether a strong-looking
    connection rests on 3 synapses or 40 -- but it reintroduces exactly the size
    confound above, so it is reported alongside the fractions, never instead of them.
    """
    mn, cols = _mn_cols(b)
    e = b.direct.merge(mn[["bodyId", "post"]].rename(columns={"bodyId": "d"}), on="d")
    e["v"] = e.w / e.post if normalise else e.w.astype(float)
    return _pivot(e[["s", "d", "v"]], b.hcs.bodyId.values, mn, cols, "v")


def premotor_matrix(b):
    """Interneuron x MN, as a fraction of the motor neuron's input.

    No figure uses this.  It exists for the left/right positive control below, which
    needs a feature space that both hemispheres live in; see collapse_side_relative.
    """
    mn, cols = _mn_cols(b)
    e = b.e2.merge(mn[["bodyId", "post"]].rename(columns={"bodyId": "d"}), on="d")
    e["v"] = e.w / e.post
    e = e.rename(columns={"i": "s"})
    return _pivot(e[["s", "d", "v"]], b.ins.bodyId.values, mn, cols, "v")


def collapse_side_relative(M, meta, key):
    """Re-express features relative to each motor neuron's own side.

    Individual HCS and interneurons cannot be matched between hemispheres, so a
    left and a right motor neuron never share a cell-level feature.  Grouping by a
    bilateral key (interneuron `group`) and splitting into ipsi/contra gives a feature
    space both sides live in, which is what makes the left/right positive control
    possible at all.  Used only by that control.
    """
    meta = meta.drop_duplicates("bodyId").set_index("bodyId")
    side = meta.side.reindex(M.index).values
    k = meta[key].astype(str).reindex(M.index).values
    out = {}
    for col in M.columns:
        rel = np.where(side == col[1], "ipsi", "contra")
        idx = pd.MultiIndex.from_arrays([rel, k], names=["rel", key])
        out[col] = M[col].groupby(idx).sum()
    R = pd.DataFrame(out).fillna(0.0)
    R.columns = pd.MultiIndex.from_tuples(R.columns, names=["type", "side"])
    return R.sort_index()


# ========================================================================= similarity
def prep(M):
    """sqrt-compress, then scale each motor neuron's vector to unit length.

    The sqrt keeps one dominant partner from deciding the whole score -- these
    weight distributions are heavy-tailed.  Scaling is per COLUMN, which is why
    adding a sixth motor neuron cannot move any existing pairwise value.
    """
    X = M.to_numpy(float)
    X = np.sign(X) * np.sqrt(np.abs(X))
    n = np.linalg.norm(X, axis=0)
    n[n == 0] = 1.0
    return pd.DataFrame(X / n, index=M.index, columns=M.columns)


def similarity(M, metric="cosine"):
    """MN x MN similarity.  Expects a prepped matrix; does not transform again."""
    X = M.to_numpy(float).T                      # motor neurons as rows
    # A motor neuron with no input at all has a zero-length vector, and its
    # similarity to anything is UNDEFINED rather than zero.  tp2 hits this in every
    # dataset, and reporting 0.00 there would read as "shares nothing with anyone"
    # when the honest statement is "has no direct input to compare".
    empty = ~np.any(X != 0, axis=1)
    if metric == "cosine":
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n[n == 0] = 1.0
        S = (X / n) @ (X / n).T
    elif metric == "pearson":
        Z = X - X.mean(axis=1, keepdims=True)
        n = np.linalg.norm(Z, axis=1, keepdims=True)
        n[n == 0] = 1.0
        S = (Z / n) @ (Z / n).T
    elif metric == "jaccard":
        B = X > 0
        inter = (B[:, None, :] & B[None, :, :]).sum(axis=2)
        union = (B[:, None, :] | B[None, :, :]).sum(axis=2)
        S = np.divide(inter, union, out=np.zeros_like(inter, float), where=union > 0)
    else:
        raise ValueError(f"unknown metric {metric!r}")
    S = np.asarray(S, dtype=float)
    S[empty, :] = np.nan
    S[:, empty] = np.nan
    lab = [t for t, _ in M.columns] if isinstance(M.columns, pd.MultiIndex) else M.columns
    if isinstance(M.columns, pd.MultiIndex):
        lab = [f"{t.replace(' MN','')}_{s}" for t, s in M.columns]
    return pd.DataFrame(S, index=lab, columns=lab)


def side(M, s):
    """Columns for one hemisphere, relabelled to bare motor neuron names."""
    sub = M.loc[:, [c for c in M.columns if c[1] == s]]
    sub.columns = pd.MultiIndex.from_tuples(list(sub.columns), names=["type", "side"])
    return sub


def offdiag(S):
    """The unique off-diagonal pairs of a similarity matrix, as a tidy frame."""
    iu = np.triu_indices(len(S), k=1)
    return pd.DataFrame({"a": np.array(S.index)[iu[0]],
                         "b": np.array(S.columns)[iu[1]],
                         "sim": S.to_numpy()[iu]})


# ============================================== ipsilateral view + cross-dataset rows
def ipsi_matrix(b, normalise=True):
    """HCS cells x motor neuron TYPES, each cell scored against its own side.

    In the (type, side) matrices half of every row is structurally zero: a left
    sensillum contacts left motor neurons and essentially nothing else.  Collapsing
    onto the cell's own side removes that guaranteed blank half and lets all cells
    share one feature space.  ~2% of input is contralateral and is dropped.

    `normalise=False` carries raw synapse counts through instead of input fractions;
    see direct_matrix for why both views are kept.
    """
    D = direct_matrix(b, normalise=normalise)
    side = b.hcs.drop_duplicates("bodyId").set_index("bodyId").side.reindex(D.index)
    types = sorted({t for t, _ in D.columns})
    out = pd.DataFrame(0.0, index=D.index, columns=types)
    for t in types:
        for s in ("L", "R"):
            if (t, s) in D.columns:
                m = (side == s).to_numpy()
                out.loc[m, t] = D.loc[m, (t, s)].to_numpy()
    out.columns.name = "type"
    return out


def hcs_crossref(refresh=False):
    """male-cns HCS bodyId -> MANC HCS bodyId, via the `mancBodyid` field.

    The two connectomes are different animals, so this is a morphological
    correspondence table Janelia provides, not an identity.  It covers 310 of the
    339 male-cns HCS; the remainder simply cannot be placed side by side.
    """
    path = os.path.join(DATA_DIR, "hcs_crossref.csv")
    if not refresh and os.path.exists(path):
        return pd.read_csv(path)
    from neuprint import fetch_custom          # imported only on the query path
    c = connect("male-cns")
    x = fetch_custom(f"""
        MATCH (s:Neuron) WHERE {_q(HCS_PRED, n='s')} AND s.mancBodyid IS NOT NULL
        RETURN s.bodyId AS mcns, s.mancBodyid AS manc, s.type AS type
        """, client=c, format="pandas")
    # keep it a genuine 1:1 pairing -- a handful of many-to-one entries would
    # otherwise let one MANC cell stand in for two male-cns rows
    x = x.drop_duplicates("manc", keep=False).drop_duplicates("mcns", keep=False)
    os.makedirs(DATA_DIR, exist_ok=True)
    x.to_csv(path, index=False)
    return x


def crossref_audit(bundles, refresh=False):
    """Does the `mancBodyid` correspondence agree with the datasets' own annotations?

    This is the evidence for NOT using it.  `mancBodyid` is undocumented -- male-cns
    declares it only as a type in Meta.neuronProperties, and neither the companion
    site nor the natverse packages describe how it was derived -- so the only check
    available is internal consistency against each dataset's own labels.

    Side is the coarse check; SApp field assignment is the fine one.  A compound
    label like 'SApp09,SApp22' is a SET of candidate fields, so it is parsed to
    integers rather than string-compared: overlapping sets are compatible, not
    contradictory.  Returns (per-pair frame, summary dict).
    """
    import re
    x = hcs_crossref(refresh=refresh)
    meta = {n: b.hcs.drop_duplicates("bodyId").set_index("bodyId")
            for n, b in bundles.items()}
    d = x[x.mcns.isin(meta["male-cns"].index) & x.manc.isin(meta["manc"].index)].copy()
    d["type_mcns"] = d.mcns.map(meta["male-cns"].type)
    d["type_manc"] = d.manc.map(meta["manc"].type)
    d["side_mcns"] = d.mcns.map(meta["male-cns"].side)
    d["side_manc"] = d.manc.map(meta["manc"].side)
    d["side_agrees"] = d.side_mcns == d.side_manc

    ids = lambda t: set(re.findall(r"SApp(\d+)", str(t)))
    def verdict(r):
        a, b = ids(r.type_mcns), ids(r.type_manc)
        if not a or not b:
            return "uncheckable"
        return "identical" if a == b else ("compatible" if a & b else "contradictory")
    d["verdict"] = d.apply(verdict, axis=1)

    counts = d.verdict.value_counts().to_dict()
    checkable = len(d) - counts.get("uncheckable", 0)
    summary = {"pairs": len(d), "side_agree": int(d.side_agrees.sum()),
               "checkable": checkable, **counts}
    return d, summary
