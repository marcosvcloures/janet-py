"""
Experiment: Path dependence of Hebbian learning.

Three variants tested:
  1. Order permutation: same queries, different order, many reps
  2. Biased distribution: same queries but some repeated more than others
  3. Cross-domain on network: queries that cross node boundaries

Question: does the system's history affect its final state?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from janet import Sorter, DIMS, DEGREES, dot, normalize
from network import Network, Node


def make_corpus():
    """Shared corpus."""
    return [
        "entropy increases in isolated thermodynamic systems over time",
        "information erasure requires minimum energy cost per bit erased",
        "Shannon entropy measures uncertainty in probability distributions",
        "thermodynamic entropy and information entropy share mathematical structure",
        "reversible computation avoids energy dissipation by preserving state",
        "Maxwell demon sorts molecules but must erase memory paying energy cost",
        "Landauer principle connects logical irreversibility to physical heat",
        "Boltzmann constant links microscopic states to macroscopic temperature",
        "entropy is the bridge between thermodynamics and information theory",
        "computation requires energy only when information is destroyed irreversibly",
    ]


def make_queries():
    """Standard query set."""
    return [
        "entropy thermodynamic systems time",
        "erasure energy cost bit Landauer",
        "Shannon uncertainty probability information",
        "demon molecules memory sorting energy",
        "computation reversible dissipation state",
        "Boltzmann microscopic macroscopic temperature",
        "entropy bridge thermodynamics information",
        "irreversibility heat logical physical",
    ]


def wave_divergence(s1: Sorter, s2: Sorter) -> tuple[float, float]:
    """Mean and max cosine distance between shared wave vectors."""
    shared = set(s1.waves.keys()) & set(s2.waves.keys())
    if not shared:
        return 0.0, 0.0
    diffs = []
    for tok in shared:
        w1 = s1.waves[tok].astype(np.int64)
        w2 = s2.waves[tok].astype(np.int64)
        d12 = int(np.dot(w1, w2))
        d11 = int(np.dot(w1, w1))
        d22 = int(np.dot(w2, w2))
        if d11 > 0 and d22 > 0:
            cos = d12 / (d11**0.5 * d22**0.5)
            diffs.append(max(0.0, 1.0 - cos))
    return (np.mean(diffs), np.max(diffs)) if diffs else (0.0, 0.0)


def retrieval_divergence(s1: Sorter, s2: Sorter, queries: list[str]) -> int:
    """Count queries that get different answers."""
    diff = 0
    for q in queries:
        a1, _ = s1.retrieve(s1.encode(q))
        a2, _ = s2.retrieve(s2.encode(q))
        if a1 != a2:
            diff += 1
    return diff


def run_queries(corpus, queries, n_reps) -> Sorter:
    """Build corpus and run queries n_reps times."""
    s = Sorter()
    s.learn_batch(corpus)
    for _ in range(n_reps):
        for q in queries:
            s.retrieve(s.encode(q))
    return s


# ── Variant 1: Order permutation ──────────────────────────────────────────

def experiment_order():
    """Same queries in different orders, many repetitions."""
    print("\n  VARIANT 1: Query order permutation")
    print("  " + "-" * 50)

    corpus = make_corpus()
    queries = make_queries()
    n_reps = 100  # more reps to accumulate effect

    rng = np.random.default_rng(42)
    orders = [list(queries)]
    for _ in range(5):
        p = list(queries)
        rng.shuffle(p)
        orders.append(p)

    sorters = [run_queries(corpus, order, n_reps) for order in orders]
    phis = [s.phi() for s in sorters]

    # Measure divergence
    mean_divs, max_divs = [], []
    for i in range(1, len(sorters)):
        m, mx = wave_divergence(sorters[0], sorters[i])
        mean_divs.append(m)
        max_divs.append(mx)

    test_q = ["entropy systems", "energy cost", "Shannon information",
              "demon memory", "computation state", "Boltzmann temperature",
              "bridge thermodynamics", "heat irreversibility", "entropy information", "Landauer principle"]
    ret_diffs = [retrieval_divergence(sorters[0], sorters[i], test_q) for i in range(1, len(sorters))]

    print(f"  {n_reps*len(queries)} retrievals per trial, 6 orderings")
    print(f"  Phi range: {max(phis)-min(phis):.4f} (all={phis[0]:.4f})")
    print(f"  Wave divergence: mean={np.mean(mean_divs):.8f}, max={np.max(max_divs):.8f}")
    print(f"  Retrieval diffs: {sum(ret_diffs)}/{len(ret_diffs)*len(test_q)}")

    return np.max(max_divs), sum(ret_diffs)


# ── Variant 2: Biased query distribution ──────────────────────────────────

def experiment_biased():
    """Same queries but with different frequency distributions."""
    print("\n  VARIANT 2: Biased query distribution")
    print("  " + "-" * 50)

    corpus = make_corpus()
    queries = make_queries()

    # Distribution A: heavy on thermodynamics queries
    dist_a = queries[:3] * 20 + queries[3:] * 5  # 60 thermo + 25 other = 85
    # Distribution B: heavy on computation queries
    dist_b = queries[:3] * 5 + queries[3:] * 20  # 15 thermo + 100 other = 115
    # Distribution C: uniform
    dist_c = queries * 12  # 96

    sorters = []
    for dist in [dist_a, dist_b, dist_c]:
        s = Sorter()
        s.learn_batch(corpus)
        for q in dist:
            s.retrieve(s.encode(q))
        sorters.append(s)

    phis = [s.phi() for s in sorters]
    labels = ["thermo-heavy", "compute-heavy", "uniform"]

    print(f"  Distributions: A={len(dist_a)} queries, B={len(dist_b)}, C={len(dist_c)}")
    for i, (label, phi) in enumerate(zip(labels, phis)):
        print(f"    {label}: phi={phi:.4f}")

    # Cross-divergence
    pairs = [(0,1), (0,2), (1,2)]
    pair_labels = ["A vs B", "A vs C", "B vs C"]
    print(f"  Wave divergence:")
    max_div = 0
    for (i,j), pl in zip(pairs, pair_labels):
        m, mx = wave_divergence(sorters[i], sorters[j])
        max_div = max(max_div, mx)
        print(f"    {pl}: mean={m:.8f}, max={mx:.8f}")

    test_q = ["entropy systems", "energy cost", "Shannon information",
              "demon memory", "computation state", "Boltzmann temperature",
              "bridge thermodynamics", "heat irreversibility", "entropy information", "Landauer principle"]
    ret_diffs = 0
    for i, j in pairs:
        ret_diffs += retrieval_divergence(sorters[i], sorters[j], test_q)

    print(f"  Retrieval diffs: {ret_diffs}/{len(pairs)*len(test_q)}")

    return max_div, ret_diffs


# ── Variant 3: Cross-domain queries on network ────────────────────────────

def experiment_network():
    """Two-node network. Compare: only local queries vs cross-domain queries."""
    print("\n  VARIANT 3: Cross-domain queries on network")
    print("  " + "-" * 50)

    physics = [
        "entropy increases in isolated thermodynamic systems",
        "Boltzmann constant links microscopic to macroscopic",
        "Maxwell demon sorts molecules paying memory cost",
        "Landauer principle connects irreversibility to heat",
    ]
    cs = [
        "algorithms have time complexity measured in big O",
        "hash tables provide constant time average lookup",
        "binary search requires sorted input for logarithmic time",
        "recursion solves problems by reducing to smaller instances",
    ]

    # Trial A: only local queries (no cross-domain)
    net_a = Network()
    s1a = Sorter()
    s1a.learn_batch(physics)
    s2a = Sorter()
    s2a.learn_batch(cs)
    n1a = Node(name="physics", sorter=s1a)
    n2a = Node(name="cs", sorter=s2a)
    n1a.add_peer(n2a)
    net_a.add_node(n1a)
    net_a.add_node(n2a)

    # Local queries only
    for _ in range(10):
        net_a.query("entropy thermodynamic systems", entry_node="physics")
        net_a.query("algorithms complexity time", entry_node="cs")

    # Trial B: cross-domain queries (force routing between nodes)
    net_b = Network()
    s1b = Sorter()
    s1b.learn_batch(physics)
    s2b = Sorter()
    s2b.learn_batch(cs)
    n1b = Node(name="physics", sorter=s1b)
    n2b = Node(name="cs", sorter=s2b)
    n1b.add_peer(n2b)
    net_b.add_node(n1b)
    net_b.add_node(n2b)

    # Cross-domain queries (CS questions entering physics node)
    for _ in range(10):
        net_b.query("algorithms complexity time", entry_node="physics")
        net_b.query("entropy thermodynamic systems", entry_node="cs")

    phi_a = net_a.phi_network()
    phi_b = net_b.phi_network()

    print(f"  Local-only queries:    phi_network={phi_a['phi_network']:.4f}, emergence={phi_a['emergence']:.4f}")
    print(f"  Cross-domain queries:  phi_network={phi_b['phi_network']:.4f}, emergence={phi_b['emergence']:.4f}")

    # Wave divergence between the physics nodes of each trial
    m, mx = wave_divergence(s1a, s1b)
    print(f"  Physics node divergence (local vs cross): mean={m:.6f}, max={mx:.6f}")

    m2, mx2 = wave_divergence(s2a, s2b)
    print(f"  CS node divergence (local vs cross):      mean={m2:.6f}, max={mx2:.6f}")

    diff = phi_b['phi_network'] - phi_a['phi_network']
    print(f"  Phi difference: {diff:+.4f}")

    return diff, mx, mx2


# ── Main ──────────────────────────────────────────────────────────────────

def experiment():
    print("=" * 70)
    print("EXPERIMENT: Path Dependence of Hebbian Learning")
    print("=" * 70)
    print()
    print("Three variants. Question: does query history affect system state?")

    max_div_1, ret_1 = experiment_order()
    max_div_2, ret_2 = experiment_biased()
    phi_diff, div_phys, div_cs = experiment_network()

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"  {'Variant':<30} {'Wave divergence':<18} {'Retrieval diff':<16} {'Verdict'}")
    print(f"  {'-'*30} {'-'*18} {'-'*16} {'-'*20}")

    v1 = "structural" if max_div_1 > 1e-6 else "none"
    v1r = "YES" if ret_1 > 0 else "no"
    print(f"  {'Order permutation':<30} {max_div_1:<18.8f} {v1r:<16} {v1}")

    v2 = "structural" if max_div_2 > 1e-6 else "none"
    v2r = "YES" if ret_2 > 0 else "no"
    print(f"  {'Biased distribution':<30} {max_div_2:<18.8f} {v2r:<16} {v2}")

    v3 = f"phi +{phi_diff:.3f}" if phi_diff > 0.01 else "minimal"
    print(f"  {'Cross-domain (network)':<30} {max(div_phys,div_cs):<18.6f} {v3:<16} {'integration gain' if phi_diff > 0.01 else 'no effect'}")

    print()
    if max_div_1 > 1e-6 or max_div_2 > 1e-6:
        print("  CONCLUSION: Path dependence exists at the structural level.")
        if ret_1 > 0 or ret_2 > 0:
            print("  Query history affects BOTH internal geometry AND external behavior.")
        else:
            print("  Query history affects internal geometry but NOT retrieval behavior.")
            print("  The system is functionally stable despite structural memory of history.")
    if phi_diff > 0.01:
        print(f"  Cross-domain queries increase network integration by {phi_diff:.3f}.")
        print("  The TYPE of experience (local vs cross-domain) matters more than order.")


if __name__ == "__main__":
    experiment()
