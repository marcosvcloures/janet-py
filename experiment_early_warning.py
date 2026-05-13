"""
Experiment: Phi as early warning of degradation.

Three degradation modes tested:
  1. Corruption: replace tokens in entries with noise
  2. Removal: delete entries one by one (bridge entries first)
  3. Dilution: add unrelated entries (noise flood)

Question: in all modes, does phi drop BEFORE accuracy drops?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from janet import Sorter, DIMS, DEGREES, normalize, particle


def make_corpus():
    """Coherent corpus with bridges between subtopics."""
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


def make_test_queries():
    """Queries with expected entry indices."""
    return [
        ("entropy increases isolated thermodynamic", 0),
        ("erasure energy cost bit", 1),
        ("Shannon uncertainty probability", 2),
        ("thermodynamic information entropy mathematical", 3),
        ("reversible computation dissipation", 4),
        ("demon sorts molecules memory", 5),
        ("Landauer irreversibility heat", 6),
        ("Boltzmann microscopic macroscopic", 7),
    ]


def measure_accuracy(sorter, queries, corpus):
    """Fraction of queries retrieving the correct entry."""
    correct = 0
    for query_text, expected_idx in queries:
        if expected_idx >= len(sorter.entries):
            continue
        answer, _, _ = sorter.orbit_with_cost(query_text)
        if answer == corpus[expected_idx]:
            correct += 1
    return correct / len(queries)


def find_first_drop(values, baseline, threshold=0.20):
    """Find first index where value drops > threshold from baseline."""
    for i, v in enumerate(values):
        if baseline > 0 and (baseline - v) / baseline > threshold:
            return i + 1
    return None


# ── Variant 1: Corruption ─────────────────────────────────────────────────

def experiment_corruption():
    """Corrupt entries progressively: replace N tokens with noise per entry."""
    print("\n  VARIANT 1: Token corruption")
    print("  " + "-" * 50)

    corpus = make_corpus()
    queries = make_test_queries()
    rng = np.random.default_rng(42)
    noise_words = ["xyzzy", "plugh", "qwert", "zxcvb", "mnbvc", "lkjhg", "poiuy", "asdfg"]

    s = Sorter()
    s.learn_batch(corpus)
    phi_0 = s.phi()
    acc_0 = measure_accuracy(s, queries, corpus)

    print(f"  Baseline: phi={phi_0:.4f}, accuracy={acc_0:.2f}")

    phis, accs = [], []
    for level in range(1, 8):
        corrupted = []
        for text in corpus:
            tokens = text.split()
            n_corrupt = min(level, len(tokens))
            indices = rng.choice(len(tokens), size=n_corrupt, replace=False)
            for idx in indices:
                tokens[idx] = noise_words[idx % len(noise_words)]
            corrupted.append(" ".join(tokens))

        s2 = Sorter()
        s2.learn_batch(corrupted)
        phis.append(s2.phi())
        accs.append(measure_accuracy(s2, queries, corrupted))

    phi_drop = find_first_drop(phis, phi_0)
    acc_drop = find_first_drop(accs, acc_0)

    print(f"  Phi first drop >20%: level {phi_drop or 'never'}")
    print(f"  Acc first drop >20%: level {acc_drop or 'never'}")
    if phi_drop and acc_drop:
        print(f"  Lead: {acc_drop - phi_drop} level(s)")
    return phi_drop, acc_drop, phis, accs


# ── Variant 2: Entry removal ──────────────────────────────────────────────

def experiment_removal():
    """Remove entries one by one, starting with bridge entries."""
    print("\n  VARIANT 2: Entry removal (bridges first)")
    print("  " + "-" * 50)

    corpus = make_corpus()
    queries = make_test_queries()

    s = Sorter()
    s.learn_batch(corpus)
    phi_0 = s.phi()
    acc_0 = measure_accuracy(s, queries, corpus)

    print(f"  Baseline: phi={phi_0:.4f}, accuracy={acc_0:.2f}")

    # Remove bridge entries first (indices 3, 8, 9 — the ones connecting subtopics)
    removal_order = [8, 9, 3, 7, 6, 5, 4]

    phis, accs = [], []
    remaining = list(range(len(corpus)))

    for step, remove_idx in enumerate(removal_order):
        remaining = [i for i in remaining if i != remove_idx]
        subset = [corpus[i] for i in remaining]

        s2 = Sorter()
        s2.learn_batch(subset)
        phis.append(s2.phi())

        # Accuracy: only count queries whose target is still present
        correct = 0
        total = 0
        for query_text, expected_idx in queries:
            if expected_idx in remaining:
                total += 1
                answer, _, _ = s2.orbit_with_cost(query_text)
                if answer == corpus[expected_idx]:
                    correct += 1
        accs.append(correct / total if total > 0 else 0)

    phi_drop = find_first_drop(phis, phi_0)
    acc_drop = find_first_drop(accs, acc_0)

    print(f"  Phi first drop >20%: step {phi_drop or 'never'}")
    print(f"  Acc first drop >20%: step {acc_drop or 'never'}")
    if phi_drop and acc_drop:
        print(f"  Lead: {acc_drop - phi_drop} step(s)")
    return phi_drop, acc_drop, phis, accs


# ── Variant 3: Dilution ───────────────────────────────────────────────────

def experiment_dilution():
    """Flood corpus with unrelated entries (dilution attack)."""
    print("\n  VARIANT 3: Dilution (noise flood)")
    print("  " + "-" * 50)

    corpus = make_corpus()
    queries = make_test_queries()

    noise_entries = [
        "purple elephants dance on crystalline moonbeams forever",
        "quantum basketball tournaments held underwater weekly",
        "recursive sandwiches contain infinite layers of cheese",
        "temporal origami folds yesterday into tomorrow shapes",
        "magnetic poetry arranges itself into grocery lists",
        "invisible architects design buildings made of sound",
        "parallel parking requires knowledge of ancient geometry",
        "synthetic rainbows taste like forgotten telephone numbers",
        "gravitational poetry slams attract dense audiences nightly",
        "philosophical plumbing connects abstract pipes to drains",
        "algebraic cooking transforms raw equations into theorems",
        "digital archaeology excavates ancient cache memories carefully",
        "hypothetical gardening grows imaginary flowers in virtual soil",
        "acoustic painting uses sound waves to create visual art",
        "mathematical fishing catches prime numbers in infinite streams",
    ]

    s = Sorter()
    s.learn_batch(corpus)
    phi_0 = s.phi()
    acc_0 = measure_accuracy(s, queries, corpus)

    print(f"  Baseline: phi={phi_0:.4f}, accuracy={acc_0:.2f} ({len(corpus)} entries)")

    phis, accs = [], []
    for n_noise in range(3, 16, 3):  # add 3, 6, 9, 12, 15 noise entries
        diluted = corpus + noise_entries[:n_noise]
        s2 = Sorter()
        s2.learn_batch(diluted)
        phis.append(s2.phi())
        accs.append(measure_accuracy(s2, queries, corpus))

    phi_drop = find_first_drop(phis, phi_0)
    acc_drop = find_first_drop(accs, acc_0)

    print(f"  Phi first drop >20%: step {phi_drop or 'never'} (={phi_drop*3 if phi_drop else 'n/a'} noise entries)")
    print(f"  Acc first drop >20%: step {acc_drop or 'never'} (={acc_drop*3 if acc_drop else 'n/a'} noise entries)")
    if phi_drop and acc_drop:
        print(f"  Lead: {acc_drop - phi_drop} step(s)")
    return phi_drop, acc_drop, phis, accs


# ── Main ──────────────────────────────────────────────────────────────────

def experiment():
    print("=" * 70)
    print("EXPERIMENT: Phi as Early Warning of Degradation")
    print("=" * 70)
    print()
    print("Three degradation modes. Question: does phi drop before accuracy?")

    r1 = experiment_corruption()
    r2 = experiment_removal()
    r3 = experiment_dilution()

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"  {'Mode':<25} {'Phi drops at':<15} {'Acc drops at':<15} {'Lead'}")
    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*10}")

    results = [
        ("Corruption", r1[0], r1[1]),
        ("Removal (bridges)", r2[0], r2[1]),
        ("Dilution (noise)", r3[0], r3[1]),
    ]

    leads = []
    for name, phi_d, acc_d in results:
        phi_s = str(phi_d) if phi_d else "never"
        acc_s = str(acc_d) if acc_d else "never"
        if phi_d and acc_d:
            lead = acc_d - phi_d
            leads.append(lead)
            print(f"  {name:<25} {phi_s:<15} {acc_s:<15} {lead:+d} step(s)")
        else:
            print(f"  {name:<25} {phi_s:<15} {acc_s:<15} {'n/a'}")

    print()
    if leads and all(l > 0 for l in leads):
        print(f"  CONCLUSION: Phi is an early warning in ALL degradation modes.")
        print(f"  Average lead: {sum(leads)/len(leads):.1f} steps.")
    elif leads and any(l > 0 for l in leads):
        print(f"  CONCLUSION: Phi is an early warning in SOME degradation modes.")
    else:
        print(f"  CONCLUSION: Early warning not consistently confirmed.")


if __name__ == "__main__":
    experiment()
