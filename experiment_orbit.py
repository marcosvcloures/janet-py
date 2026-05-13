"""
Experiment: The orbit as primary output.

The orbit — not the destination — is the answer.

Demonstrates that the full trajectory (path, cost, amplitude, convergence)
carries information that the final entry alone does not:

1. Cost discriminates confidence levels
2. Path reveals associative chains
3. Amplitude detects domain boundaries
4. Non-convergence signals genuine ambiguity
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from janet import Sorter


def experiment_cost_as_confidence():
    """Cost = confidence. Low cost = certain. High cost = uncertain."""
    print("=" * 70)
    print("PART 1: Cost as confidence signal")
    print("=" * 70)
    print()

    corpus = [
        "entropy increases in isolated thermodynamic systems over time",
        "information erasure costs energy according to Landauer principle",
        "Shannon entropy measures uncertainty in communication channels",
        "thermodynamic entropy and information entropy are deeply related",
        "reversible computation does not require energy dissipation",
        "Maxwell demon sorts molecules but pays irreversible memory cost",
        "Boltzmann constant relates microscopic energy to macroscopic temperature",
        "heat flows spontaneously from hot regions to cold regions",
    ]

    s = Sorter()
    s.learn_batch(corpus)

    # Queries at different "distances" from the corpus
    queries = [
        ("entropy increases isolated thermodynamic", "exact match (inside sphere)"),
        ("entropy information", "partial match (near surface)"),
        ("energy cost computation", "tangential (on surface)"),
        ("xyzzy plugh garply waldo", "foreign (outside sphere)"),
    ]

    print(f"  {'Query':>40s} | {'Cost':>4s} | {'Amp':>5s} | {'Conv':>4s} | Interpretation")
    print(f"  {'-'*40}-+-{'-'*4}-+-{'-'*5}-+-{'-'*4}-+-{'-'*20}")

    for query, label in queries:
        o = s.orbit(query)
        print(f"  {query:>40s} | {o['cost']:>4d} | {o['amplitude']:>5.2f} | {'yes' if o['converged'] else 'no':>4s} | {label}")

    print()
    print("  → Cost + amplitude together give a confidence signal:")
    print("     cost=1, amp=1.0: certain (self-recognition)")
    print("     cost=2, amp>0.5: confident (strong match)")
    print("     cost=5, amp≈0.0: uncertain (outside domain)")
    print()


def experiment_path_as_chain():
    """Path = associative chain. Shows HOW the system got to the answer."""
    print("=" * 70)
    print("PART 2: Path as associative chain")
    print("=" * 70)
    print()

    corpus = [
        "entropy increases in isolated thermodynamic systems",
        "thermodynamic entropy relates to molecular disorder",
        "molecular disorder measured by Boltzmann formula",
        "Boltzmann constant links microscopic and macroscopic",
        "macroscopic temperature emerges from molecular motion",
        "information entropy measures uncertainty in messages",
        "Shannon channel capacity limits communication rate",
        "communication requires energy for signal transmission",
    ]

    s = Sorter()
    s.learn_batch(corpus)

    queries = [
        "entropy disorder Boltzmann",
        "information communication energy",
        "entropy",
    ]

    for query in queries:
        o = s.orbit(query, max_steps=8)
        print(f"  Query: \"{query}\"")
        print(f"  Path ({o['steps']} steps, converged={o['converged']}):")
        for i, entry in enumerate(o["path"]):
            marker = " ← answer" if i == len(o["path"]) - 1 else ""
            print(f"    {i+1}. {entry[:55]}{marker}")
        print()

    print("  → The path shows the chain of associations.")
    print("  → Different queries to the same answer take different paths.")
    print("  → The path IS the explanation of why this answer was chosen.")
    print()


def experiment_amplitude_as_boundary():
    """Amplitude = domain detection. Low amplitude = outside the sphere."""
    print("=" * 70)
    print("PART 3: Amplitude as domain boundary detector")
    print("=" * 70)
    print()

    corpus = [
        "entropy increases in isolated thermodynamic systems",
        "information erasure costs energy according to Landauer",
        "Shannon entropy measures uncertainty in bits",
        "thermodynamic entropy and information entropy are related",
        "reversible computation does not require energy dissipation",
        "Maxwell demon sorts molecules but pays memory cost",
    ]

    s = Sorter()
    s.learn_batch(corpus)

    # In-domain vs out-of-domain queries
    queries = [
        ("entropy thermodynamic systems", "in-domain"),
        ("information energy cost", "in-domain"),
        ("Shannon uncertainty bits", "in-domain"),
        ("feline rested upon carpet quietly", "out-of-domain"),
        ("crimson azure vermillion chartreuse", "out-of-domain"),
        ("kubernetes dockerfile orchestration", "out-of-domain"),
    ]

    print(f"  {'Query':>35s} | {'Amplitude':>9s} | {'Domain':>12s}")
    print(f"  {'-'*35}-+-{'-'*9}-+-{'-'*12}")

    for query, domain in queries:
        o = s.orbit(query)
        print(f"  {query:>35s} | {o['amplitude']:>9.4f} | {domain}")

    print()
    print("  → Amplitude separates in-domain from out-of-domain.")
    print("  → No test queries needed — the geometry tells you.")
    print("  → A RAG system would return a result for all of these.")
    print("  → Janet says: amplitude=0 means 'outside my sphere'.")
    print()


def experiment_convergence_as_ambiguity():
    """Non-convergence = genuine ambiguity in the corpus."""
    print("=" * 70)
    print("PART 4: Non-convergence as ambiguity signal")
    print("=" * 70)
    print()

    # Corpus with deliberate ambiguity: two equally valid answers
    corpus = [
        "python is a programming language for general purpose computing",
        "python is a large snake found in tropical regions worldwide",
        "java is a programming language for enterprise applications",
        "java is an island in Indonesia with dense population",
        "mercury is the closest planet to the sun in solar system",
        "mercury is a toxic heavy metal used in old thermometers",
    ]

    s = Sorter()
    s.learn_batch(corpus)

    queries = [
        ("python programming", "unambiguous"),
        ("python snake tropical", "unambiguous"),
        ("python", "ambiguous (language or snake?)"),
        ("java", "ambiguous (language or island?)"),
        ("mercury", "ambiguous (planet or metal?)"),
        ("mercury planet sun", "unambiguous"),
    ]

    print(f"  {'Query':>25s} | {'Cost':>4s} | {'Steps':>5s} | {'Conv':>4s} | {'Note':>30s}")
    print(f"  {'-'*25}-+-{'-'*4}-+-{'-'*5}-+-{'-'*4}-+-{'-'*30}")

    for query, note in queries:
        o = s.orbit(query)
        print(f"  {query:>25s} | {o['cost']:>4d} | {o['steps']:>5d} | {'yes' if o['converged'] else 'NO':>4s} | {note}")

    print()
    print("  → Ambiguous queries have higher cost and may not converge.")
    print("  → This is not a failure — it is information.")
    print("  → 'python' alone is genuinely ambiguous in this corpus.")
    print("  → A RAG system would pick one silently. Janet says: ambiguous.")
    print()


def summary():
    print("=" * 70)
    print("SUMMARY: The orbit is the answer")
    print("=" * 70)
    print()
    print("  The orbit carries four signals that the destination alone does not:")
    print()
    print("  1. COST        → confidence (how certain is this answer?)")
    print("  2. PATH        → explanation (how did we get here?)")
    print("  3. AMPLITUDE   → domain (is this query inside our sphere?)")
    print("  4. CONVERGENCE → ambiguity (is there one answer or many?)")
    print()
    print("  Treating janet as a RAG system (query → answer) discards 3/4")
    print("  of the information. The orbit is the primary output.")
    print()


if __name__ == "__main__":
    print()
    experiment_cost_as_confidence()
    experiment_path_as_chain()
    experiment_amplitude_as_boundary()
    experiment_convergence_as_ambiguity()
    summary()
