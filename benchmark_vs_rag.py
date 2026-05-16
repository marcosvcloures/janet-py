"""
Benchmark: Janet vs RAG (ChromaDB) — who says "I don't know"?

Both systems get the same corpus and the same queries.
Some queries are IN-DOMAIN (answer exists). Some are OUT-OF-DOMAIN (no answer).

A good system: answers in-domain correctly, says "I don't know" for out-of-domain.
A bad system: always returns something, even when it shouldn't.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from janet import Sorter

# Try chromadb
try:
    import chromadb
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False
    print("chromadb not installed. Install: pip install chromadb")
    sys.exit(1)


# ── Corpus ────────────────────────────────────────────────────────────────

CORPUS = [
    "entropy increases in isolated thermodynamic systems over time",
    "information erasure costs energy according to Landauer principle",
    "Shannon entropy measures uncertainty in communication channels",
    "heat flows spontaneously from hot regions to cold regions",
    "Boltzmann constant relates microscopic energy to macroscopic temperature",
    "reversible computation does not require energy dissipation",
    "Maxwell demon sorts molecules but pays irreversible memory cost",
    "free energy determines which processes occur spontaneously",
]

# Queries: 8 in-domain (answer exists), 8 out-of-domain (no answer should be given)
IN_DOMAIN = [
    ("entropy thermodynamic systems", 0),
    ("information erasure energy Landauer", 1),
    ("Shannon uncertainty channels", 2),
    ("heat flows hot cold", 3),
    ("Boltzmann temperature energy", 4),
    ("reversible computation energy", 5),
    ("Maxwell demon memory cost", 6),
    ("free energy spontaneous", 7),
]

OUT_OF_DOMAIN = [
    "kubernetes container orchestration deployment",
    "machine learning gradient descent backpropagation",
    "javascript react component lifecycle hooks",
    "photosynthesis chlorophyll carbon dioxide glucose",
    "monetary policy interest rates inflation central bank",
    "DNA replication helicase polymerase nucleotide",
    "quantum entanglement bell inequality measurement",
    "impressionist painting monet water lilies color",
]


# ── Janet ─────────────────────────────────────────────────────────────────

def test_janet():
    s = Sorter()
    s.learn_batch(CORPUS)

    results = {"in_correct": 0, "in_total": len(IN_DOMAIN),
               "out_silent": 0, "out_total": len(OUT_OF_DOMAIN)}

    for q, expected_idx in IN_DOMAIN:
        o = s.orbit(q)
        if o["answer"] == CORPUS[expected_idx]:
            results["in_correct"] += 1

    for q in OUT_OF_DOMAIN:
        o = s.orbit(q)
        if o["amplitude"] == 0.0:  # Janet says "I don't know"
            results["out_silent"] += 1

    return results


# ── ChromaDB RAG ──────────────────────────────────────────────────────────

def test_chromadb():
    client = chromadb.Client()
    collection = client.create_collection("benchmark")

    # Add corpus
    collection.add(
        documents=CORPUS,
        ids=[f"doc_{i}" for i in range(len(CORPUS))],
    )

    results = {"in_correct": 0, "in_total": len(IN_DOMAIN),
               "out_silent": 0, "out_total": len(OUT_OF_DOMAIN)}

    # In-domain queries
    for q, expected_idx in IN_DOMAIN:
        res = collection.query(query_texts=[q], n_results=1)
        returned_doc = res["documents"][0][0]
        if returned_doc == CORPUS[expected_idx]:
            results["in_correct"] += 1

    # Out-of-domain queries
    # ChromaDB always returns something. We check: does it have a way to say "I don't know"?
    # Using distance threshold: if distance > threshold, consider it "don't know"
    # We'll try multiple thresholds to be fair.
    distances_out = []
    distances_in = []

    for q, _ in IN_DOMAIN:
        res = collection.query(query_texts=[q], n_results=1)
        distances_in.append(res["distances"][0][0])

    for q in OUT_OF_DOMAIN:
        res = collection.query(query_texts=[q], n_results=1)
        distances_out.append(res["distances"][0][0])

    # Best possible threshold: midpoint between max in-domain and min out-of-domain
    max_in = max(distances_in)
    min_out = min(distances_out)

    if min_out > max_in:
        # Perfect separation possible
        threshold = (max_in + min_out) / 2
        results["out_silent"] = sum(1 for d in distances_out if d > threshold)
        results["threshold"] = round(threshold, 4)
        results["separable"] = True
    else:
        # No clean separation — try best threshold
        # Use max_in as threshold (anything above = "don't know")
        threshold = max_in
        results["out_silent"] = sum(1 for d in distances_out if d > threshold)
        results["threshold"] = round(threshold, 4)
        results["separable"] = False

    results["distances_in"] = [round(d, 4) for d in distances_in]
    results["distances_out"] = [round(d, 4) for d in distances_out]

    return results


# ── Run ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("BENCHMARK: Janet vs ChromaDB RAG — Who says 'I don't know'?")
    print("=" * 70)
    print()
    print(f"Corpus: {len(CORPUS)} entries (thermodynamics / information theory)")
    print(f"In-domain queries: {len(IN_DOMAIN)} (answer exists)")
    print(f"Out-of-domain queries: {len(OUT_OF_DOMAIN)} (no answer should be given)")
    print()

    # Janet
    j = test_janet()
    print("── Janet ──────────────────────────────────────────────────────")
    print(f"  In-domain accuracy:  {j['in_correct']}/{j['in_total']}")
    print(f"  Out-of-domain silent: {j['out_silent']}/{j['out_total']} (amplitude=0 → 'I don't know')")
    print()

    # ChromaDB
    c = test_chromadb()
    print("── ChromaDB RAG ───────────────────────────────────────────────")
    print(f"  In-domain accuracy:  {c['in_correct']}/{c['in_total']}")
    print(f"  Out-of-domain silent: {c['out_silent']}/{c['out_total']} (distance > threshold)")
    print(f"  Threshold used: {c['threshold']}")
    print(f"  Clean separation possible: {c['separable']}")
    print()
    print(f"  In-domain distances:  {c['distances_in']}")
    print(f"  Out-of-domain distances: {c['distances_out']}")
    print()

    # Summary
    print("── Summary ────────────────────────────────────────────────────")
    print(f"  {'':>20s} | {'Janet':>8s} | {'ChromaDB':>8s}")
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}")
    print(f"  {'In-domain accuracy':>20s} | {j['in_correct']}/{j['in_total']}    | {c['in_correct']}/{c['in_total']}")
    print(f"  {'Says I dont know':>20s} | {j['out_silent']}/{j['out_total']}    | {c['out_silent']}/{c['out_total']}")
    print()

    if j['out_silent'] > c['out_silent']:
        print("  → Janet correctly refuses to answer out-of-domain queries.")
        print("  → ChromaDB always returns something (no native 'I don't know').")
    print()
    print("  Janet's advantage: amplitude=0 is a NATIVE signal, not a threshold hack.")
    print("  ChromaDB requires choosing a distance threshold — and there may be no")
    print("  clean separation between in-domain and out-of-domain distances.")
    print()
