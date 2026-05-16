# janet-py

A knowledge system built on geometry and self-reference. The orbit — not the destination — is the answer.

Janet is not a RAG. It is a dynamical system that knows its own health, says when it doesn't know, explains how it got to an answer, and tells you how to make it better.

---

## What this is

A proof of concept for a **self-prescriptive knowledge system**:

1. **The orbit is the answer** — a query evolves through geometric projections until convergence. The full trajectory (path, cost, amplitude, convergence) carries confidence, ambiguity, domain boundaries, and associative chains. The final entry is just where the orbit stopped.
2. **The system measures itself** — phi and self_cost quantify corpus health without test queries. Degradation is detected before accuracy drops.
3. **The system prescribes its own improvement** — `suggest_route()` tells you which tokens create optimal bridges. `improve_routes()` identifies weak entries. The geometry says what to fix.
4. **Structure emerges from 0 + self-reference** — from 1 seed + state-correlated noise + selection, structured corpora self-generate. Networks integrate through use.

Zero infrastructure. One dependency (numpy). Deterministic.

---

## What janet offers that RAG does not

| | RAG | Janet |
|---|---|---|
| Confidence | None (returns top-k blindly) | Orbit cost + amplitude = measurable confidence |
| "I don't know" | Never says it | amplitude=0 → outside the sphere |
| Corpus health | Not measurable | phi, self_cost — no test queries needed |
| Early warning | Discovers problems after accuracy drops | phi detects damage before accuracy drops |
| How to improve | Trial and error | suggest_route, improve_routes — geometry prescribes |
| Explainability | Opaque score | Path = verifiable associative chain |
| Ambiguity | Hides it (returns top-1) | Non-convergence = explicit signal |
| Determinism | No (float embeddings, models change) | Yes (int32, same input = same output forever) |

Janet is for systems where **knowing what you don't know** matters more than answering everything.

---

## Quick start

```bash
pip install numpy
python3 experiment_orbit.py          # the orbit as primary output (confidence, path, domain, ambiguity)
python3 experiments.py               # integration experiments (controls, scale, emergence, phase transition)
python3 experiment_conditionality.py # phi-cost equivalence depends on faithful encoding (A4)
python3 experiment_genesis.py        # autonomous structure formation from 1 seed
python3 experiment_dualism.py        # phi-function equivalence demonstration
python3 experiment_early_warning.py  # phi as early warning (3 degradation modes)
python3 experiment_path_dependence.py # query history effects (3 variants)
```

As MCP server for LLM agents (see [janet-mcp](https://github.com/marcosvcloures/janet-mcp)):
```json
{
  "mcpServers": {
    "janet": {
      "command": "python3",
      "args": ["/path/to/janet-mcp/mcp.py"],
      "env": { "JANET_DIR": "/path/to/your/project" }
    }
  }
}
```

---

## Key results

| Experiment | Finding |
|---|---|
| Orbit as output | Cost=confidence, amplitude=domain boundary, path=explanation, convergence=ambiguity |
| Controls | Coherent corpus: phi=0.095. Independent/identical: phi=0. |
| Scale | Phi decreases with corpus size. Integration is expensive. |
| Emergence | Network phi > max individual phi only with semantic bridges. |
| Phase transition | Remove all bridges → integration collapses abruptly. |
| Genesis | From 1 seed, phi=0.30 emerges via noise + selection. |
| Phi-function equivalence | Phi predicts retrieval accuracy *conditional on faithful encoding*. self_cost is the stronger predictor. |
| Decay vs heal | Surgical token removal preserves phi; entry replacement destroys it. |
| GWT feedback | Cross-domain queries increase network phi (0.06 → 0.63). Network integrates through use. |
| Early warning | Phi drops 2 levels before accuracy. In 2/3 modes, phi detects damage accuracy cannot see. |
| Route field | Dual representation (route + claim) improves phi by 107% and self_cost by 45%. |

All results are deterministic. Same input = same output on any platform.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Orbit (the primary output)                             │
│                                                         │
│  query (initial condition)                              │
│    → encode → argmax → re-encode → argmax → ...        │
│    → convergence (fixed point) or max steps (ambiguity) │
│                                                         │
│  Output = full trajectory:                              │
│    path:        [entry₁, entry₂, ..., entryₙ]          │
│    cost:        n (steps to converge = confidence⁻¹)    │
│    amplitude:   initial alignment (domain membership)   │
│    converged:   bool (unambiguous vs ambiguous)         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Self-prescription (the feedback loop)                  │
│                                                         │
│  suggest_route(claim) → geometry says which tokens      │
│                         create optimal bridges          │
│  improve_routes()     → identifies weak entries         │
│  phi / self_cost      → measures corpus health          │
│  decay_unstable()     → removes noise, preserves phi   │
│                                                         │
│  The system measures → diagnoses → prescribes.          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Node (Sorter)                                          │
│                                                         │
│  Corpus: N entries encoded as int32 vectors             │
│  Retrieval: argmax(dot(query, entries))                 │
│  Integration: whole - best partition (phi)              │
│  Resonance: average cross-prediction (self_cost)        │
│  Generation: geometry says direction, corpus says words │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Network (Global Workspace)                             │
│                                                         │
│  Specialized nodes compete → winner broadcasts →        │
│  participating nodes learn → network integrates         │
│  through USE, not just topology.                        │
└─────────────────────────────────────────────────────────┘
```

---

## Mechanisms

| Mechanism | What it does |
|---|---|
| `orbit()` | **Primary output.** Full trajectory: path, cost, amplitude, convergence. |
| `suggest_route()` | **Self-prescription.** Geometry recommends routing tokens for new entries. |
| `improve_routes()` | **Self-prescription.** Identifies weak entries and suggests better routes. |
| `atom_stability()` | **Periodic table.** Measures each token's contribution to integration. |
| `phi()` | Measure integration from outside (whole - best partition) |
| `self_cost()` | Measure integration from inside (average resonance). Stronger predictor. |
| `decay_unstable()` | Remove least stable token from weakest entry (preserves phi) |
| `generate_coherent(state)` | Find corpus fragments aligned with a novel direction |

### Why two integration measures?

`phi()` requires an external observer: it tests all partitions and compares. Detects phase transitions and structural collapse.

`self_cost()` is what the system experiences: how well do its parts resonate? No observer needed. Stronger predictor of accuracy (r=0.82 vs r=0.39 for phi).

They are complementary:
- **phi > 0 and self_cost < 0.5** → integrated (coherent, differentiated)
- **phi = 0 and self_cost ≈ 0** → redundant (identical entries, no differentiation)
- **phi = 0 and self_cost ≈ 1** → fragmented (independent, no connection)

---

## Conditional axioms

The phi-cost equivalence holds when — and only when — four conditions are satisfied:

| Axiom | Statement | In code |
|---|---|---|
| A1. Finite inner-product space | States are vectors in ℤ^d with exact dot product | `Vec = np.ndarray`, `dot(a,b) = int64` |
| A2. Convergent projection | Retrieval contracts the search space at each step | `orbit`: encode → argmax → re-encode → repeat |
| A3. Self-referential ground state | A fixed-point attractor prevents degenerate convergence | `seed_fixed_point()`: first entry amplified |
| A4. Faithful encoding | The encoding preserves structure from the source domain | DCE: shared tokens → geometric alignment. Controllable via `route` field + `suggest_route()`. |

**A1–A3 are properties of the system.** Always satisfied by construction.

**A4 is now controllable.** `suggest_route()` uses corpus geometry to recommend tokens that create optimal bridges. `improve_routes()` identifies entries with weak connections. The system is self-prescriptive: it measures its own health and tells you how to improve it. A4 is satisfied by design, not by luck.

**self_cost is the stronger predictor.** In controlled degradation experiments:
- Correlation(self_cost, accuracy) = 0.82
- Correlation(phi, accuracy) = 0.39

---

## The claim

> Janet is a self-prescriptive knowledge system built on geometry and self-reference.
>
> Given a corpus with faithful encoding (A4), three properties are equivalent:
> integration (phi), retrieval cost (orbit steps), and corpus structure (Gram matrix).
> They are one property measured three ways.
>
> The system is self-prescriptive: it measures its own health (phi, self_cost),
> detects degradation before accuracy drops (early warning), and tells you how
> to improve it (suggest_route, improve_routes). The geometry prescribes.
>
> The orbit — not the destination — is the primary output. It carries confidence,
> ambiguity, domain boundaries, and associative chains. A system that only returns
> the destination discards most of the information.
>
> These are empirical results about this specific system. Not claims about
> consciousness, physics, or general AI.

---

## Use cases

**As a research tool** (this repo — janet-py):
- Fast testbed for integration experiments (phi in <1s)
- Reproducible: deterministic, same input = same output
- All experiments runnable with `pip install numpy`

**As a knowledge system for LLM agents** ([janet-mcp](https://github.com/marcosvcloures/janet-mcp)):
- Orbit gives the LLM metacognition (confidence, domain, ambiguity)
- suggest_route tells the LLM how to write entries for optimal routing
- phi/self_cost let the LLM monitor corpus health across sessions
- Deterministic memory that persists across context windows
- Federated centers: each specialist has N=20-50 entries, queries auto-route via GWT
- Scales horizontally: more centers, not more entries per center

---

## Operational costs

| N entries | orbit (query) | self_cost | phi | learn_batch |
|---|---|---|---|---|
| 10 | 0.1ms | <1ms | <1ms | 2ms |
| 50 | 0.1ms | 1ms | 25ms | 3ms |
| 100 | 0.3ms | 5ms | 350ms | 6ms |
| 200 | 1ms | 42ms | 5.5s | 14ms |
| 500 | 6ms | 550ms | impractical | 51ms |
| 1000 | 113ms | 4.8s | impractical | 188ms |

**Retrieval (orbit) scales to thousands.** O(N) per query — just a matrix-vector dot product.

**self_cost scales to hundreds.** O(N²) — practical for real knowledge systems. And it's the stronger predictor of accuracy (r=0.82).

**phi is limited to N≤200.** O(N³) — useful for experiments and small corpora. For production monitoring, use self_cost.

For a typical knowledge system (50-200 entries per center), all operations are real-time.

---

## Limitations

1. **No semantic understanding.** Janet routes by token co-occurrence, not meaning. "free energy" (in corpus) and "Gibbs free energy" (not in corpus) share tokens → amplitude is high for both. The system cannot distinguish related-but-different concepts that share vocabulary.

2. **Adjacent queries are not detected.** amplitude=0 works perfectly for foreign queries (zero token overlap). But queries from the same field that share vocabulary with the corpus get high amplitude even when the specific answer doesn't exist. The LLM must judge whether the returned entry actually answers the question.

3. **phi is O(N³).** Only computable for N≤200. For production, use self_cost (O(N²), stronger predictor).

4. **Orbits are short.** With small corpora, most orbits converge in 2 steps. The path rarely has more than 1-2 entries. Longer, more informative orbits require denser corpora.

5. **Requires vocabulary discipline.** A4 (faithful encoding) depends on entries sharing tokens deliberately. Random or inconsistent vocabulary breaks the geometry. The system rewards discipline and punishes sloppiness.

6. **Not a replacement for RAG.** For large-scale retrieval with semantic understanding (paraphrases, synonyms), transformer-based RAG is better. Janet is for systems where determinism, measurable confidence, and self-prescription matter more than coverage.

---

## Open questions

1. **Is phi ∝ 1/cost a theorem?** Both are functions of the Gram matrix. The correlation is perfect when encoding preserves structure. Open: formal proof for convergent-projection + fixed point + faithful encoding.

2. **Does it generalize?** The equivalence holds for int32, N≤64. Does it hold for float embeddings, larger corpora? Hypothesis: systems without geometric funneling (brute-force kNN) show phi=0 regardless of corpus quality.

3. **Is the phase transition universal?** Bridge removal causes phi to collapse abruptly. Does this belong to a known universality class (percolation)?

4. **Does learning necessarily increase integration?** GWT feedback increases phi automatically. Does *any* learning-through-use rule necessarily increase integration?

5. **Phi as early warning at scale?** Confirmed for N≤64. Does it hold for N=10³–10⁶ and real-world systems?

---

## Files

```
janet.py                      — core (encoding, phi, orbit, generation, decay)
network.py                    — network of nodes (GWT routing, broadcast, feedback)
experiments.py                — integration experiments (controls, scale, emergence, phase)
experiment_orbit.py           — orbit as primary output demonstration
experiment_conditionality.py  — phi-cost equivalence depends on faithful encoding (A4)
experiment_genesis.py         — autonomous structure formation from 1 seed
experiment_dualism.py         — phi-function equivalence
experiment_early_warning.py   — phi as leading indicator (3 degradation modes)
experiment_path_dependence.py — query history effects (3 variants)
know.py                       — corpus CLI
preprint/                     — paper (methods, results, discussion)
```

---

## Theory correspondence

| Concept | Implementation | Validity |
|---|---|---|
| Finite reservoir | Corpus (N entries, int32 vectors) | Exact |
| Observer (demon) | Sorter (observe, select, pay cost) | Exact |
| Landauer erasure | Each orbit step = 1 irreversible operation | Exact |
| Shannon entropy | Energy distribution across subspaces | Exact |
| Integration (IIT proxy) | Whole - best bipartition | Simplified proxy |
| Global workspace (GWT) | Network routing + broadcast + feedback | Structural implementation |

These are structural analogies, not claims of physical equivalence.

---

*Self-prescriptive knowledge. Geometry as ground truth. The orbit is the answer.*
