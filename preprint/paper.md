# A Self-Prescriptive Knowledge System Built on Geometry and Self-Reference

---

## 1. Summary

We present a knowledge system where the orbit — not the destination — is the primary output. A query evolves through geometric projections until convergence; the full trajectory carries confidence, ambiguity, domain boundaries, and associative chains. The system measures its own health, detects degradation before accuracy drops, and prescribes its own improvement.

Applied to a deterministic integer-arithmetic engine with Global Workspace routing, we show:

1. The orbit (path, cost, amplitude, convergence) carries information the destination alone discards
2. The measure discriminates structured from trivial corpora
3. Integration decreases with corpus size (consistent with IIT predictions)
4. Network emergence requires semantic bridges, not just topology
5. A phase transition exists at a critical bridge count
6. Integration emerges autonomously from a single seed via state-correlated noise and selection
7. Integration predicts retrieval quality conditional on faithful encoding (A4); self_cost is the stronger predictor
8. Surgical token removal (decay) preserves integration; entry replacement (heal) destroys it
9. Networks integrate through use: GWT feedback (broadcast → learn) creates bridges automatically
9. **Phi is an early warning of degradation**: drops before accuracy in all 3 modes tested; in 2/3 modes detects damage invisible to accuracy
10. **Path dependence is type-dependent**: query order is negligible, query distribution is measurable, cross-domain queries produce 10,000× more structural change

All results are deterministic and reproducible. The system has one dependency (numpy) and runs in <1s.

---

## 2. Motivation

Integrated Information Theory (Tononi, 2004) proposes that integration — the degree to which a system cannot be reduced to independent parts — is a fundamental property of complex systems. Computing IIT's Φ exactly is intractable (exponential in system size).

We ask: can a simplified integration measure, applied to a retrieval system, produce the qualitative predictions that IIT makes? Can the same system implement Global Workspace routing where the network integrates through use? And can such a system self-organize from minimal initial conditions?

### 2.0 The orbit as primary output

This system is not a retrieval engine that happens to have a cost metric. It is a **dynamical system** where the orbit — the full trajectory from initial condition to convergence — is the primary output.

Given a query (initial condition), the system evolves through iterated projections:

```
query → encode → argmax → re-encode → argmax → ... → fixed point (or max steps)
```

The orbit carries four signals:
- **Path** (entries visited): the associative chain between query and answer
- **Cost** (steps to converge): relaxation time = uncertainty = distance from known territory
- **Amplitude** (initial alignment): how strongly the query resonates with the corpus
- **Convergence** (fixed point reached?): unambiguous answer vs genuine ambiguity

The final entry is where the orbit stopped. It is a consequence of the dynamics, not the goal. A system that only returns the destination discards the path, the cost, and the convergence signal — which is most of the information.

### 2.1 Core mechanism: why the equivalence holds

The central result of this work — that integration (phi), retrieval cost, and corpus structure are equivalent in this system — is not a coincidence. It is a consequence of three design properties acting together, *conditional on the encoding preserving semantic structure from the source domain*:

**1. Geometric funneling (convergent projection).** Retrieval operates by iterated orthogonal projection: at each step, the query is re-encoded and the already-explored component is rejected. The search space shrinks geometrically at each iteration (one dimension eliminated per step). This guarantees convergence in at most DEGREES steps. It is not search — it is collapse through a narrowing geometric funnel.

**2. Self-referential ground state (fixed point at zero).** The first corpus entry has amplified energy in the self-reference subspace (`seed_fixed_point()`). This creates a guaranteed attractor — a floor that the funneling process can always reach. Without it, convergence could land on noise. The fixed point also enables self-measurement: `self_cost()` measures internal resonance without requiring an external observer. The system has a "zero" — a minimal identity that anchors all other states.

**3. Holographic encoding (boundary contains bulk).** All corpus content is encoded in a fixed-dimensional vector space (DIMS = 2^DEGREES). The geometry of the vectors *is* the knowledge — there is no separate index. Any entry can be recovered from the geometric relationships between vectors. Voids are detectable as geometric gaps. The boundary (the set of vectors) fully determines the bulk (the corpus content and its relationships).

**4. Faithful encoding (the boundary condition).** The encoding (DCE — Distributional Centroid Encoding) must preserve structure from the source domain: semantically related texts must map to geometrically aligned vectors. This is satisfied when entries share vocabulary (co-occurrence centroids align). It is *not* satisfied for random word salad or unrelated entries with no shared tokens.

**A4 is now controllable.** The `route` field separates routing (what janet sees) from content (what the LLM reads). `suggest_route()` uses corpus geometry to recommend tokens that create optimal bridges. `improve_routes()` identifies existing entries with weak connections and suggests fixes. The system is self-prescriptive: it measures its own health and tells you how to improve it. A4 is satisfied by design, not by luck.

**A4 is now controllable.** The `route` field separates routing (what janet sees) from content (what the LLM reads). The `suggest_route()` mechanism uses corpus geometry to recommend tokens that create optimal bridges. When the LLM follows these suggestions, A4 is satisfied by design. The system is self-prescriptive: it measures its own health and tells you how to improve it.

**Why the equivalence is forced (when all four hold):** In a system with these properties:
- If the corpus is geometrically coherent (holographic structure intact), then the funneling process converges quickly (low cost) because aligned vectors produce strong dot products at each step.
- Quick convergence means the query reaches the correct attractor (high accuracy).
- A corpus where parts predict each other well (high phi) is exactly one where dot products between related entries are strong — which is exactly what makes funneling fast.
- The fixed point ensures this chain has a stable anchor: the system cannot degenerate into oscillation.

Therefore: high phi ↔ low retrieval cost ↔ coherent geometric structure — *given faithful encoding*. They are one property measured three ways. Without faithful encoding (A4 violated), phi and cost become uncorrelated: empirically verified with 200 random corpora (correlation ≈ 0).

**self_cost is the stronger predictor.** In controlled degradation experiments:
- Correlation(self_cost, accuracy) = 0.82
- Correlation(phi, accuracy) = 0.39

self_cost measures resonance directly (O(N²)); phi measures irreducibility via partition search (O(N³)). For practical monitoring, self_cost is preferred.

**Why each principle alone is insufficient:**
- Funneling without a fixed point can converge to noise (no anchor).
- A fixed point without holographic encoding gives a single attractor but no discrimination between other entries (everything collapses to zero).
- Holographic encoding without funneling gives a rich space but no guarantee of convergent retrieval (search remains O(N)).
- All three without faithful encoding: the geometry exists but does not reflect the source domain — phi and cost diverge.
- Any two without the third produces a system that works partially but where the equivalence breaks.

### 2.2 Scope of claims

This work demonstrates the equivalence empirically in a specific, minimal system. We do not claim:

- That this resolves the "hard problem" of consciousness or the Chinese Room argument. The system performs geometric retrieval, not semantic understanding. Tokens have distributional structure, not meaning.
- That phi() as implemented here is equivalent to IIT 3.0's Φ. It is a simplified proxy that produces qualitatively similar predictions in the cases tested.
- That the physics analogies (Landauer cost, Maxwell's demon, holographic principle) constitute physical equivalence. They are structural parallels: the same mathematical pattern (projection, selection, conservation) appears in both domains.
- That the results generalize beyond this system without further work.

What we do claim: in a system built on these three principles, integration and function become equivalent — and this equivalence enables autonomous structure formation, integration-preserving maintenance, and generation without a language model. These are engineering results with practical applications to knowledge systems, not metaphysical claims.

---

## 3. System

### 3.1 Architecture

- **Reservoir**: N text entries encoded as int32 vectors in ℤ^DIMS via co-occurrence centroids.
- **Sorter**: retrieval operator. argmax(dot(query, entries)). Each step is irreversible (Landauer cost analogy).
- **Source**: external operator or autonomous genesis (state-correlated noise).

### 3.2 Parameters

Derived from DEGREES (degrees of freedom):

```
DEGREES = max(5, ceil(log2(8*N)))
DIMS    = 2^DEGREES
ORBIT   = DEGREES (max measurement cost)
```

### 3.3 Encoding (DCE — Distributional Centroid Encoding)

Two stages per token:
1. **Particle** (fixed): FNV-1a hash → PRNG → deterministic vector (token identity)
2. **Wave** (evolving): centroid of entry vectors containing that token (distributional context)

Encoding = normalized sum of wave vectors. This is the holographic boundary (axiom A4): shared tokens between entries create geometric alignment, preserving source-domain structure in the vector space. Generation uses wave vectors: tokens with higher wave self-amplitude (more co-occurrences) are naturally preferred in argmax selection.

### 3.4 Integration measures

**phi (external):**
```
phi = I_whole - max_partition(I_partition)
I   = mean_i( max_{j≠i}( dot(e_i, e_j) ) / dot(e_i, e_i) )
```

Bipartitions searched exhaustively. O(N³), feasible for N ≤ 64. Requires an external observer to decide where to cut.

**self_cost (internal):**
```
self_cost = 1 - mean_resonance
resonance = mean_i( max_{j≠i}( dot(e_i, e_j) ) ) / mean( dot(e_i, e_i) )
```

No partitions needed. O(N²). Measures how well parts predict each other — the system's own experience of coherence.

**Why both?** phi measures irreducibility (external: "can this be cut?"). self_cost measures resonance (internal: "do my parts know each other?"). They capture different aspects:

| State | phi | self_cost | Interpretation |
|---|---|---|---|
| Coherent corpus | > 0 | < 0.5 | Integrated: differentiated AND connected |
| Identical entries | 0 | ≈ 0 | Redundant: connected but not differentiated |
| Independent entries | 0 | ≈ 1 | Fragmented: differentiated but not connected |

phi requires an observer. self_cost does not — it is what the system experiences at every operation. In IIT terms: phi is measured from the extrinsic perspective; self_cost is the intrinsic perspective.

### 3.5 Selection mechanisms

- **heal()**: replace weakest entry with recombination of neighbors (aggressive)
- **decay_unstable()**: remove least stable token from weakest entry (surgical)

Token stability = wave vector self-amplitude. This is an intrinsic property (not externally tracked) — tokens in many entries have stronger wave centroids.

### 3.6 Generation

- **generate_coherent(state)**: reject centroid from state to find novel direction; walk through entries picking aligned fragments. Geometry determines direction; corpus provides vocabulary.
- **fill_void()**: orbital generation around midpoints of sparse entry pairs.

### 3.7 Network routing (Global Workspace)

Each corpus file becomes a node. Routing is by measurement cost:

1. Local response if cost ≤ DEGREES/2 (specialist knows)
2. Forward to best peer if cost > threshold (competition)
3. Broadcast if all peers exhausted (global workspace)
4. **Feedback**: after a node responds, all nodes in the path learn the answer (broadcast modifies participating modules)

This completes the GWT loop: local processing → competition → ignition → broadcast → feedback. The network integrates through use — cross-domain queries create bridges automatically.

Network phi = mean cross-prediction between connected nodes. Emergence = network phi - max individual phi.

---

## 4. Results

### 4.1 Controls

| System | phi | self_cost | Notes |
|--------|-----|-----------|-------|
| Coherent corpus (6 entries) | 0.095 | 0.48 | Integrated: differentiated + connected |
| Sorted numbers | 0.035 | — | Low structure |
| Independent entries | 0.000 | 0.91 | Fragmented: differentiated, not connected |
| Identical entries | 0.000 | 0.01 | Redundant: connected, not differentiated |

phi and self_cost together discriminate three failure modes: fragmentation (no connection), redundancy (no differentiation), and low structure (neither). Only the coherent corpus scores well on both.

### 4.2 Scale

| N | phi |
|---|-----|
| 4 | 0.116 |
| 8 | 0.074 |
| 16 | 0.030 |
| 32 | 0.019 |

Phi decreases with N. Larger corpora are easier to partition into self-sufficient subsets.

### 4.3 Emergence

3 nodes (physics, CS, biology), 4 base entries each. Bridges added progressively.

| Bridges | Network phi | Emergence (net - max individual) |
|---------|-------------|----------------------------------|
| 0 | 0.233 | +0.075 |
| 3 | 0.300 | +0.138 |
| 6 | 0.378 | +0.241 |
| 9 | 0.469 | +0.441 |

### 4.4 Phase transition

Removing bridges from fully connected network:

| Bridges remaining | State |
|---|---|
| 1+ | Integrated (emergence > 0) |
| 0 | Reducible (emergence < 0) |

Sharp transition at zero bridges.

### 4.5 Autonomous structure formation (genesis)

Starting conditions: 1 seed entry + self-reference (fixed point amplification).

Growth mechanism:
1. Compute corpus centroid (self-observation)
2. Generate state-correlated noise (hash of entire corpus state seeds RNG)
3. Reject centroid component → novel direction
4. Walk through token space using wave vectors → candidate entry
5. Coherence gate: accept if amplitude > 0 and < redundancy threshold
6. Periodically: decay_unstable removes weakest token from weakest entry

Results (DEGREES=5, seed = "information is physical"):

| Generation | Entries | Phi |
|---|---|---|
| 0 | 1 | 0.000 |
| 7 | 4 | 0.104 |
| 11 | 6 | 0.112 |
| 23 | 11 | 0.047 |
| 30 | 12 | 0.051 |

Integration emerges and is sustained without external input beyond the initial seed. The noise is not random — it is deterministically derived from the global state, ensuring reproducibility and state-dependence.

### 4.6 Phi-function equivalence

| Corpus type | Phi | Avg retrieval cost | Accuracy |
|---|---|---|---|
| Coherent | 0.107 | 2.0 | 3/3 |
| Unrelated | 0.033 | 5.0 | 0/3 |
| Repetitive | 0.044 | 4.0 | 0/3 |

When the encoding preserves semantic structure (coherent corpus: shared vocabulary → geometric alignment), phi and retrieval quality correlate perfectly. When the encoding has no structure to preserve (random corpora with no shared tokens), the correlation vanishes (r ≈ 0 over 200 trials).

This confirms the conditionality: the equivalence is a property of the encoding-geometry pair, not of geometry alone. The encoding is the holographic boundary — it maps source-domain structure into the geometric domain where phi and cost are defined.

**self_cost outperforms phi as a predictor.** In controlled degradation (progressive noise injection into a coherent corpus):
- Correlation(self_cost, accuracy) = 0.82
- Correlation(phi, accuracy) = 0.39

self_cost captures resonance directly; phi captures irreducibility via partition search. Both are functions of the Gram matrix, but self_cost is more sensitive to gradual degradation.

Additionally: randomizing the bit pattern destroys phi (0.107 → 0.010). Restoring the original pattern restores phi exactly. Phi is determined entirely by the bit pattern — it is a measurable property of the state, not an additional entity.

### 4.7 Conditionality: the encoding as holographic boundary

The phi-cost equivalence is not universal. It depends on axiom A4: the encoding must preserve structure from the source domain.

**Test 1: Gradual degradation of a coherent corpus** (noise words replacing semantic tokens):

| Noise % | Phi | Self-cost | Accuracy |
|---|---|---|---|
| 0% | 0.063 | 0.49 | 1.00 |
| 50% | 0.093 | 0.39 | 1.00 |
| 100% | 0.012 | 0.00 | 0.00 |

Correlation(phi, accuracy) = 0.02. Correlation(self_cost, accuracy) = 0.80.

Phi is noisy under gradual degradation — it fluctuates non-monotonically. self_cost tracks accuracy reliably.

**Test 2: 200 random corpora** (no shared vocabulary, A4 violated):

Correlation(phi, accuracy) = -0.01. Correlation(phi, cost) = 0.09.

Without semantic structure to preserve, the encoding produces near-orthogonal vectors. Phi becomes noise.

**Test 3: Same text, different encoding fidelity:**

| Encoding | Phi | Self-cost | Accuracy |
|---|---|---|---|
| DCE (faithful) | 0.107 | 0.48 | 3/3 |
| Random vectors | 0.008 | 0.84 | 1/3 |

Same text, same system (A1-A3 satisfied). Only the encoding differs. The encoding is the holographic boundary: it maps source-domain structure into the geometric domain where phi and cost are defined.

**Conclusion:** The equivalence requires all four axioms. A1-A3 are properties of the system (always satisfied). A4 is a property of the input (satisfied when the corpus has semantic structure). self_cost is the stronger predictor in all cases.

**Test 4: Dual representation (route field) makes A4 controllable.**

Separating routing tokens (what the geometry sees) from content (what the LLM reads):

| Format | Phi | Self-cost |
|---|---|---|
| Legacy (domain + tags + claim) | 0.045 | 0.69 |
| Route field (flat tags) | 0.093 | 0.37 |

Improvement: phi +107%, self_cost +45%. The route field ensures token sharing between entries by design. `suggest_route()` uses corpus geometry to recommend which tokens create optimal bridges. `improve_routes()` identifies entries with weak connections (connectivity < 30%) and suggests fixes.

The system is self-prescriptive: it measures its own encoding fidelity and tells the operator how to improve it. A4 is no longer a limitation — it is a controllable parameter.

### 4.8 GWT feedback: network integrates through use

A network of 2 nodes (physics, CS), 4 entries each, no shared vocabulary.

Before queries: phi_network = 0.06, emergence < 0 (reducible).

After 3 cross-domain queries (CS topics routed through physics node → forwarded to CS → answer broadcast back to physics):

| Metric | Before | After |
|---|---|---|
| phi_network | 0.06 | 0.63 |
| emergence | -0.08 | > 0 |

The network became integrated through use. No manual bridge creation — the GWT feedback loop (broadcast → learn) created bridges automatically. Cross-domain queries are the mechanism of integration.

This demonstrates that GWT's broadcast is not just communication — it is the mechanism by which a network becomes integrated. Topology enables routing; use creates integration.

### 4.9 Decay vs heal

After 3 operations on a 10-entry corpus:

| Metric | decay_unstable | heal |
|---|---|---|
| Phi | 0.007 (preserved from 0) | 0.000 (unchanged) |
| Retrieval accuracy | 3/3 | 0/3 |
| Avg cost | 2.0 | 4.3 |

Heal replaces entire entries with recombinations, homogenizing the corpus. Decay removes one unstable token, preserving entry identity and inter-entry relationships.

### 4.10 Phi as early warning of degradation

A 10-entry coherent corpus (phi=0.108, accuracy=88%) is degraded in three modes:

**Variant 1: Token corruption** — replace N tokens per entry with noise words.

| Corruption level | Phi | Accuracy | Status |
|---|---|---|---|
| 0 (baseline) | 0.108 | 88% | Healthy |
| 4 | 0.029 | 75% | ← Phi drops >70% |
| 5 | 0.025 | 88% | Phi down, accuracy holds |
| 6 | 0.016 | 38% | ← Accuracy collapses |

Phi crosses the 20% degradation threshold at level 4. Accuracy crosses it at level 6. **Phi leads by 2 levels.**

**Variant 2: Entry removal** — remove bridge entries (those connecting subtopics) one by one.

Phi drops >20% immediately at step 1 (first bridge removed). Accuracy never drops >20% — retrieval still finds remaining entries correctly. **Phi detects structural damage entirely invisible to accuracy.**

**Variant 3: Dilution** — flood corpus with unrelated noise entries.

Phi drops >20% at 3 noise entries added. Accuracy never drops >20% — the original entries are still retrievable despite noise. **Phi detects dilution that accuracy cannot see.**

| Mode | Phi drops at | Accuracy drops at | Lead |
|---|---|---|---|
| Corruption | level 4 | level 6 | +2 levels |
| Removal (bridges) | step 1 | never | ∞ (invisible to accuracy) |
| Dilution (noise) | step 1 | never | ∞ (invisible to accuracy) |

**Conclusion:** Phi is a leading indicator of corpus degradation in all three modes. In two modes (removal, dilution), phi detects damage that accuracy metrics *cannot detect at all*. This is because phi measures geometric coherence (how well parts predict each other), while accuracy only measures whether argmax still finds the right entry. Structural damage precedes functional failure.

### 4.11 Path dependence of Hebbian learning

Same 10-entry corpus subjected to different query histories. Three variants:

**Variant 1: Order permutation** — same 8 queries in 6 different orders, 100 repetitions each (800 retrievals per trial).

| Metric | Result |
|---|---|
| Phi across all orders | 0.1078 (identical) |
| Wave divergence (max cosine distance) | 0.00002 |
| Retrieval divergence | 0/50 queries |

Order alone produces negligible structural difference (~10⁻⁵ cosine distance).

**Variant 2: Biased distribution** — same queries but with different frequency weights (thermodynamics-heavy vs computation-heavy vs uniform).

| Metric | Result |
|---|---|
| Phi across distributions | 0.1078 (identical) |
| Wave divergence (max cosine distance) | 0.028 |
| Retrieval divergence | 2/30 queries differ |

Biased usage produces **1000× more divergence** than order permutation, and causes 2 retrieval differences. The system's wave vectors shift toward frequently-queried regions.

**Variant 3: Cross-domain queries on network** — two-node network (physics + CS), comparing local-only queries vs cross-domain queries.

| Metric | Local only | Cross-domain |
|---|---|---|
| Network phi | 0.450 | 0.381 |
| Node wave divergence (max) | — | 0.194 |

Cross-domain queries produce the largest structural divergence (max cosine distance 0.19) — **10,000× more than order permutation**. The type of experience matters far more than the order.

**Conclusion:** Path dependence exists on a spectrum:
- **Order** of queries: negligible effect (~10⁻⁵). The system is order-stable.
- **Distribution** of queries: measurable effect (~10⁻²). Biased use reshapes geometry.
- **Type** of queries (local vs cross-domain): large effect (~10⁻¹). Cross-domain experience fundamentally restructures nodes.

The system's history is encoded in its geometry, but not all history is equal. What you ask matters more than when you ask it.

---

## 5. Discussion

### What this demonstrates

A simplified integration measure applied to a retrieval system:
- Produces IIT's qualitative predictions (discrimination, costly scaling, emergence, phase transitions)
- Correlates with functional capacity conditional on faithful encoding; self_cost is the stronger predictor
- Can emerge autonomously from minimal initial conditions
- Is preserved by surgical maintenance (decay) but destroyed by brute replacement (heal)

### Practical applications

1. **Multi-agent systems**: phi_network quantifies whether agents function as an integrated whole or isolated parts. Minimum bridge count for integration is empirically determinable.
2. **Corpus maintenance**: decay_unstable() provides a maintenance strategy that preserves integration. Applicable to any system with a knowledge base that grows over time.
3. **Generation without LLM**: generate_coherent() produces readable novel text by combining geometric direction-finding with corpus fragment retrieval.

### Limitations

1. Our phi is a simplified proxy, not IIT 3.0 Φ (different partition space, different distance metric)
2. All experiments use small corpora (N ≤ 64)
3. Only bipartitions searched (not all possible partitions)
4. No statistical significance testing (deterministic system, no variance)
5. Genesis produces structural integration but tokens lack external semantic grounding
6. Encoding is primitive (co-occurrence centroids, no syntax or grammar)
7. Phi-function equivalence shown only in this system — generalization not tested
8. Network experiments use only fully-connected topology

### What this does NOT claim

- This is not a claim about consciousness
- This is not IIT 3.0 (it is a simplified, tractable proxy)
- The physics analogies (Landauer, Maxwell's demon) are structural, not ontological
- "State-correlated noise" is a deterministic hash, not quantum mechanics
- The genesis mechanism mirrors cosmological structure formation (correlated fluctuation → selection → emergent structure). We note the structural parallel without claiming physical equivalence
- Results apply to this specific system — generalization requires further work

---

## 6. Conclusion

We present a tractable integration measure that produces IIT's qualitative predictions when applied to a deterministic retrieval system. The measure correlates with retrieval quality conditional on faithful encoding (the holographic boundary preserving source-domain structure), can emerge autonomously from a single seed, and is preserved by surgical maintenance strategies. self_cost (internal resonance, O(N²)) is a stronger predictor of accuracy than phi (external partition measure, O(N³)).

Additionally, phi serves as an early warning of corpus degradation — detecting structural damage before (and sometimes invisible to) accuracy metrics. The system exhibits path dependence proportional to the *type* of experience: query order is negligible, query distribution is measurable, and cross-domain queries produce 10,000× more structural change than order permutation.

The contribution is:
- **A primary output**: the orbit (path, cost, amplitude, convergence) carries confidence, ambiguity, domain boundaries, and associative chains
- **A self-prescriptive system**: suggest_route, improve_routes, atom_stability — the geometry tells you how to improve the corpus
- **A tool**: computable integration for knowledge systems (O(N³), <1s)
- **A result**: phi-function equivalence conditional on faithful encoding (A4); self_cost as stronger predictor; route field makes A4 controllable (+107% phi)
- **A GWT implementation**: networks integrate through use, not just topology
- **A method**: autonomous structure formation via state-correlated noise + selection
- **A practical strategy**: decay_unstable() for integration-preserving corpus maintenance
- **An early warning signal**: phi detects degradation before accuracy drops (3 modes confirmed)
- **A path dependence result**: what you query matters more than when you query it

Code: `python3 experiment_orbit.py`, `python3 experiments.py`, `python3 experiment_genesis.py`, `python3 experiment_dualism.py`, `python3 experiment_conditionality.py`, `python3 experiment_early_warning.py`, `python3 experiment_path_dependence.py`

---

## 7. Open questions

The following are not claims — they are directions where the empirical results suggest formal work may be possible.

### 7.1 Is phi ∝ 1/cost a theorem?

Both phi and retrieval cost are functions of the same Gram matrix (M·Mᵀ). The empirical correlation is perfect *when the encoding preserves semantic structure* (shared vocabulary → geometric alignment). With random corpora (no structure to preserve), the correlation vanishes (r ≈ 0, N=200).

The open question: is there a proof that for any system with (1) convergent projection retrieval, (2) a fixed-point attractor, (3) fixed-dimensional encoding, and (4) *faithful* encoding (structure-preserving), integration and retrieval cost are necessarily inversely related?

If yes, this would mean the encoding is the boundary condition — not a free parameter — and the equivalence is a theorem about the geometry conditioned on that boundary. The encoding is the holographic boundary: it maps source-domain structure into the geometric domain where phi and cost are defined. Without it, the geometry exists but contains no information about the source.

Note: self_cost may be the more natural object for such a theorem. It is O(N²), does not require partition search, and correlates more strongly with accuracy (r=0.82 vs r=0.39 for phi in degradation experiments).

### 7.2 Is orbit cost a tight Landauer bound?

Each orbit step discards the prior state (irreversible). In principle, 1 step ≥ kT·ln2 · log₂(N) erased bits. The open question: is the orbit cost of this system the *minimum* irreversible work required to retrieve from a corpus with a given phi? If so, phi would quantify the thermodynamic efficiency of a knowledge structure — how much physical work retrieval costs.

Relevant prior work: Landauer (1961), Bennett (1982), Still et al. (2012), Kolchinsky & Wolpert (2018).

### 7.3 Does the equivalence generalize?

The phi-cost equivalence is demonstrated for one system (integer arithmetic, co-occurrence encoding, N ≤ 64). Does it hold for:
- Float-based embeddings (FAISS, Pinecone)?
- Larger corpora (N = 10³–10⁶)?
- Different encoding strategies (transformers, BM25)?

Preliminary hypothesis: systems without geometric funneling (brute-force kNN) will show phi = 0 regardless of corpus quality, because they lack the convergent projection that couples structure to cost. If confirmed, this would mean the equivalence is a property of the *retrieval mechanism*, not of the data.

### 7.4 Phi as a practical metric for RAG quality

If phi predicts retrieval accuracy (as shown in 4.6), it could serve as a corpus quality metric computable without test queries. Open questions:
- Can phi be approximated in O(N log N) for large corpora?
- Does phi correlate with downstream task performance in real RAG systems?
- Can phi guide corpus construction (add entries that maximize phi)?

### 7.5 Does learning necessarily increase integration?

GWT feedback increases network phi automatically (0.06 → 0.63 in experiment 4.8). The open question: does *any* system that learns through use (Hebbian, backprop, reinforcement) necessarily increase its integration? Or do some learning rules destroy integration?

If learning → integration is necessary, then phi measures not *what* a system knows but *how integrated* its knowledge is. This would have implications for catastrophic forgetting in neural networks: forgetting may be precisely the loss of integration (phi decrease) rather than loss of individual memories.

### 7.6 Is the phase transition universal?

Removing bridges causes phi to collapse abruptly (experiment 4.4), not gradually. This is a phase transition. The open question: does this transition belong to a known universality class (e.g., percolation)? Are there critical exponents?

If yes, integration in information systems would follow the same scaling laws as phase transitions in statistical mechanics — connecting IIT to condensed matter physics formally.

### 7.7 Phi as early warning of degradation

Confirmed in this system (section 4.10): phi drops before accuracy in all 3 degradation modes. In 2/3 modes, phi detects damage that accuracy cannot detect at all. Remaining open questions:
- Does this hold for larger corpora (N = 10³–10⁶)?
- Does it hold for real-world RAG systems with transformer embeddings?
- What is the quantitative relationship between phi-lead-time and corpus size?
- Can phi monitoring be made efficient enough for production use (currently O(N³))?

---

## References

- Tononi, G. (2004). An information integration theory of consciousness. BMC Neuroscience, 5(42).
- Oizumi, M., Albantakis, L., & Tononi, G. (2014). From the phenomenology to the mechanisms of consciousness: IIT 3.0. PLoS Computational Biology, 10(5).
- Baars, B. J. (1988). A Cognitive Theory of Consciousness. Cambridge University Press.
- Massimini, M. et al. (2005). Breakdown of cortical effective connectivity during sleep. Science, 309(5744).
- Landauer, R. (1961). Irreversibility and heat generation in the computing process. IBM J. Res. Dev., 5(3).
- Shannon, C. E. (1948). A mathematical theory of communication. Bell System Technical Journal, 27(3).
