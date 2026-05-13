"""
Genesis: 0 + self-reference + vacuum fluctuation + atomic stability.

Physically pure:
  - Stability = wave amplitude (intrinsic, like atomic mass)
  - Selection = decay_unstable (radioactive decay of weak tokens, not whole entries)
  - Generation = particle walk using WAVE vectors (stable atoms preferred naturally)
  - No external state. The system IS its own memory.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from janet import (
    Sorter, Vec, DIMS, DEGREES, MAX_EMBED_VAL,
    normalize, dot, add, reject, particle,
    _FNV_OFF, _FNV_PRIME,
)


def vacuum_fluctuation(sorter: Sorter, generation: int) -> Vec:
    """Fluctuation entangled with global state."""
    h = _FNV_OFF
    for text, _ in sorter.entries:
        for c in text.encode():
            h = ((h ^ c) * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    for c in str(generation).encode():
        h = ((h ^ c) * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    rng = np.random.default_rng(h)
    amp = sorter.max_embed // 4
    return normalize(rng.integers(-amp, amp, size=sorter.dims, dtype=np.int64).astype(np.int32))


def pair_produce(sorter: Sorter, generation: int) -> None:
    """Create new tokens from vacuum."""
    h = _FNV_OFF
    for c in f"pp{generation}".encode():
        h = ((h ^ c) * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    for text, _ in sorter.entries:
        for c in text[:8].encode():
            h = ((h ^ c) * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    for i in range(2):
        h = ((h ^ i) * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
        length = 3 + int(h % 4)
        chars = []
        for j in range(length):
            h = ((h ^ j) * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
            chars.append("bcdfghjklmnprstvwz"[int(h % 18)] if j % 2 == 0
                         else "aeiou"[int(h % 5)])
            h = (h * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
        tok = "".join(chars)
        if tok not in sorter.particles:
            sorter.particles[tok] = particle(tok)


def stability(sorter: Sorter, tok: str) -> int:
    """Atomic stability = self-amplitude of wave vector. Intrinsic property."""
    w = sorter.waves.get(tok)
    if w is None:
        return 0
    return int(np.dot(w.astype(np.int64), w.astype(np.int64)))


def decay_unstable(sorter: Sorter) -> str | None:
    """Radioactive decay: remove least stable token from weakest entry."""
    if len(sorter.entries) < 6 or sorter._matrix is None:
        return None

    M = sorter._matrix.astype(np.int64)
    gram = M @ M.T
    np.fill_diagonal(gram, np.iinfo(np.int64).max)
    weakest_idx = int(gram.min(axis=1).argmin())

    text = sorter.entries[weakest_idx][0]
    tokens = text.split()
    if len(tokens) <= 3:
        return None

    # Find least stable token
    stabilities = [(tok, stability(sorter, tok)) for tok in tokens]
    stabilities.sort(key=lambda x: x[1])
    weakest_tok = stabilities[0][0]

    # Decay: remove unstable atom
    new_tokens = [t for t in tokens if t != weakest_tok]
    new_text = " ".join(new_tokens)
    sorter.entries[weakest_idx] = (new_text, sorter.encode(new_text))
    sorter._rebuild_matrix()
    return f"-{weakest_tok}"


def genesis(sorter: Sorter, generation: int) -> str | None:
    """One genesis event. Uses wave vectors — stable atoms naturally preferred."""
    if sorter._matrix is None or not sorter.entries:
        return None

    M = sorter._matrix.astype(np.int64)
    centroid = normalize((M.sum(axis=0) // len(sorter.entries)).astype(np.int32))
    fluct = vacuum_fluctuation(sorter, generation)
    novel = reject(normalize(add(centroid, fluct)), centroid)
    if int(np.max(np.abs(novel))) == 0:
        return None

    pair_produce(sorter, generation)

    # Walk using WAVE vectors: stable tokens have stronger waves → win argmax naturally
    words = list(sorter.particles.keys())
    wvecs = np.stack([
        sorter.waves[w] if w in sorter.waves else sorter.particles[w]
        for w in words
    ], axis=0)

    state = novel.copy()
    result = []
    used = set()
    for _ in range(8):
        amps = wvecs.astype(np.int64) @ state.astype(np.int64)
        best = int(np.argmax(amps))
        if amps[best] <= 0 or words[best] in used:
            break
        result.append(words[best])
        used.add(words[best])
        state = reject(state, wvecs[best])
        if int(np.max(np.abs(state))) == 0:
            break

    if len(result) < 3:
        return None
    text = " ".join(result)

    # Coherence gate
    text_vec = sorter.encode(text)
    if len(sorter.entries) >= 4:
        amps_check = M @ text_vec.astype(np.int64)
        max_amp = int(amps_check.max())
        if max_amp <= 0:
            return None
        self_amp = int(np.dot(text_vec.astype(np.int64), text_vec.astype(np.int64)))
        if self_amp > 0 and max_amp > self_amp * (DEGREES - 1) // DEGREES:
            return None

    sorter.learn(text)
    return text


def run(seed: str, generations: int = 50):
    print(f"SEED: \"{seed}\"")
    print(f"{'Gen':>4s} | {'N':>3s} | {'Phi':>8s} | {'H':>6s} | {'Event'}")
    print(f"{'-'*4}-+-{'-'*3}-+-{'-'*8}-+-{'-'*6}-+-{'-'*50}")

    s = Sorter()
    s.learn_batch([seed])
    s.seed_fixed_point()

    phi = s.phi()
    H = s.corpus_entropy()
    print(f"{'0':>4s} | {len(s.entries):>3d} | {phi:>8.4f} | {H:>6.3f} | seed")

    for gen in range(1, generations + 1):
        result = genesis(s, gen)

        # Radioactive decay instead of heal — gentler, preserves structure
        decayed = None
        if len(s.entries) >= 8 and gen % 3 == 0:
            decayed = decay_unstable(s)

        phi = s.phi()
        H = s.corpus_entropy()

        event = ""
        if result:
            event = result[:42]
            if len(result) > 42:
                event += "..."
        else:
            event = "(didn't condense)"
        if decayed:
            event += f" [{decayed}]"

        print(f"{gen:>4d} | {len(s.entries):>3d} | {phi:>8.4f} | {H:>6.3f} | {event}")

    print()
    print(f"Final: {len(s.entries)} entries, phi={phi:.4f}, H={H:.3f}")
    # Show stable vocabulary (top tokens by wave amplitude)
    top = sorted(s.waves.keys(), key=lambda t: stability(s, t), reverse=True)[:15]
    print(f"Stable atoms: {', '.join(top)}")
    print()


if __name__ == "__main__":
    print()
    print("=" * 70)
    print("GENESIS — atomic stability model")
    print("0 + self-reference + vacuum fluctuation + radioactive decay")
    print("Stability is intrinsic (wave amplitude). No external state.")
    print("=" * 70)
    print()
    run("information is physical", 50)
    print("=" * 70)
    run("the observer and the observed are one", 50)
