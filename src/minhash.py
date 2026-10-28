"""Vectorized MinHash + LSH near-duplicate detection (numpy only).

Documents are sets of word 5-grams ("shingles"). Two documents' MinHash
signatures agree in a fraction of positions that estimates the Jaccard
similarity of their shingle sets. LSH banding proposes candidate pairs,
which are then verified against the full signature.
"""
import re

import numpy as np
import pandas as pd

TOKEN_RE = re.compile(r"[a-z0-9]+")
SHINGLE = 5
_MULT = np.uint64(0x9E3779B97F4A7C15)  # odd constant for rolling shingle hash


def _word_hashes(texts):
    """Concatenated uint64 word hashes for all texts, plus per-text lengths."""
    words, lengths = [], []
    for t in texts:
        w = TOKEN_RE.findall(t.lower())
        words.extend(w)
        lengths.append(len(w))
    hashes = pd.util.hash_array(np.array(words, dtype=object)) if words else np.zeros(0, np.uint64)
    return hashes.astype(np.uint64), np.asarray(lengths, dtype=np.int64)


def _shingle_hashes(texts):
    """Hash of every word 5-gram, grouped by document (offsets into the array)."""
    w, lengths = _word_hashes(texts)
    doc = np.repeat(np.arange(len(lengths)), lengths)
    n = len(w) - SHINGLE + 1
    if n <= 0:
        raise ValueError("texts too short for shingling")
    with np.errstate(over="ignore"):
        h = np.zeros(n, dtype=np.uint64)
        for k in range(SHINGLE):
            h = h * _MULT + w[k : k + n]
    valid = doc[:n] == doc[SHINGLE - 1 :]
    h, owner = h[valid], doc[:n][valid]
    # A document shorter than 5 words gets one shingle: the hash of all its words.
    missing = np.setdiff1d(np.arange(len(lengths)), owner)
    if len(missing):
        fallback = pd.util.hash_array(np.array([" ".join(TOKEN_RE.findall(texts[i].lower())) for i in missing], dtype=object))
        h = np.concatenate([h, fallback.astype(np.uint64)])
        owner = np.concatenate([owner, missing])
        order = np.argsort(owner, kind="stable")
        h, owner = h[order], owner[order]
    starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
    return h, starts


def signatures(texts, n_perm=128, seed=0, chunk=20_000):
    """MinHash signatures, shape (len(texts), n_perm), dtype uint32."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 2**63, size=n_perm, dtype=np.uint64) | np.uint64(1)
    b = rng.integers(0, 2**63, size=n_perm, dtype=np.uint64)
    out = np.empty((len(texts), n_perm), dtype=np.uint32)
    for lo in range(0, len(texts), chunk):
        h, starts = _shingle_hashes(texts[lo : lo + chunk])
        with np.errstate(over="ignore"):
            for k in range(n_perm):
                hk = ((h * a[k] + b[k]) >> np.uint64(32)).astype(np.uint32)
                out[lo : lo + len(starts), k] = np.minimum.reduceat(hk, starts)
    return out


def candidate_pairs(sigs, n_bands, max_bucket=50):
    """Pairs (i, j), i < j, that share at least one LSH band."""
    rows = sigs.shape[1] // n_bands
    rng = np.random.default_rng(1)
    mix = rng.integers(1, 2**63, size=rows, dtype=np.uint64) | np.uint64(1)
    pairs = []
    for band in range(n_bands):
        block = sigs[:, band * rows : (band + 1) * rows].astype(np.uint64)
        with np.errstate(over="ignore"):
            key = (block * mix).sum(axis=1)
        order = np.argsort(key, kind="stable")
        k = key[order]
        bounds = np.flatnonzero(np.r_[True, k[1:] != k[:-1], True])
        multi = np.flatnonzero(np.diff(bounds) >= 2)  # only buckets with 2+ documents
        for s, e in zip(bounds[multi], bounds[multi + 1]):
            members = np.sort(order[s:e])
            if e - s <= max_bucket:
                ii, jj = np.triu_indices(len(members), 1)
                pairs.append(np.stack([members[ii], members[jj]], 1))
            else:  # very large bucket: link to the first member and to neighbours
                first = np.stack([np.full(len(members) - 1, members[0]), members[1:]], 1)
                chain = np.stack([members[:-1], members[1:]], 1)
                pairs.append(np.concatenate([first, chain]))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(np.concatenate(pairs), axis=0)


def similarity(sigs, pairs):
    """Estimated Jaccard similarity for each pair."""
    if len(pairs) == 0:
        return np.zeros(0)
    return (sigs[pairs[:, 0]] == sigs[pairs[:, 1]]).mean(axis=1)


def components(n, pairs):
    """Connected components (union-find); returns a root index for every node."""
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(n)])
