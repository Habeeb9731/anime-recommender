"""Content features: synopsis embeddings plus genres, studio, type and release year."""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def embed_synopses(synopses: list[str | None], model_name: str, batch_size: int = 64) -> np.ndarray:
    """L2-normalised sentence embeddings. Titles without a synopsis get a zero vector."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    dim = model.get_sentence_embedding_dimension()
    out = np.zeros((len(synopses), dim), dtype=np.float32)
    idx = [i for i, s in enumerate(synopses) if s]
    if idx:
        out[idx] = model.encode(
            [synopses[i] for i in idx],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    return out


def _multi_hot(rows: list[list[str]], vocab: list[str]) -> np.ndarray:
    pos = {v: i for i, v in enumerate(vocab)}
    m = np.zeros((len(rows), len(vocab)), dtype=np.float32)
    for r, vals in enumerate(rows):
        for v in vals:
            if v in pos:
                m[r, pos[v]] = 1.0
    return m


def _unit_rows(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n < 1e-9] = 1.0
    return m / n


def _split(col: list[str | None]) -> list[list[str]]:
    return [[x for x in (s or "").split("|") if x] for s in col]


def structured_blocks(items: pl.DataFrame, top_studios: int, year_bucket: int) -> dict[str, np.ndarray]:
    """One-hot style blocks, each row unit-normalised. Vocabularies use metadata only."""
    genres = _split(items["genres"].to_list())
    studios = _split(items["studios"].to_list())
    freq: dict[str, int] = {}
    for row in studios:
        for s in row:
            freq[s] = freq.get(s, 0) + 1
    top = sorted(freq, key=lambda s: (-freq[s], s))[:top_studios]
    genre_vocab = sorted({g for row in genres for g in row})
    type_vocab = sorted({t for t in items["type"].to_list() if t})
    buckets = [[] if y is None else [str(int(y) // year_bucket * year_bucket)] for y in items["year"].to_list()]
    year_vocab = sorted({b for row in buckets for b in row})
    return {
        "genre": _unit_rows(_multi_hot(genres, genre_vocab)),
        "studio": _unit_rows(_multi_hot(studios, top)),
        "type": _unit_rows(_multi_hot([[t] if t else [] for t in items["type"].to_list()], type_vocab)),
        "year": _unit_rows(_multi_hot(buckets, year_vocab)),
    }


def build_item_vectors(
    text: np.ndarray, blocks: dict[str, np.ndarray], weights: dict[str, Any]
) -> np.ndarray:
    """Weighted concatenation of feature blocks, L2-normalised per title."""
    parts = [float(weights["text"]) * text.astype(np.float32)]
    parts += [float(weights[name]) * blocks[name] for name in ("genre", "studio", "type", "year")]
    return _unit_rows(np.hstack(parts)).astype(np.float32)
