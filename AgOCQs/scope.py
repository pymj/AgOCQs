import logging
from typing import Tuple, Dict, List, Optional, Counter, Union
from sklearn.metrics.pairwise import cosine_similarity 
import traceback
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import amrlib
import os
logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt
amrlib.setup_spacy_extension()
import os
import spacy
from collections import defaultdict
import ast
import matplotlib.pyplot as plt
from collections import Counter
from AgOCQs.relevance import entity_extraction
# save_scope_path = os.getcwd() + "/outputs/plots/scope_zipfian_distribution.png"
save_scope_path = os.getcwd() + "/outputs/covid_data/covid_data_metrics/plots/scope_zipfian_distribution.png"
nlp = spacy.load("en_core_web_sm")
def convert_to_2d_embeddings(col: pd.Series) -> np.ndarray:
    """
    Convert a pandas Series of embeddings (list/np.array/str-repr) into a 2D numpy array.
    """
    embs: List[np.ndarray] = []
    for x in col.tolist():
        if x is None or (isinstance(x, float) and pd.isna(x)):
            continue
        if isinstance(x, str):
            x = ast.literal_eval(x)
        arr = np.asarray(x, dtype=float).ravel()
        embs.append(arr)

    if len(embs) == 0:
        return np.empty((0, 0), dtype=float)

    # Ensure consistent dimensionality
    dim0 = embs[0].shape[0]
    for i, e in enumerate(embs):
        if e.shape[0] != dim0:
            raise ValueError(f"Embedding dimension mismatch at row {i}: {e.shape[0]} vs {dim0}")

    return np.vstack(embs)


def cqs_scope(
    cqs_corpus: pd.DataFrame,
    df_clusters: Dict[str, pd.DataFrame],
    threshold: float = 0.50
) -> Tuple[float, float, float, float, pd.DataFrame]:
    
    logging.info("Calculating CQ scope coverage...")
    # os.makedirs(output_dir, exist_ok=True)

    # --- Prepare CQ embeddings ---
    cq_texts = cqs_corpus["questions"].fillna("").astype(str).tolist()
    total_cqs = len(cq_texts)
    logging.info(f"Total CQs to evaluate: {total_cqs}")

    if "cq_embeddings" not in cqs_corpus.columns:
        raise KeyError("cqs_corpus has no CQ vectors column for processing.")
    # convert to 2d embenning sinve data was stored as csv previously
    cq_embeddings = convert_to_2d_embeddings(cqs_corpus["cq_embeddings"])
    if cq_embeddings.size == 0:
        # create an empty dataframe to return should there no values
        cluster_scope_df = pd.DataFrame(columns=["cq", "cluster_sentence", "coverage_score", "cluster", "doc_index"])
        return 0.0, 0.0, 0.0, 0.0, cluster_scope_df

    cq_embeddings_norm = normalize(cq_embeddings, norm="l2")

    # --- Storage variable---
    match_rows = []
    cq_best_matches: Dict[int, dict] = {}    
    cluster_cohesion_scores: Dict[str, Dict[int, float]] = {}

    # --- Process each cluster Lexical coverage. Comparing how each CQ matches u in each of the clusters ---
    for cluster_name, cluster_df in df_clusters.items():
        logging.info(f"Processing cluster: {cluster_name}")

        if cluster_df is None or cluster_df.empty:
            logging.warning(f"Cluster {cluster_name} is empty. Skipping.")
            continue

        # Get embeddings column (handle both naming conventions)
        if "embeddings" in cluster_df.columns:
            emb_col = "embeddings"
        elif "cluster_embeddings" in cluster_df.columns:
            emb_col = "cluster_embeddings"
        else:
            raise KeyError(f"Cluster {cluster_name} column not in dataframe.")

        text_col = "sentences_new" if "sentences_new" in cluster_df.columns else ("text" if "text" in cluster_df.columns else None)
        cluster_texts = cluster_df[text_col].fillna("").astype(str).tolist() if text_col else [""] * len(cluster_df)

        # convert to 2d embedding since data was stored as csv previously
        cluster_embeddings = convert_to_2d_embeddings(cluster_df[emb_col])
        if cluster_embeddings.size == 0:
            logging.warning(f"No valid embeddings in cluster {cluster_name}. Skipping.")
            continue

        corpus_embeddings_norm = normalize(cluster_embeddings, norm="l2")

        # --- Check-cluster cohesion ---
        label_col = None
        if "cluster_labels" in cluster_df.columns:
            label_col = "cluster_labels"
        elif "cluster_label" in cluster_df.columns:
            label_col = "cluster_label"

        if label_col is not None:
            cohesion = _calculate_cluster_cohesion(corpus_embeddings_norm, cluster_df[label_col].values)
            cluster_cohesion_scores[cluster_name] = cohesion

        # --- CQ to corpus coverage ---
        sim_matrix = cosine_similarity(cq_embeddings_norm, corpus_embeddings_norm)

        for cq_idx, cq_text in enumerate(cq_texts):
            cq_sims = sim_matrix[cq_idx]

            match_indices = np.where(cq_sims >= threshold)[0]
            for doc_idx in match_indices:
                sim_score = float(cq_sims[doc_idx])
                match_rows.append({
                    "cq": cq_text,
                    "cq_index": int(cq_idx),
                    "cluster_sentence": cluster_texts[int(doc_idx)],
                    "coverage_score": sim_score,
                    "cluster": cluster_name,
                    "doc_index": int(doc_idx),
                })

            # Best match for this CQ within this cluster
            if match_indices.size > 0:
                best_idx = match_indices[np.argmax(cq_sims[match_indices])]
                best_sim = float(cq_sims[best_idx])

                # Track best match across all clusters
                if cq_idx not in cq_best_matches or best_sim > cq_best_matches[cq_idx]["score"]:
                    cq_best_matches[cq_idx] = {
                        "score": best_sim,
                        "cluster": cluster_name,
                        "matched_text": cluster_texts[int(best_idx)],
                        "cq_text": cq_text,
                    }

    cluster_scope_df = pd.DataFrame(match_rows)

    # --- Aggregate metrics ---
    covered_cq_indices = set(cq_best_matches.keys())
    not_covered_cq_indices = set(range(total_cqs)) - covered_cq_indices

    percentage_cqs_covered = (len(covered_cq_indices) / total_cqs) * 100 if total_cqs > 0 else 0.0

    logging.info(f"CQs covered: {len(covered_cq_indices)}/{total_cqs} ({percentage_cqs_covered:.1f}%)")
    if not_covered_cq_indices:
        preview = [cq_texts[i] for i in list(not_covered_cq_indices)[:5]]
        logging.warning(f"CQs NOT covered ({len(not_covered_cq_indices)}): {preview}...")

    # Mean coverage similarity (all matches)
    mean_coverage_similarity = float(np.mean(cluster_scope_df["coverage_score"])) if not cluster_scope_df.empty else 0.0

    # Mean best-match similarity (one per CQ)
    mean_best_match_sim = float(np.mean([v["score"] for v in cq_best_matches.values()])) if cq_best_matches else 0.0

    logging.info(f"Mean coverage similarity (all matches): {mean_coverage_similarity:.3f}")
    logging.info(f"Mean best-match similarity (per CQ): {mean_best_match_sim:.3f}")

    # Zipf exponent (only if entity_matched exists)
    zipf_exponent = _calculate_zipf_exponent(cluster_scope_df, save_scope_path=None)

    # Composite scope score
    coverage_component = percentage_cqs_covered / 100
    similarity_component = mean_best_match_sim

    if zipf_exponent[0] > 0:
        print("distribution is greater then 0")
        zipf_component = min(1.0, 1.0 / zipf_exponent[0])
        scope_score = (coverage_component + similarity_component + zipf_component) / 3
    else:
        scope_score = (coverage_component + similarity_component) / 2

    logging.info(f"Scope score: {scope_score:.3f}")

    coverage_summary(
        total_cqs=total_cqs,
        cq_texts=cq_texts,
        covered_cq_indices=covered_cq_indices,
        not_covered_cq_indices=not_covered_cq_indices,
        cq_best_matches=cq_best_matches,
        cluster_cohesion_scores=cluster_cohesion_scores,
        threshold=threshold,
        scope_score=scope_score
    )

    return scope_score, percentage_cqs_covered, mean_best_match_sim, zipf_exponent, cluster_scope_df


def _calculate_cluster_cohesion(embeddings_norm: np.ndarray, labels: np.ndarray) -> Dict[int, float]:
    cohesion: Dict[int, float] = {}
    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        cluster_embs = embeddings_norm[mask]

        if len(cluster_embs) > 1:
            centroid = cluster_embs.mean(axis=0, keepdims=True)
            centroid = normalize(centroid, norm="l2")
            sims = cosine_similarity(cluster_embs, centroid).flatten()
            cohesion[int(label)] = float(np.mean(sims))
        else:
            cohesion[int(label)] = 1.0

    return cohesion

def entity_extraction(text_or_doc):
    """Extract entities safely -> returns List[str]"""
    doc = text_or_doc if hasattr(text_or_doc, "ents") else nlp(str(text_or_doc))

    named_entities = defaultdict(set)
    try:
        for ent in doc.ents:
            if ent.text and len(ent.text.strip()) > 1:
                named_entities[ent.label_].add(ent.text.strip())
    except Exception:
        pass

    noun_phrases = set()
    try:
        for chunk in doc.noun_chunks:
            t = chunk.text.strip()
            if len(t) > 2:
                noun_phrases.add(t)
    except Exception:
        try:
            for tok in doc:
                if tok.pos_ in ("NOUN", "PROPN"):
                    t = tok.text.strip()
                    if len(t) > 2:
                        noun_phrases.add(t)
        except Exception:
            pass

    if noun_phrases:
        named_entities["NOUN_PHRASES"] = noun_phrases

    # flatten all unique entities
    # unique_entities = set()
    # for s in named_entities.values():
    #     unique_entities.update(s)

    return named_entities
def _calculate_zipf_exponent(cluster_scope_df: pd.DataFrame, save_scope_path: None) -> float:
    zipf_exponent = 0.0

    if cluster_scope_df.empty:
        return zipf_exponent
    freq_ent = cluster_scope_df["cluster_sentence"].fillna("").astype(str).apply(entity_extraction)
    # counts = Counter(freq_ent)
    from itertools import chain

    counts = Counter(chain.from_iterable(d.keys() for d in freq_ent))
    freqs = np.array(sorted(counts.values(), reverse=True), dtype=float)
    freqs = freqs[freqs > 0]

    if len(freqs) < 2:
        logging.warning("Not enough data points for Zipf analysis (need >=2)")
        return 0.0

    ranks = np.arange(1, len(freqs) + 1, dtype=float)
    x = np.log(ranks)
    y = np.log(freqs)

    b, a = np.polyfit(x, y, 1)
    zipf_exponent = float(-b)
    C = float(np.exp(a))
    figsize=(8, 6)
    logging.info(f"Zipf Exponent: {zipf_exponent:.3f}, Constant: {C:.3f}")
    fig = plt.figure(figsize=figsize)
    plt.figure(figsize=(8, 6))
    plt.loglog(ranks, freqs, "b.", markersize=8, label="Observed")
    plt.loglog(ranks, C * ranks ** (-zipf_exponent), "r-", linewidth=2, label=f"Zipf fit (α={zipf_exponent:.2f})")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title("Zipfian Distribution of Matched Entities")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_scope_path = os.getcwd() + "/outputs/covid_data/covid_data_metrics/plots/scope_zipfian_distribution.png"
    # save_scope_path = os.getcwd() + "/outputs/plots/scope_zipfian_distribution.png"
    plt.savefig(save_scope_path, dpi=150, bbox_inches="tight", facecolor="white")
    logging.info(f"Zipf plot saved to: {save_scope_path}")
    plt.show()

    return zipf_exponent, fig


def coverage_summary(
    total_cqs: int,
    cq_texts: List[str],
    covered_cq_indices: set,
    not_covered_cq_indices: set,
    cq_best_matches: dict,
    cluster_cohesion_scores: dict,
    threshold: float,
    scope_score: float
):
    print("\n" + "=" * 70)
    print("CQ COVERAGE SUMMARY")
    print("=" * 70)
    print(f"Threshold: {threshold}")
    print(f"Total CQs: {total_cqs}")

    if total_cqs > 0:
        print(f"Covered:   {len(covered_cq_indices)} ({len(covered_cq_indices) / total_cqs * 100:.1f}%)")
    else:
        print(f"Covered:   0 (0.0%)")

    print(f"Uncovered: {len(not_covered_cq_indices)}")

    if cq_best_matches:
        scores = [v["score"] for v in cq_best_matches.values()]
        print("\nBest-match similarity stats (covered CQs):")
        print(f"  Mean:   {np.mean(scores):.3f}")
        print(f"  Median: {np.median(scores):.3f}")
        print(f"  Min:    {np.min(scores):.3f}")
        print(f"  Max:    {np.max(scores):.3f}")

    if cluster_cohesion_scores:
        print("\nCluster cohesion scores:")
        for cluster_name, cohesion in cluster_cohesion_scores.items():
            mean_coh = float(np.mean(list(cohesion.values()))) if cohesion else 0.0
            print(f"  {cluster_name}: {mean_coh:.3f}")

    if not_covered_cq_indices:
        # print("\nUncovered CQs:")
        for idx in list(not_covered_cq_indices)[:5]:
            cq = cq_texts[idx]
            # print(f"  • {cq[:80]}...")

    print(f"\n★ SCOPE SCORE: {scope_score:.3f}")
    print("=" * 70)


def analyze_coverage_at_thresholds(
    cqs_corpus: pd.DataFrame,
    df_clusters: Dict[str, pd.DataFrame],
    thresholds: List[float] = [0.4, 0.5, 0.6, 0.7, 0.8]
) -> pd.DataFrame:
    results = []
    for thresh in thresholds:
        scope, pct_covered, mean_sim, zipf, _ = cqs_scope(cqs_corpus, df_clusters, threshold=thresh)
        results.append({
            "threshold": thresh,
            "scope_score": scope,
            "pct_covered": pct_covered,
            "mean_similarity": mean_sim,
            "zipf_exponent": zipf
        })

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("COVERAGE vs THRESHOLD ANALYSIS")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("=" * 70)

    return results_df
