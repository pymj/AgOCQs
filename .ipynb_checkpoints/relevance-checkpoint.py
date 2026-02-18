import traceback
import pandas as pd
import numpy as np
import logging
import re
import os
logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt
import logging
from typing import Tuple, List, Set
from collections import Counter
from difflib import SequenceMatcher
import math 
import logging
import spacy
import logging, math
import numpy as np
import pandas as pd
from themes_with_kmeans import encode_with_automodel
from typing import Tuple, Dict
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import torch
from spacy.tokens import Doc, Span
from transformers import AutoTokenizer, AutoModel
nlp = spacy.load("en_core_web_sm")
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model configuration
model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
max_length = 128
model = AutoModel.from_pretrained(model_id).to(device)
def cqs_relevance_data(
    cqs_corpus: pd.DataFrame,
    df_clusters: dict,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, float]:

    logging.info("Calculating relevance...")

    if not df_clusters:
        return (pd.DataFrame(columns=["entities"]), pd.DataFrame(columns=["relations"]), 0.0)

    df_flatclusters = pd.concat(df_clusters.values(), ignore_index=True)

    # expects a spaCy nlp object available in scope
    if "nlp" not in globals():
        raise ValueError("spaCy `nlp` pipeline is not defined in scope.")
    # --- Per-row extraction (CORRECT: apply per cell, not over whole column each time) ---
    corpus_entities_per_row = df_flatclusters["sentences_new"].fillna("").astype(str).apply(entity_extraction)
    corpus_relations_per_row = df_flatclusters["sentences_new"].fillna("").astype(str).apply(lambda x: relation_extraction(x, nlp))

    cqs_entities_per_row = cqs_corpus["questions"].fillna("").astype(str).apply(entity_extraction)
    cqs_relations_per_row = cqs_corpus["questions"].fillna("").astype(str).apply(lambda x: relation_extraction(x, nlp))

    # --- Flatten ---
    corpus_entity_lst = [e for row in corpus_entities_per_row for e in row]
    corpus_relations_lst = [r for row in corpus_relations_per_row for r in row]
    cqs_entity_lst = [e for row in cqs_entities_per_row for e in row]
    cqs_relations_lst = [r for row in cqs_relations_per_row for r in row]

    # --- Frequency tables ---
    ents_by_freq = Counter(corpus_entity_lst)
    rels_by_freq = Counter(corpus_relations_lst)

    ent_df = (
        pd.DataFrame(ents_by_freq.items(), columns=["entities", "frequency"])
        .sort_values(by="frequency", ascending=False)
        .reset_index(drop=True)
    )
    rel_df = (
        pd.DataFrame(rels_by_freq.items(), columns=["relation", "frequency"])
        .sort_values(by="frequency", ascending=False)
        .reset_index(drop=True)
    )
    # save dataframe to file
    # print("rel_df",len(rel_df), rel_df.head(20))
    # print("ent_df",len(ent_df), ent_df.head(20))
    return corpus_entity_lst, corpus_relations_lst, cqs_entity_lst, cqs_relations_lst

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
    unique_entities = set()
    for s in named_entities.values():
        unique_entities.update(s)

    return sorted(unique_entities)


def relation_extraction(text_or_doc, nlp):
    """Extract verb lemmas/text from a spaCy Doc/Span or raw text -> List[str]"""
    if isinstance(text_or_doc, (Doc, Span)):
        doc = text_or_doc
    else:
        doc = nlp(str(text_or_doc))

    rels = set()
    for tok in doc:
        if tok.pos_ == "VERB":  # or tok.pos_ in {"VERB", "AUX"} if you want auxiliaries too
            rel = (tok.lemma_ or tok.text).strip()
            if rel:
                rels.add(rel)
    return sorted(rels)
    
def normalize_relation(relation: str) -> str:
    """
    Normalize relation string for comparison.
    Handles: underscores, camelCase, extra spaces, case differences.
    """
    # Convert camelCase to spaces: "partOf" -> "part Of" -> "part of"
    relation = re.sub(r'([a-z])([A-Z])', r'\1 \2', relation)
    
    # Replace underscores and hyphens with spaces
    relation = relation.replace('_', ' ').replace('-', ' ')
    
    # Lowercase
    relation = relation.lower()
    
    # Remove extra whitespace
    relation = ' '.join(relation.split())
    
    # Remove common filler words that don't change meaning
    # fillers = ['is', 'was', 'are', 'were', 'has', 'have', 'had', 'the', 'a', 'an']
    words = relation.split()
    # words = [w for w in words if w not in fillers]
    final_rel = ' '.join(words)
    # print("normalized", final_rel)
    return ' '.join(words)


def match_normalizer(
    corpus_relations: List[str],
    cq_relations: Set[str],
    threshold: float = 0.8
    ) -> Tuple[List[str], float]:
    """
    Match relations using normalized string comparison.
    
    Returns:
        matched_relations: list of matched corpus relations
        match_ratio: percentage of corpus relations matched
    """
    matched_relations = []
    
    # Normalize all relations or entities
    corpus_normalized = {r: normalize_relation(r) for r in corpus_relations}
    cq_normalized = {r: normalize_relation(r) for r in cq_relations}
    
    for corpus_rel, corpus_norm in corpus_normalized.items():
        for cq_rel, cq_norm in cq_normalized.items():
            # Exact match after normalization
            if corpus_norm == cq_norm:
                matched_relations.append(corpus_rel)
                break
            
            # Fuzzy match on normalized strings (still use SequenceMatcher but on clean strings)
            if SequenceMatcher(None, corpus_norm, cq_norm).ratio() >= threshold:
                matched_relations.append(corpus_rel)
                break
    
    matched_relations = list(set(matched_relations))
    unique_ent = list(set(corpus_relations))
    match_ratio = len(matched_relations) / len(unique_ent) if unique_ent else 0.0
    
    return matched_relations, match_ratio


def semantic_rel_match(
    corpus_relations: List[str],
    cq_relations: List[str],
    threshold: float = 0.6
    ) -> Tuple[List[str], float, Dict]:
    """
    Match relations using embedding similarity.
    
    Parameters:
    -----------
    corpus_relations : List[str]
        Relations from the corpus
    cq_relations : List[str]
        Relations from CQs
    threshold : float
        Cosine similarity threshold (lower than string matching!)
    model : optional
        Sentence embedding model (e.g., SentenceTransformer)
        
    Returns:
    --------
    matched_relations : List[str]
    match_ratio : float
    details : Dict with similarity scores
    """
    if not corpus_relations or not cq_relations:
        return [], 0.0, {}
    
    corpus_list = list(set(corpus_relations))
    cq_list = list(set(cq_relations))
    # Use provided embedding model
    corpus_emb = encode_with_automodel(corpus_list)
    cq_emb = encode_with_automodel(cq_list)
    # # *********
    # if not corpus_emb :
    #     # Alternative model: TF-IDF on normalized strings (fit on ALL together)
    #     from sklearn.feature_extraction.text import TfidfVectorizer
        
    #     corpus_normalized = [normalize_relation(r) for r in corpus_list]
    #     cq_normalized = [normalize_relation(r) for r in cq_list]
        
    #     # Fit vectorizer on combined data
    #     all_relations = corpus_normalized + cq_normalized
    #     vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
        
    #     try:
    #         all_embeddings = vectorizer.fit_transform(all_relations).toarray()
    #         corpus_emb = all_embeddings[:len(corpus_list)]
    #         cq_emb = all_embeddings[len(corpus_list):]
    #     except ValueError:
    #         # Fallback for edge cases
    #         return [], 0.0, {}
    
    # Normalize for cosine similarity
    corpus_emb_norm = normalize(corpus_emb, norm='l2')
    cq_emb_norm = normalize(cq_emb, norm='l2')
    
    # Compute similarity matrix
    sim_matrix = cosine_similarity(corpus_emb_norm, cq_emb_norm)
    
    # Find matches
    matched_relations = []
    match_details = {}
    
    for i, corpus_rel in enumerate(corpus_list):
        best_match_idx = np.argmax(sim_matrix[i])
        best_score = sim_matrix[i, best_match_idx]
        
        if best_score >= threshold:
            matched_relations.append(corpus_rel)
            match_details[corpus_rel] = {
                'matched_to': cq_list[best_match_idx],
                'similarity': float(best_score)
            }
    
    match_ratio = len(matched_relations) / len(corpus_list) if corpus_list else 0.0
    
    return matched_relations, match_ratio, match_details


def hybrid_match(
    corpus_relations: List[str],
    cq_relations: Set[str],
    string_threshold: float = 0.75,
    semantic_threshold: float = 0.6
    ) -> Tuple[List[str], float, Dict]:
    """
    Hybrid matching: try exact/normalized first, fall back to semantic.
    
    This gives you the speed of string matching for obvious matches
    and the accuracy of semantic matching for tricky cases.
    """
    corpus_list = list(set(corpus_relations))
    cq_list = list(cq_relations)
    
    matched_relations = []
    match_details = {}
    unmatched_corpus = []
    
    # Normalize all
    corpus_normalized = {r: normalize_relation(r) for r in corpus_list}
    cq_normalized = {r: normalize_relation(r) for r in cq_list}
    cq_norm_to_orig = {normalize_relation(r): r for r in cq_list}
    
    # Pass 1: Exact match after normalization
    for corpus_rel in corpus_list:
        corpus_norm = corpus_normalized[corpus_rel]
        
        if corpus_norm in cq_normalized.values():
            matched_relations.append(corpus_rel)
            # Find original CQ relation
            orig_cq = cq_norm_to_orig.get(corpus_norm, corpus_norm)
            match_details[corpus_rel] = {
                'matched_to': orig_cq,
                'method': 'exact_normalized',
                'similarity': 1.0
            }
        else:
            unmatched_corpus.append(corpus_rel)
    
    # Pass 2: Sequence Matching of string on remaining
    still_unmatched = []
    for corpus_rel in unmatched_corpus:
        corpus_norm = corpus_normalized[corpus_rel]
        best_match = None
        best_score = 0
        
        for cq_rel in cq_list:
            cq_norm = cq_normalized[cq_rel]
            score = SequenceMatcher(None, corpus_norm, cq_norm).ratio()
            if score > best_score:
                best_score = score
                best_match = cq_rel
        
        if best_score >= string_threshold:
            matched_relations.append(corpus_rel)
            match_details[corpus_rel] = {
                'matched_to': best_match,
                'method': 'fuzzy_string',
                'similarity': best_score
            }
        else:
            still_unmatched.append(corpus_rel)
    
    # Pass 3: Semantic matching on remaining (if any)
    if still_unmatched:
        sem_matched, _, sem_details = semantic_rel_match(
            still_unmatched, cq_list, 
            threshold=semantic_threshold
        )
        matched_relations.extend(sem_matched)
        for rel, details in sem_details.items():
            details['method'] = 'semantic'
            match_details[rel] = details
    # elif still_unmatched:
    #     # Use TF-IDF based semantic as alternative
    #     sem_matched, _, sem_details = semantic_rel_match(
    #         still_unmatched, cq_list,
    #         threshold=semantic_threshold
    #     )
    #     matched_relations.extend(sem_matched)
    #     for rel, details in sem_details.items():
    #         details['method'] = 'semantic_tfidf'
    #         match_details[rel] = details
    
    matched_relations = list(set(matched_relations))
    match_ratio = len(matched_relations) / len(corpus_list) if corpus_list else 0.0
    
    return matched_relations, match_ratio, match_details

    
def compute_relevance_score(
    corpus_entities: List[str],
    corpus_relations: List[str],
    cq_entities: Set[str],
    cq_relations: Set[str],
    entity_threshold: float = 0.75,
    relation_threshold: float = 0.7,
    entity_weight: float = 0.5,
    relation_weight: float = 0.5,
    use_semantic: bool = True
    ) -> Tuple[float,Dict, Dict, Dict, Dict]:
    """
    Compute relevance score with improved matching.
    
    Returns dict with overall score and component breakdowns.
    """
    # Match entities (usually proper nouns, less variation)
    matched_entities, entity_ratio = match_normalizer(
        corpus_entities, cq_entities, threshold=entity_threshold
    )
    unique_ent = list(set(corpus_entities))
    print(f"Matched entities without semantics: {len(matched_entities)}/{len(unique_ent)} ({entity_ratio*100:.1f}%)")
    # Match relations (more variation, benefit from semantic matching)
    if use_semantic:
        matched_relations, relation_ratio, relation_details = hybrid_match(
            corpus_relations, cq_relations,
            string_threshold=relation_threshold,
            semantic_threshold=0.6
        )
        unique_rel = list(set(corpus_relations))
        print(f"Matched relations with semantics: {len(matched_relations)}/{len(unique_rel)} ({relation_ratio*100:.1f}%)")
    else:
        matched_relations, relation_ratio = match_normalizer(
            corpus_relations, cq_relations, threshold=relation_threshold
        )
        unique_rel = list(set(corpus_relations))
        relation_details = {}
        print(f"Matched relations without semantics: {len(matched_relations)}/{len(unique_rel)} ({relation_ratio*100:.1f}%)")
    
    # Weighted relevance score
    relevance_score = (entity_ratio) + (relation_ratio)
    # relevance_score = (entity_ratio * entity_weight) + (relation_ratio * relation_weight)
    relevance_summary = {'relevance_score': relevance_score,
        'entity_match_ratio': entity_ratio,
        'relation_match_ratio': relation_ratio,
        'total_matched_entities': len(matched_entities),
        'total_entities': len(set(corpus_entities)),
        'total_matched_relations': len(matched_relations),
        'total_relations': len(set(corpus_relations))
        }
    relevance_summary_df=  pd.DataFrame({
        'relevance_score': [relevance_score],
        'entity_match_ratio': [entity_ratio],
        'relation_match_ratio': [relation_ratio],
        'total_matched_entities': [len(matched_entities)],
        'total_entities': [len(set(corpus_entities))],
        'total_matched_relations': [len(matched_relations)],
        'total_relations': [len(set(corpus_relations))]
        })
    relevance_summary_df.to_csv(os.getcwd() + "/outputs/relevance_results/relevance_summary.csv")
    # entities
    matched_entities_dict = {'matched_entities': matched_entities}
    matched_entities_df= pd.DataFrame({'matched_entities': [matched_entities]})
    matched_entities_df.to_csv(os.getcwd() + "/outputs/relevance_results/matched_entities.csv")
    # relations
    matched_relations_dict = {'matched_relations': matched_relations}
    matched_relations_df= pd.DataFrame({'matched_relations': [matched_relations]})
    matched_relations_df.to_csv(os.getcwd() + "/outputs/relevance_results/matched_relations.csv")
    relation_match_details_dict = {'relation_match_details': relation_details}
    return relevance_score, relevance_summary, matched_entities_dict, matched_relations_dict, relation_match_details_dict
       
