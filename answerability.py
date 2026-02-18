import logging
from typing import Tuple, Union, List
# from transformers import pipeline
# from sentence_transformers import SentenceTransformer
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
# from sklearn.decomposition import PCA
import os 
import traceback
import pandas as pd
import spacy
import numpy as np
logging.basicConfig(level=logging.INFO)
nlp = spacy.load("en_core_web_sm")
from themes_with_kmeans import encode_with_automodel

def cqs_answerability(clusterd_data: pd.DataFrame, 
                  model, cqs_corpus: pd.DataFrame,
                  threshold: float = 0.85) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Calculate answerability score of a set of CQs with respect to given themes or clusters in the corpus.
    
    Args:
        clusterd_data (pd.DataFrame): DataFrame containing clustered sentences.
        model: SentenceTransformer model for encoding.
        cqs_corpus (pd.DataFrame): DataFrame containing competency questions (CQs).
        threshold (float): Similarity threshold for considering entity-relation pairs from sentences relevant.

    Returns:
        Tuple[List[str], List[float]]: DataFrame of relevant sentences, DataFrame of relevant CQs, and answerability score.
    """
    logging.info("Calculating answerability...")
    def extract_noun_verb_phrases(text):
        """Extract noun phrases and verb phrases from the given text."""
        doc = nlp(text)
        noun_phrases = [tok.text for tok in doc if tok.pos_ in ("NOUN", "PROPN")]
        verb_phrases = [tok.lemma_ for tok in doc if tok.pos_ == "VERB"]
        return f"{' '.join(noun_phrases)} - {' '.join(verb_phrases)}"
    
    # create a column inthe cqs_corpus dataframe for noun-phrases and verb-phrases. ecah row will have a list of noun-phrases and verb-phrases extracted from the questions column.
    cqs_corpus["extracted_phrases"] = cqs_corpus["questions"].apply(extract_noun_verb_phrases)
    # CQ embeddings
    cqs_corpus_questions = cqs_corpus["extracted_phrases"].fillna('').astype(str).tolist()
    cq_embeddings = encode_with_automodel(cqs_corpus_questions)
    cqs_corpus["cq_embeddings"] = cq_embeddings.tolist()
    # separate the clustered_data into different clusters based on the 'label' column into a dictionary of dataframes
    corpus = clusterd_data.to_dict(orient="records")
    df_clusters = {f'cluster_{label}': pd.DataFrame([record for record in corpus if record['cluster_label'] == label]) for label in clusterd_data['cluster_label'].unique()}
    # process each dataframe in corpus separately

    # Encode the corpus sentences and CQs
    list_of_cqs_dfs = []
    list_of_final_dfs = []
    theme_answerability_scores = 0
    all_unique_columns = None
    
    for cluster_key in df_clusters:
        df_clusters[cluster_key] = df_clusters[cluster_key].reset_index(drop=True)
    
        df_clusters[cluster_key]["extracted_phrases"] = df_clusters[cluster_key]["sentences_new"].apply(extract_noun_verb_phrases)
        corpus_extract = df_clusters[cluster_key]["extracted_phrases"].fillna('').astype(str).tolist()
    
        sentence_embeddings = encode_with_automodel(corpus_extract)
        df_clusters[cluster_key]['sentence_embeddings'] = sentence_embeddings.tolist()
    
        # make sure both are numpy arrays
        cq_emb = np.asarray(cq_embeddings)
        sent_emb = np.asarray(sentence_embeddings)
    
        cluster_label = int(df_clusters[cluster_key]["cluster_label"].iloc[0])
    
        try:
            S = cosine_similarity(cq_emb, sent_emb)  # (n_cq, n_sent)
    
            # Best CQ per sentence
            best_score_per_sent = S.max(axis=0)          # (n_sent,)
            best_cq_for_sent = S.argmax(axis=0)          # (n_sent,)
    
            relevant_sentence_idx = np.where(best_score_per_sent >= threshold)[0]
            relevant_sentence_idx2 = np.where(S >= threshold)[0]
            relevant_sentences = df_clusters[cluster_key].loc[relevant_sentence_idx, "sentences_new"].values
            relevant_phrases = df_clusters[cluster_key].loc[relevant_sentence_idx, "extracted_phrases"].values
            # ==============================
            relevant_sentences2 = df_clusters[cluster_key].loc[relevant_sentence_idx2, "sentences_new"].values
            relevant_phrases2 = df_clusters[cluster_key].loc[relevant_sentence_idx2, "extracted_phrases"].values
            # ===========================
            matched_cq_idx = best_cq_for_sent[relevant_sentence_idx2]
            matched_scores = best_score_per_sent[relevant_sentence_idx2]
    
            df = pd.DataFrame({
                "text": relevant_sentences2,
                "extracted_phrases": relevant_phrases2,
                "cluster": cluster_label,
                "matched_cq_index": matched_cq_idx,
                "similarity": matched_scores,
                "matched_cq_text": cqs_corpus.loc[matched_cq_idx, "questions"].values,
                "matched_cq_phrases": cqs_corpus.loc[matched_cq_idx, "extracted_phrases"].values,
            })
    
            # CQ-side summary (optional): best sentence per CQ
            best_score_per_cq = S.max(axis=1)     # (n_cq,)
            relevant_cq_idx = np.where(best_score_per_cq >= threshold)[0]
    
            df_cluster_cqs = pd.DataFrame({
                "text": cqs_corpus.loc[relevant_cq_idx, "questions"].values,
                "extracted_phrases": cqs_corpus.loc[relevant_cq_idx, "extracted_phrases"].values,
                "cluster": cluster_label,
                "best_similarity_in_cluster": best_score_per_cq[relevant_cq_idx],
                "cq_index": relevant_cq_idx,
            })
    
            logging.info(f"Cluster {cluster_label} has {len(df)} relevant sentences and {len(df_cluster_cqs)} relevant CQs.")
    
            if len(df) >= 1:
                theme_answerability_scores += 1
    
            list_of_final_dfs.append(df)
            list_of_cqs_dfs.append(df_cluster_cqs)
    
            all_unique_columns = sorted(set(df.columns) | set(df_cluster_cqs.columns))
    
        except Exception as e:
            logging.error(f"Error processing cluster {cluster_label}: {e}")
            logging.error(traceback.format_exc())
            continue

    
    # Concatenate all DataFrames into a single DataFrame
    if list_of_final_dfs != []:
        final_df = pd.concat(list_of_final_dfs, ignore_index=True)
        ans_cqs_df = pd.concat(list_of_cqs_dfs, ignore_index=True)
    # if list_of_cqs_dfs:
    else:
        final_df = pd.DataFrame(columns=all_unique_columns)
    # Calculate the answerability score
    num_themes = len(df_clusters)
    answerability_score = theme_answerability_scores / num_themes if num_themes > 0 else 0
    logging.info(f"Answerability score: {answerability_score}")
    # drop embedding columns before returning
    final_df = final_df.drop(columns=['sentence_embeddings'], errors='ignore')
    ans_cqs_df = ans_cqs_df.drop(columns=['cq_embeddings'], errors='ignore')
    df_clusters = {key: df.drop(columns=['sentence_embeddings'], errors='ignore') for key, df in df_clusters.items()}
    # cqs_corpus = cqs_corpus.drop(columns=['cq_embeddings'], errors='ignore')
    # final_df.to_csv(os.getcwd() + "/outputs/answerability_results/answerable_sentences.csv", index=False)
    # ans_cqs_df.to_csv(os.getcwd() + "/outputs/answerability_results/answerable_cqs.csv", index=False)
    # covid data version
    final_df.to_csv(os.getcwd() + "/outputs/covid_data/covid_data_metrics/answerability_results/answerable_sentences.csv", index=False)
    ans_cqs_df.to_csv(os.getcwd() + "/outputs/covid_data/covid_data_metrics/answerability_results/answerable_cqs.csv", index=False)
    
    return (cqs_corpus, df_clusters, answerability_score)
    