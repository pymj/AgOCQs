"""
Semantic Text Clustering using K-means
Replaces topic modeling with K-means clustering for more controlled semantic grouping
"""

import logging
import re
import numpy as np
from typing import Tuple, Union, List, Optional
# from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import pandas as pd
# Keep the existing imports that are still needed
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
TEST = False
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model configuration
model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
max_length = 128
model = AutoModel.from_pretrained(model_id).to(device)
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
# model.max_seq_length = 128
# tokenizer = model.tokenizer

# Updated model options for K-means
model_options = {
    'chunk_size': 'max_length', 
    'chunk_overlap': '256',
    'n_clusters': 5,  # Number of clusters for K-means
    'max_iter': 300,  # Maximum iterations for K-means
    'n_init': 10,  # Number of times K-means will be run with different centroid seeds
    'random_state': 42
}

# K-means specific parameters
kmeans_kwargs = {
    'n_clusters': 5,  # Default number of clusters
    'top_n_words': 10,  # Top words to extract per cluster
    'max_features': 5000,  # Max features for TF-IDF
    'min_df': 2,  # Minimum document frequency for TF-IDF
    'max_df': 0.95  # Maximum document frequency for TF-IDF
}


class KMeansTextClusterer:
    """
    K-means clustering for semantic text grouping
    """
    
    def __init__(self, n_clusters: int, embedding_model: model, 
                 random_state: int = 42):
        """
        Initialize K-means clusterer with embedding model
        
        Args:
            n_clusters: Number of clusters
            embedding_model: SentenceTransformer model for embeddings
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.embedding_model = embedding_model 
        self.random_state = random_state
        self.kmeans = None
        self.embeddings = None
        self.tfidf_vectorizer = None
        self.cluster_keywords = {}
        
    def fit_transform(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit K-means on text embeddings and return cluster assignments
        
        Args:
            texts: List of text documents
            
        Returns:
            clusters: Array of cluster assignments
            distances: Array of distances to cluster centers
        """
        print(" text datatype in fit-transform", type(texts))
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Perform K-means clustering
        logger.info(f"Performing K-means clustering with {self.n_clusters} clusters...")
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        
        clusters = self.kmeans.fit_predict(self.embeddings)
        
        # Calculate distances to cluster centers (can be used as confidence scores)
        distances = self.kmeans.transform(self.embeddings)
        min_distances = np.min(distances, axis=1)
        
        # Convert distances to probabilities (inverse distance weighting)
        # Normalize so closer points have higher probability
        max_dist = np.max(min_distances)
        probabilities = 1 - (min_distances / max_dist)
        
        # Extract keywords for each cluster
        self._extract_cluster_keywords(texts, clusters)
        
        return clusters, probabilities
    
    def _extract_cluster_keywords(self, texts: List[str], clusters: np.ndarray, 
                                 top_n: int = 10):
        """
        Extract representative keywords for each cluster using TF-IDF
        
        Args:
            texts: Original texts
            clusters: Cluster assignments
            top_n: Number of top keywords to extract per cluster
        """
        # Fit TF-IDF vectorizer
        print("datatype in _extract_cluster_keywords", type(texts))
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=kmeans_kwargs['max_features'],
            min_df=kmeans_kwargs['min_df'],
            max_df=kmeans_kwargs['max_df'],
            stop_words='english'
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # Extract keywords for each cluster
        for cluster_id in range(self.n_clusters):
            # Get texts belonging to this cluster
            cluster_mask = clusters == cluster_id
            
            if not np.any(cluster_mask):
                self.cluster_keywords[cluster_id] = []
                continue
            
            # Calculate mean TF-IDF scores for this cluster
            cluster_tfidf = tfidf_matrix[cluster_mask].mean(axis=0).A1
            
            # Get top keywords
            top_indices = cluster_tfidf.argsort()[-top_n:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            
            self.cluster_keywords[cluster_id] = keywords
    
    def get_cluster_info(self) -> pd.DataFrame:
        """
        Get information about clusters
        
        Returns:
            DataFrame with cluster information
        """
        cluster_info = []
        for cluster_id, keywords in self.cluster_keywords.items():
            cluster_info.append({
                'Cluster': cluster_id,
                'Keywords': ', '.join(keywords[:10]),  # Top 10 keywords
                'Top_Keywords_List': keywords[:10]
            })
        
        return pd.DataFrame(cluster_info)
    
    def calculate_silhouette_score(self) -> float:
        """
        Calculate silhouette score for cluster quality evaluation
        
        Returns:
            Silhouette score
        """
        if self.embeddings is not None and self.kmeans is not None:
            labels = self.kmeans.labels_
            score = silhouette_score(self.embeddings, labels)
            return score
        return None


def find_optimal_clusters(texts: List[str], min_k: int = 2, max_k: int = 10) -> dict:
    """
    Find optimal number of clusters using elbow method and silhouette score
    
    Args:
        texts: List of text documents
        min_k: Minimum number of clusters to try
        max_k: Maximum number of clusters to try
        embedding_model: SentenceTransformer model
        
    Returns:
        Dictionary with evaluation metrics for each k
    """
    print("datatype in find_optimal_clusters", type(texts))
    if len(texts) < 2:
        print(f"Error: Only {len(texts)} text(s) provided. Need at least 2 for clustering.")
        return None
    embedding_model = model 
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    
    results = {
        'k': [],
        'inertia': [],
        'silhouette_score': []
    }
    max_k = min(max_k, embeddings.shape[0] - 1)  # Can't have more clusters than samples
    if min_k > max_k:
        min_k = 2
        max_k = embeddings.shape[0] - 1
    
    print(f"Testing k from {min_k} to {max_k}")
    silhouette_scores = []
    for k in range(min_k, max_k + 1):
        if k >= embeddings.shape[0]:
            break
            
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        if len(set(labels)) > 1:  # Only calculate silhouette if we have 2+ clusters
            score = silhouette_score(embeddings, labels)
            silhouette_scores.append(score)
            results['silhouette_score'].append(score)
        else:
            silhouette_scores.append(0)
            results['silhouette_score'].append(0)
    # for k in range(min_k, max_k + 1):
    #     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    #     labels = kmeans.fit_predict(embeddings)
        
        results['k'].append(k)
        results['inertia'].append(kmeans.inertia_)
        
    #     if k > 1:  # Silhouette score needs at least 2 clusters
    #         sil_score = silhouette_score(embeddings, labels)
    #         results['silhouette_score'].append(sil_score)
    #     else:
    #         results['silhouette_score'].append(0)
    cluster_scores = pd.DataFrame(silhouette_scores)
    return pd.DataFrame(results), cluster_scores

def encode_with_automodel(texts, batch_size=32):
    """Manual encoding with AutoModel"""
    model.eval()
    all_embeddings = []
    # should be a list of strings
    print("datatype in encode_with_automodel:", type(texts))
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**encoded)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)

def text_chunks(df, text_column):
    # model.max_seq_length = max_length
    texts = df[text_column].fillna('').astype(str).tolist()
    # embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = encode_with_automodel(texts)
    df['embeddings'] = embeddings.tolist()
    print(f"Generated {len(embeddings)} embeddings with shape {embeddings[0].shape}")
    
    return df


def data_chunks(record, tokeniser, sentence_count=None):#might not need this function
    """
    Process data into chunks for clustering
    
    Args:
        record: Input data records
        tokeniser: Tokenizer to use
        sentence_count: Optional sentence count parameter
    
    Returns:
        DataFrame with chunked data
    """
    # Convert to DataFrame if needed
    if not isinstance(record, pd.DataFrame):
        records = pd.DataFrame(record)
    else:
        records = record.copy()
    
    if 'text_col' not in records.columns:
        records = records.rename(columns={records.columns[0]: "text_col"})
    
    text = records['text_col']
    
    # Generate chunks
    chunks = text_chunks(text, text_column)
    n_chunks = len(chunks)
    
    logger.info(f"Generated {n_chunks} chunks from input data")
    
    # Create DataFrame from chunks
    chunked_df = pd.DataFrame(chunks, columns=['text_col'])
    
    return chunked_df


def model_pipeline(cleaned_data, tokeniser, model_options) -> Tuple[pd.DataFrame, dict]:
    """
    Main pipeline for K-means clustering
    
    Args:
        cleaned_data: Cleaned input data
        tokeniser: Tokenizer to use
        model_options: Model configuration options
    
    Returns:
        Tuple of (output DataFrame, cluster metadata)
    """
    # Create clusterer
    n_clusters = model_options.get('n_clusters', 5)
    clusterer = KMeansTextClusterer(
        n_clusters=n_clusters,
        embedding_model=model
    )
    
    # Chunk the data
    chunked_data = data_chunks(cleaned_data, tokeniser)
    
    # Perform clustering
    def _cluster_inference(textFile):
        """Run K-means clustering on text data"""
        lst_chunked_data = textFile['text_col'].astype(str).tolist()
        
        # Remove empty strings
        lst_chunked_data = [text for text in lst_chunked_data if text.strip()]
        
        logger.info(f"Clustering {len(lst_chunked_data)} text chunks...")
        clusters, probabilities = clusterer.fit_transform(lst_chunked_data)
        
        cluster_info = clusterer.get_cluster_info()
        
        # Calculate clustering quality
        silhouette = clusterer.calculate_silhouette_score()
        logger.info(f"Clustering silhouette score: {silhouette:.3f}")
        
        return clusters, probabilities, cluster_info
    
    # Run clustering
    clusters, probs, cluster_meta = _cluster_inference(chunked_data)
    
    # Process results
    def _postprocess_clusters(cluster_list, metadata):
        """Post-process clustering results"""
        cluster_metadata = {
            'clusters': cluster_list,
            'metadata': metadata,
            'n_clusters': n_clusters
        }
        return cluster_list, cluster_metadata
    
    processed_clusters, processed_meta = _postprocess_clusters(clusters, cluster_meta)
    
    # Ensure proper data types and lengths
    processed_clusters = [int(cluster) for cluster in processed_clusters]
    probs = [float(prob) for prob in probs]
    
    # Filter chunked_data to match the length of processed clusters
    # (in case empty chunks were removed)
    chunked_data_filtered = chunked_data[chunked_data['text_col'].str.strip() != '']
    
    # Verify lengths match
    if len(processed_clusters) != len(chunked_data_filtered):
        raise ValueError(f"Length mismatch: {len(processed_clusters)} clusters vs {len(chunked_data_filtered)} chunks")
    
    # Create output DataFrame
    cluster_output_df = pd.DataFrame({
        'text_col': chunked_data_filtered['text_col'].values,
        'cluster_id': processed_clusters,
        'cluster_confidence': probs
    })
    
    return cluster_output_df, processed_meta


def run_model_pipeline(textFile: pd.DataFrame, model_options: dict) -> Tuple[pd.DataFrame, dict]:
    """
    Run the complete K-means clustering pipeline
    
    Args:
        textFile: Input DataFrame with text data
        model_options: Configuration options including n_clusters
    
    Returns:
        Tuple of (clustered data DataFrame, cluster metadata)
    """
    if TEST:
        textFile = textFile.iloc[:1000]
    
    return model_pipeline(textFile, tokenizer, model_options)


def post_process_results(model_output: pd.DataFrame, metadata: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Post-process clustering results for final output
    
    Args:
        model_output: DataFrame with clustered text
        metadata: Dictionary with cluster metadata
    
    Returns:
        Tuple of (processed output DataFrame, processed metadata DataFrame)
    """
    # Extract metadata DataFrame
    if 'metadata' in metadata:
        metadatadf = metadata['metadata']
    else:
        metadatadf = pd.DataFrame(metadata)
    
    # Ensure proper column names
    if 'cluster_id' in model_output.columns:
        model_output = model_output.rename(columns={'cluster_id': 'topic_id'})
    if 'cluster_confidence' in model_output.columns:
        model_output = model_output.rename(columns={'cluster_confidence': 'topic_probability'})
    
    # Process keywords if needed
    if 'Keywords' in metadatadf.columns:
        metadatadf['Keywords_List'] = metadatadf['Keywords'].apply(
            lambda x: x.split(', ') if isinstance(x, str) else []
        )
    
    # Save outputs
    model_output.to_csv("CQsMetrics/outputs/model_output_kmeans.csv", sep="|", index=False)
    metadatadf.to_csv("CQsMetrics/outputs/metadata_kmeans.csv", sep="|", index=False)

    logger.info(f"Processed {len(model_output)} text chunks into {metadata.get('n_clusters', 'unknown')} clusters")
    
    return model_output, metadatadf


# Utility function for determining optimal number of clusters
def suggest_n_clusters(texts: List[str], max_k: int = 15) -> int:
    """
    Suggest optimal number of clusters based on silhouette score
    
    Args:
        texts: List of texts to cluster
        max_k: Maximum k to evaluate
    
    Returns:
        Suggested number of clusters
    """
    results, cluster_scores = find_optimal_clusters(texts, min_k=2, max_k=min(max_k, len(texts) - 1))
    
    # Find k with highest silhouette score
    best_k = results.loc[results['silhouette_score'].idxmax(), 'k']
    
    logger.info(f"Suggested number of clusters: {int(best_k)} (based on silhouette score)")
    
    return int(best_k)
