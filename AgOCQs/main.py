import pandas as pd
import numpy as np
from typing import List, Optional
import logging
# Import the clustering modules
import sys
import transformers
if not hasattr(transformers, 'is_torch_npu_available'):
    transformers.is_torch_npu_available = lambda: False
import transformers.utils as utils
if not hasattr(utils, 'is_torch_npu_available'):
    utils.is_torch_npu_available = lambda: False

# Fix import_utils
import transformers.utils.import_utils as import_utils

# Add all potentially missing functions
missing_functions = {
    'is_torch_npu_available': lambda: False,
    'is_torch_mps_available': lambda: False,
    'is_torch_xpu_available': lambda: False,
    'is_torch_tpu_available': lambda: False,
    'is_nltk_available': lambda: False,
    'is_torch_neuroncore_available': lambda: False,
    'is_torch_fx_available': lambda: False,
}

for func_name, func in missing_functions.items():
    if not hasattr(import_utils, func_name):
        setattr(import_utils, func_name, func)
    if not hasattr(transformers, func_name):
        setattr(transformers, func_name, func)

if not hasattr(import_utils, 'NLTK_IMPORT_ERROR'):
    import_utils.NLTK_IMPORT_ERROR = "NLTK not available"
    print("✓ Added NLTK_IMPORT_ERROR")

if not hasattr(import_utils, 'is_nltk_available'):
    import_utils.is_nltk_available = lambda: False
    print("✓ Added is_nltk_available")
# Model configuration
from AgOCQs.cluster_optimizer import ClusterOptimizer
from typing import List, Optional
from AgOCQs.answerability import *
from AgOCQs.relevance import *
from AgOCQs.scope import *
from AgOCQs.combined_score import *
from AgOCQs.reliability import *
from AgOCQs.themes_with_kmeans import (
    KMeansTextClusterer,
    run_model_pipeline,
    post_process_results,
    model_pipeline
)
from AgOCQs.themes_with_kmeans import text_chunks, tokenizer
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model configuration
model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
max_length = 128
model = AutoModel.from_pretrained(model_id).to(device)
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load automatically generated CQs and their corpus
def get_paths(dataset: str) -> Path:
    paths = {
        "wastewater": (
            os.getcwd() + "/outputs/cleaned_text.csv",
            os.getcwd() + "/outputs/nonDuplicate.csv",
        ),
        "covid": (
            os.getcwd() + "/outputs/covid_data/cleaned_text.csv",
            os.getcwd() + "/outputs/covid_data/nonDuplicate.csv",
        ),
    }
    return paths[dataset]

def main():
    """
    Main function to demonstrate usage of the TextPreprocessor.
    """
    # --------------------------- RAW TEXT PROCESSING PIPELINE - only applicable with raw PDF or TXT data without AgOCQs------------------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["wastewater", "covid"], default="wastewater")
    args = parser.parse_args()
    base_corpus_path, base_cqs_path = get_paths(args.dataset)
    # covid data version
    #  root = Path.cwd()
    # base_corpus_path = os.getcwd() + "/outputs/covid_data/cleaned_text.csv"
    # base_cqs_path = os.getcwd() + "/outputs/covid_data/nonDuplicate.csv"
    df = pd.read_csv(base_corpus_path, sep =",") 
    df_cqs = pd.read_csv(base_cqs_path, sep ="|") 
    # *************************pre-process CQs and Corpus******************
     # process cqs from nonDuplicate.csv
    df_first = df_cqs[["questions"]]
    df_second = df_cqs[["similar_question"]]
    dup_df= df_first.drop_duplicates(keep='first')
    dup_sec_df= df_second.drop_duplicates(keep='first')
    dup_sec_df = dup_sec_df.rename(columns={"similar_question":"questions"})
    all_cqs= pd.concat([dup_df, dup_sec_df])
    print("lenght all_cqs",len(all_cqs))
    final_df= all_cqs.drop_duplicates(keep='first')
    print("lenght final_df",len(final_df))
    final_df.to_csv(os.getcwd() + "/outputs/covid_data/covid_data_metrics/cleaned_cqs.csv")
    print("lenght final_df",len(dup_sec_df))


    # *************************** THEME EXTRACTION WITH KMEANS CLUSTERING ************************** #
    cluster_df = text_chunks(df,'sentences_new')
    print(f"Generated {len(cluster_df)} chunks from document")
    # print("datatype", type(cluster_df))
    cluster_df.head()
    
    # --------------- optimization on chunks-----------------
    if len(cluster_df) > 10:
        cluster_optimizer = ClusterOptimizer()
        cluster_optimizer.set_k_range(range(2, 25))
        results = cluster_optimizer.evaluate_clusters(cluster_df, normalize_embeddings=True, n_stability_runs=20, verbose=True)
        
        # Display summary table
        cluster_optimizer.get_summary_table()
        
        # Plot dashboard
        fig, optimal_k = cluster_optimizer.plot_dashboard(
                figsize=(16, 12),
                save_path=os.getcwd() + "/outputs/covid_data/covid_data_metrics/plots/metric_plots.png"
                # save_path=os.getcwd() + "/outputs/plots/metric_plots.png"
            )
        labels, kmeans_model = cluster_optimizer.get_final_clusters(k=optimal_k)
        cluster_df['cluster_label'] = labels
        
    else:
        optimal_k = 3  # Default for small corpus
        
    print(f"alternative to Optimal number of Clusters(i.e themes)is a default value: {optimal_k}")
    print("Number of clusters identified",cluster_df['cluster_label'].unique())
    
    # ----------------------Save Clusters (to access for later) and read them in-------------------------------
    cluster_df.to_csv(os.getcwd() + "/outputs/covid_data/covid_data_metrics/model_output_kmeans.csv")

    clusterd_data = pd.read_csv(os.getcwd() + "/outputs/covid_data/covid_data_metrics/model_output_kmeans.csv")
    cqs_corpus=pd.read_csv(os.getcwd() + "/outputs/covid_data/covid_data_metrics/cleaned_cqs.csv")
    # waste water version
    # clusterd_data = pd.read_csv(os.getcwd() + "/outputs/model_output_kmeans.csv")
    # cqs_corpus=pd.read_csv(os.getcwd() + "/outputs/cleaned_cqs.csv")

    # ----------------Compute answerability scores-------------------
    print("----------------Compute answerability scores-------------------")
    cqs_corpus, df_clusters, answerability_score = cqs_answerability(clusterd_data, 
              model, cqs_corpus, threshold=0.85)
    print(f"Answerability Score: {answerability_score}")

    # ------------------Compute relevance scores--------------------
    print("----------------Compute relevance scores-------------------")
    corpus_entity_lst, corpus_relations_lst, cqs_entity_lst, cqs_relations_lst= cqs_relevance_data(
    df_clusters=df_clusters,
    cqs_corpus=cqs_corpus
    )
    
    (relevance_score, 
    relevance_summary, 
    matched_entities_dict, 
    matched_relations_dict, 
    relation_match_details_dict
    ) = compute_relevance_score( 
                corpus_entity_lst, 
                corpus_relations_lst, 
                cqs_entity_lst, 
                cqs_relations_lst,
                entity_threshold = 0.85,
                relation_threshold = 0.7,
                entity_weight = 0.5,
                relation_weight = 0.5, 
                use_semantic = True
                )
    print(f"Relevance Score: {relevance_score}")
    # ---------------------Compute scope scores--------------------
    print("----------------Compute scope scores-------------------")
    scope_score,percentage_cqs_covered, total_similarity_score, Zipf_exponent,cluster_scope_df = cqs_scope(
    df_clusters=df_clusters,
    cqs_corpus=cqs_corpus,
    threshold=0.50
    )
    print(f"Scope Score: {scope_score}")
    # ---------------------------Compute combined-score------------------------
    print("---------------------------Compute composite score------------------------")
    combined_score = combined_score(
        answerability_score,
        relevance_score,
        scope_score,
        answerability_weight=1.0,
        relevance_weight=1.0,
        scope_weight=1.0
    )
    print(f"Combined Score: {combined_score}")
    # -------------------Compute Reliability Test--------------------
    # Run reliability test
    tracker = run_reliability_test(
        cqs_corpus=cqs_corpus,
        df_clusters=df_clusters,
        n_runs=5,
        verbose=True
    )
    # Print summary
    tracker.print_summary()
    
    # Show the runs DataFrame
    print("\n--- Final runs_df ---")
    print(tracker.runs_df)
    
    # Plot results
    tracker.plot_runs(save_path=os.getcwd() + '/outputs/covid_data/covid_data_metrics/reliability/reliability_test_results.png')
    # wate water version
    # tracker.plot_runs(save_path=os.getcwd() + '/outputs/reliability/reliability_test_results.png')
    # Access the DataFrame directly
    runs_df = tracker.runs_df
    print(f"\nDataFrame shape: {runs_df.shape}")
    print(f"Columns: {runs_df.columns.tolist()}")
   

