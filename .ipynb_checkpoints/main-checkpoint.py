
from answerability import *
from scope import *
from relevance import *
from combined_score import *
from text_preprocessing import *
from cluster_optimizer import ClusterOptimizer
import pandas as pd
import numpy as np
from typing import List, Optional
import logging
# Import the clustering modules
from themes_with_kmeans import (
    KMeansTextClusterer,
    run_model_pipeline,
    post_process_results,
    model_pipeline
)
from themes_with_kmeans import text_chunks, tokenizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to demonstrate usage of the TextPreprocessor.
    """
    # --------------------------- RAW TEXT PROCESSING PIPELINE - only applicable with raw data inputs------------------------- #
    # processinputs = ProcessInputData()
    # file_paths=processinputs.get_files(base_path)
    # outFile= processinputs.process_rawfile(file_paths)
    # outText = processinputs.readTextFile(outFile)
    # corpus = processinputs.format(outText)
    cqs_file = processinputs.get_files(base_path)
    cqs_outFile = processinputs.process_rawfile(cqs_file)
    clean_CQs = processinputs.readTextFile(cqs_outFile)
    cqs_data = processinputs.clean_questions(clean_CQs)
    
    # --------------------------- TEXT PREPROCESSING PIPELINE ------------------------- #
    # preprocessor = TextPreprocessor()
    # # Define output file paths
    # output_file = "/CQsMetrics/data/txt_files/input_cleaned.txt"
    
    # # Check if input file exists
    # if isinstance(corpus, str):
    #     print(f"Processing file corpus")
        
    #     # Process with all cleaning options enabled
    #     cleaned_text = preprocessor.process_file(
    #         corpus,
    #         output_file,
    #         remove_numbers=True,
    #         remove_headers=True,
    #         normalize_spaces=True,
    #         fix_breaks=True,
    #         clean_chars=True,
    #         merge_sentences=False  # Set to True if you want to merge broken sentences
    #     )
        
    #     # Show sample of cleaned text
    #     print("\n" + "="*50)
    #     print("SAMPLE OF CLEANED TEXT (first 1000 characters):")
    #     print("="*50)
    #     print(cleaned_text[:100])
    #     print("...")
        
    #     # Extract and show sections
    #     sections = preprocessor.extract_sections(cleaned_text)
    #     print("\n" + "="*50)
    #     print(f"EXTRACTED SECTIONS ({len(sections)} sections found):")
    #     print("="*50)
    #     for i, section_title in enumerate(list(sections.keys())[:10], 1):
    #         print(f"{i}. {section_title}")
        
    #     # Show statistics
    #     # original_lines = open(input_file, 'r', encoding='utf-8', errors='ignore').readlines()
    #     cleaned_lines = cleaned_text.split('\n')
    #     print("\n" + "="*50)
    #     print("PREPROCESSING STATISTICS:")
    #     print("="*50)
    #     print(f"Original lines: {len(corpus.splitlines())}")
    #     print(f"Cleaned lines: {len(cleaned_lines)}")
    #     print(f"Lines removed: {len(corpus.splitlines()) - len(cleaned_lines)}")
    #     print(f"Original characters: {sum(len(line) for line in corpus.splitlines())}")
    #     print(f"Cleaned characters: {len(cleaned_text)}")
        
    # else:
    #     print(f"Input file not found: input_file")
    #     print("\nUsage example:")
    #     print("preprocessor = TextPreprocessor()")
    #     print("cleaned_text = preprocessor.process_file('input.txt', 'output.txt')")

    # *************************** THEME MODELING PIPELINE ************************** #
    base_corpus_path = os.getcwd() + "/outputs/cleaned_text.csv"
    base_cqs_path = os.getcwd() + "/outputs/semanticCQs.csv"
    # input corpus and CQs
    df = pd.read_csv(base_corpus_path, sep =",") 
    df_cqs = pd.read_csv(base_cqs_path, sep ="|") 
    df_first = df_cqs[["firstCQs"]]
    df_second = df_cqs[["secondCQs"]]
    dup_df= df_first.drop_duplicates(keep='first')
    dup_sec_df= df_second.drop_duplicates(keep='first')
    dup_sec_df = dup_sec_df.rename(columns={"secondCQs":"firstCQs"})
    all_cqs= pd.concat([dup_df, dup_sec_df])
    print("lenght all_cqs",len(all_cqs))
    final_df= all_cqs.drop_duplicates(keep='first')
    print("lenght final_df",len(final_df))
    # First, find optimal number of clusters
    print("\n Finding optimal number of clusters...")
    
    chunks = text_chunks(final_df, 'firstCQs')
    print(f"Generated {len(chunks)} chunks from document")
    
    # --------------- optimization on chunks-----------------
    if len(chunks) > 10:
        sample_chunks = chunks[:50]  # Use sample for optimization
        optimizer = ClusterOptimizer()
        results = optimizer.evaluate_clusters(sample_chunks, min_k=2, max_k=6)
        labels, kmeans_model = cluster_optimizer.get_final_clusters()
        cluster_df['cluster'] = labels
        optimal_k = cluster_optimizer.best_k
        All_results= cluster_optimizer.results
        print(f"Optimal number of Clusters(i.e themes): {optimal_k}")
    else:
        optimal_k = 3  # Default for small corpus

        print(f"Optimal number of Clusters(i.e themes): {optimal_k}")

    # Process with optimal settings
    model_options = {
        'n_clusters': optimal_k,
        'chunk_size': 512,
        'chunk_overlap': 256,
        'random_state': 42
    }
    
    # ----------------------Run Cluster pipeline-------------------------------
    output_df, metadata = run_model_pipeline(df, model_options)
    final_output, final_metadata = post_process_results(output_df, metadata)
    print("\nProcessing Results:")
    print("-"*30)
    print(f"Generated chunks: {len(final_output)}")
    print(f"Number of themes: {optimal_k}")

    # Show theme distribution
    print("\n Themes Distribution:")
    print("-"*30)
    cluster_counts = final_output['topic_id'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        percentage = (count / len(final_output)) * 100
        print(f"Theme {cluster_id}: {count} chunks ({percentage:.1f}%)")
    
    print("\n Metrics computation results")
    # ----------------Compute answerability scores-------------------
    clusterd_data = pd.read_csv("CQsMetrics/outputs/model_output_kmeans.csv")
    cqs_corpus=pd.read_csv("CQsMetrics/outputs/cleaned_cqs.csv")
    cqs_corpus, df_clusters, answerability_score = cqs_answerability(clusterd_data, 
                  sentence_model, cqs_corpus, threshold=0.85)
    print(f"Answerability Score: {answerability_score}")
    # ------------------Compute relevance scores--------------------
    relevance_score, matched_entities_extracted, matched_relations_extracted = cqs_relevance(
        df_clusters=df_clusters,
        cqs_corpus=cqs_corpus,
        threshold=0.85
    )
    print(f"Relevance Score: {relevance_score}")
    # ---------------------Compute scope scores--------------------
    scope_score,percentage_cqs_covered, total_similarity_score, Zipf_exponent = cqs_scope(
        df_clusters=df_clusters,
        cqs_corpus=cqs_corpus,
        matched_entities_extracted=matched_entities_extracted,
        matched_relations_extracted=matched_relations_extracted,
        threshold=0.85
    )
    print(f"Scope Score: {scope_score}")
    # ---------------------------Compute composite score------------------------
    composite = composite_score(
        answerability=answerability_score,
        relevance=relevance_score,
        scope=scope_score,
        answerability_weight=1.0,
        relevance_weight=1.0,
        scope_weight=1.0
    )
    print(f"Composite Score: {composite}")
   

