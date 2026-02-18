"""
Performance Reliability Testing Framework

Runs evaluation metrics multiple times and collects results in a DataFrame
to assess stability and reliability of the scoring system.
"""
import os
import logging
import time
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from AgOCQs.combined_score import *
from AgOCQs.scope import *
from AgOCQs.relevance import *
from AgOCQs.answerability import *
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
from AgOCQs.themes_with_kmeans import encode_with_automodel
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
clusterd_data = pd.read_csv(os.getcwd() + "/outputs/covid_data/covid_data_metrics/model_output_kmeans.csv")
cqs_corpus=pd.read_csv(os.getcwd() + "/outputs/covid_data/covid_data_metrics/cleaned_cqs.csv")
# clusterd_data = pd.read_csv(os.getcwd() + "/outputs/model_output_kmeans.csv")
# cqs_corpus=pd.read_csv(os.getcwd() + "/outputs/cleaned_cqs.csv")
@dataclass
class PerformanceTracker:
    """
    Track performance metrics across multiple runs for reliability testing.
    """
    n_runs: int = 5
    runs_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    run_details: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        self._initialize_dataframe()
    
    def _initialize_dataframe(self):
        """Initialize empty runs DataFrame with expected columns."""
        self.runs_df = pd.DataFrame({
            "run_id": pd.Series(dtype='int'),
            "relevance": pd.Series(dtype='float'),
            "scope": pd.Series(dtype='float'),
            "answerability": pd.Series(dtype='float'),
            "coverage_pct": pd.Series(dtype='float'),
            "combined_score": pd.Series(dtype='float'),
            "runtime_seconds": pd.Series(dtype='float')
        })
    
    def add_run(
        self,
        run_id: int,
        relevance: float,
        scope: float,
        answerability: float,
        coverage_pct: float = 0.0,
        combined_score: float = 0.0,
        runtime_seconds: float = 0.0,
        details: Optional[Dict] = None
    ):
        """Add a single run's results to the tracker."""
        new_row = pd.DataFrame([{
            "run_id": run_id,
            "relevance": relevance,
            "scope": scope,
            "answerability": answerability,
            "coverage_pct": coverage_pct,
            "combined_score": combined_score,
            "runtime_seconds": runtime_seconds
        }])
        
        self.runs_df = pd.concat([self.runs_df, new_row], ignore_index=True)
        
        if details:
            details['run_id'] = run_id
            self.run_details.append(details)
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """Calculate summary statistics across all runs."""
        if self.runs_df.empty:
            return pd.DataFrame()
        
        numeric_cols = ['relevance', 'scope', 'answerability', 'coverage_pct', 
                        'combined_score', 'runtime_seconds']
        
        stats = []
        for col in numeric_cols:
            if col in self.runs_df.columns:
                values = self.runs_df[col].dropna()
                if len(values) > 0:
                    stats.append({
                        'metric': col,
                        'mean': values.mean(),
                        'std': values.std(),
                        'min': values.min(),
                        'max': values.max(),
                        'cv': (values.std() / values.mean() * 100) if values.mean() != 0 else 0,  # Coefficient of variation
                        'range': values.max() - values.min()
                    })
        
        return pd.DataFrame(stats)
    
    def print_summary(self):
        """Print formatted summary of all runs."""
        print("\n" + "="*80)
        print("PERFORMANCE RELIABILITY SUMMARY")
        print("="*80)
        
        print(f"\nTotal runs: {len(self.runs_df)}")
        
        print("\n--- All Runs ---")
        print(self.runs_df.to_string(index=False))
        
        print("\n--- Summary Statistics ---")
        stats_df = self.get_summary_statistics()
        if not stats_df.empty:
            print(stats_df.to_string(index=False))
        
        # Reliability assessment
        print("\n--- Reliability Assessment ---")
        for metric in ['relevance', 'scope', 'answerability']:
            if metric in self.runs_df.columns:
                values = self.runs_df[metric].dropna()
                if len(values) > 1:
                    cv = (values.std() / values.mean() * 100) if values.mean() != 0 else 0
                    stability = "HIGH" if cv < 5 else "MEDIUM" if cv < 15 else "LOW"
                    print(f"  {metric}: CV={cv:.2f}% → Stability: {stability}")
        
        print("="*80)
    
    def plot_runs(self, save_path: Optional[str] = None):
        """Visualize metrics across runs."""
        if self.runs_df.empty:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics = ['relevance', 'scope', 'answerability']
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        # Plot 1: Line plot of main metrics
        ax1 = axes[0, 0]
        for metric, color in zip(metrics, colors):
            if metric in self.runs_df.columns:
                ax1.plot(self.runs_df['run_id'], self.runs_df[metric], 
                        'o-', label=metric, color=color, linewidth=2, markersize=8)
        ax1.set_xlabel('Run ID')
        ax1.set_ylabel('Score')
        ax1.set_title('Metrics Across Runs')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])
        
        # Plot 2: Box plot of main metrics
        ax2 = axes[0, 1]
        data_to_plot = [self.runs_df[m].dropna().values for m in metrics if m in self.runs_df.columns]
        bp = ax2.boxplot(data_to_plot, labels=metrics, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax2.set_ylabel('Score')
        ax2.set_title('Score Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Coverage percentage
        ax3 = axes[1, 0]
        if 'coverage_pct' in self.runs_df.columns:
            ax3.bar(self.runs_df['run_id'], self.runs_df['coverage_pct'], 
                   color='#9b59b6', alpha=0.7)
            ax3.axhline(y=self.runs_df['coverage_pct'].mean(), color='red', 
                       linestyle='--', label=f"Mean: {self.runs_df['coverage_pct'].mean():.1f}%")
        ax3.set_xlabel('Run ID')
        ax3.set_ylabel('Coverage %')
        ax3.set_title('Coverage Percentage per Run')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Runtime
        ax4 = axes[1, 1]
        if 'runtime_seconds' in self.runs_df.columns:
            ax4.bar(self.runs_df['run_id'], self.runs_df['runtime_seconds'],
                   color='#f39c12', alpha=0.7)
            ax4.axhline(y=self.runs_df['runtime_seconds'].mean(), color='red',
                       linestyle='--', label=f"Mean: {self.runs_df['runtime_seconds'].mean():.2f}s")
        ax4.set_xlabel('Run ID')
        ax4.set_ylabel('Seconds')
        ax4.set_title('Runtime per Run')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
        return fig


def run_reliability_test(
    cqs_corpus: pd.DataFrame,
    df_clusters: Dict[str, pd.DataFrame],
    n_runs: int = 5,
    # combined_score: Optional[Callable] = None,
    # cqs_answerability: Optional[Callable] = None,
    # cqs_scope: Optional[Callable] = None,
    # cqs_relevance_data: Optional[Callable] = None,
    # compute_relevance_score: Optional[Callable] = None,
    verbose: bool = True
) -> PerformanceTracker:
    """
    Run multiple evaluation iterations to test reliability.
    
    Parameters:
    -----------
    cqs_corpus : pd.DataFrame
        CQ corpus with 'questions' and 'cq_embeddings' columns
    df_clusters : Dict[str, pd.DataFrame]
        Clustered corpus
    n_runs : int
        Number of runs to perform

    verbose : bool
        Print progress
        
    Returns:
    --------
    PerformanceTracker with all run results
    """
    random_seed =42
    np.random.seed(random_seed)
    tracker = PerformanceTracker(n_runs=n_runs)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"RELIABILITY TEST: {n_runs} runs")
       
        print(f"{'='*60}")
    
    for run_id in range(n_runs):
        start_time = time.time()
        
        if verbose:
            print(f"\n--- Run {run_id + 1}/{n_runs} ---")
         # answerability_score
        cqs_corpus, df_clusters, answerability_score = cqs_answerability(clusterd_data, 
              model, cqs_corpus, threshold=0.85)
        #  # compute_relevance_score
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
        
        # Compute scope (main metric)
        scope_score,percentage_cqs_covered, total_similarity_score, Zipf_exponent,cluster_scope_df = cqs_scope(
            df_clusters=df_clusters,
            cqs_corpus=cqs_corpus,
            threshold=0.50
            )
        coverage_pct = scope_score*100
        
        # combined_score
        final_score = combined_score(
            answerability_score,
            relevance_score,
            scope_score,
            answerability_weight=1.0,
            relevance_weight=1.0,
            scope_weight=1.0
            )
        
        runtime = time.time() - start_time
        
        # Store results
        tracker.add_run(
            run_id=run_id,
            relevance=relevance_score,
            scope=scope_score,
            combined_score = final_score,
            answerability=answerability_score,
            coverage_pct=coverage_pct,
            runtime_seconds=runtime,
            details={
                'size_of_cqs': len(cqs_corpus),
                'num_clusters': len(df_clusters),
            }
        )
        
        if verbose:
            print(f"  Relevance: {relevance_score:.4f}")
            print(f"  Scope: {scope_score:.4f}")
            print(f"  Answerability: {answerability_score:.4f}")
            print(f"  Coverage percentage: {coverage_pct:.1f}%")
            print(f"  combined_score: {final_score:.1f}%")
            print(f"  Runtime: {runtime:.2f}s")
    
    return tracker

   
