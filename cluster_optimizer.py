"""
Cluster Analysis and Optimization Tools
Helper utilities for determining optimal number of clusters for domain-specific text
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Dict
import transformers
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
import seaborn as sns
import traceback
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model configuration
model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
max_length = 128
model = AutoModel.from_pretrained(model_id).to(device)
# from themes_with_kmeans import encode_with_automodel

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
# save_path= os.getcwd() + "/outputs/plots/metric_plots.png"
save_path= os.getcwd() + "/outputs/covid_data/covid_data_metrics/plots/metric_plots.png"
class ClusterOptimizer:
    """
    Tools for finding optimal number of clusters for text data
    """
    
    def __init__(self, k_range=range(2, 21), random_state=42):
   
        """
        Initialize with embedding model
        
        Args:
            embedding_model: SentenceTransformer model for text embeddings
        """
        self.embedding_model = model 
        self.text_embeddings = None
        self.silhouette_scores = None
        self.texts = None
        self.results = None
        self.best_k = None
        self.kmeans_models = {}
        self._computed = False
        self.k_range = list(k_range)
        self.random_state = random_state
    def reset(self):
        """Reset all computed results. Call this before re-running with same instance."""
        self.results = {
            'k': list(self.k_range),
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': [],
            'inertia': [],
            'stability_mean': [],
            'stability_std': []
        }
        self.kmeans_models = {}
        self._computed = False
        print(" Results reset")
    
    def set_k_range(self, k_range):
        """Change k_range and reset results."""
        self.k_range = list(k_range)
        self.reset()
        print(f" k_range updated to: {min(self.k_range)} to {max(self.k_range)}")
    
    def evaluate_clusters(self, texts: pd.DataFrame, normalize_embeddings=True, n_stability_runs=20, verbose=True) -> pd.DataFrame:
        """
        Evaluate different numbers of clusters using multiple metrics
        
        Args:
            texts: List of text documents
            min_k: Minimum number of clusters
            max_k: Maximum number of clusters
            
        Returns:
            DataFrame with evaluation metrics
        """
        # min_k=2 
        # max_k=21
        # k_range=range(2,21)
        # self.k_range = list(k_range)
        text_embeddings= np.array(texts['embeddings'].tolist())
        if normalize_embeddings:
            self.text_embeddings = normalize(text_embeddings, norm='l2')
            print("Embeddings L2-normalized (spherical K-Means)")
        else:
            self.text_embeddings = text_embeddings
        n_samples = len(self.text_embeddings)
        if n_samples < 2:
            raise ValueError(f"Need at least 2 samples, got {n_samples}")
        min_k = 2
        max_k = min(21, n_samples - 1)
        if max_k < min_k:
            raise ValueError(f"Not enough samples ({n_samples}) to evaluate k >= {min_k}")
        # IMPORTANT: k values must match what you actually compute
        self.k_range = list(range(min_k, max_k + 1))     
        self.results = {
            'k': [],
            'inertia': [],
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': [],
            'stability_mean': [],
            'stability_std': []
            
        }
        self._computed = False
        self.kmeans_models = {}
        self.silhouette_scores = []
        # max_k = min(max_k, n_samples - 1)
        # max_k = min(max_k, len(texts) - 1)  #text is a dataframe, so this is wrong
        print(f"Evaluating clusters from k={min_k} to k={max_k}...")
        
        for k in self.k_range:
            print(f"  Testing k={k}...")
            # Fit K-means
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=6, max_iter=300)
            labels = kmeans.fit_predict(self.text_embeddings)
            print(" Store kmeans results")
            self.results['k'].append(k)
            self.results['inertia'].append(kmeans.inertia_)
            score = davies = calinski = 0.0
            # Calculate metrics
            if k > 1 and len(set(labels)) > 1:
                score = silhouette_score(self.text_embeddings, labels, metric="cosine")
                davies= davies_bouldin_score(self.text_embeddings, labels)
                calinski = calinski_harabasz_score(self.text_embeddings, labels)
                
            self.silhouette_scores.append(score)
            self.results['silhouette'].append(score)
            self.results['davies_bouldin'].append(davies)
            self.results['calinski_harabasz'].append(calinski)
            # else:
            #     self.silhouette_scores.append(0)
            #     self.results['silhouette'].append(0)
            #     self.results['davies_bouldin'].append(0)
            #     self.results['calinski_harabasz'].append(0)
            if verbose:
                print(f"Sil={score:.3f}, DB={davies:.3f}, CH={calinski:.1f}", end=" ")
            
            # Stability analysis
            stability_scores = self._compute_stability(k, n_runs=n_stability_runs)
            self.results['stability_mean'].append(np.mean(stability_scores))
            self.results['stability_std'].append(np.std(stability_scores))
            
            lens = {k: len(v) for k, v in self.results.items()}
            print(lens)
            assert len(set(lens.values())) == 1, f"Mismatch: {lens}"
            if verbose:
                print(f"Stab={np.mean(stability_scores):.3f}±{np.std(stability_scores):.3f}")
        
        self._computed = True
        
        if verbose:
            print("\n" + "="*60)
            print(" All metrics computed successfully")
            print("="*60)

        
        # if self.silhouette_scores:
        #     best_idx = np.argmax(self.silhouette_scores)
        #     self.best_k = self.results['k'][best_idx]
            # print(f"\nBest k={self.best_k} with silhouette score={self.silhouette_scores[best_idx]:.3f}")
        
        return self.results
    def _compute_stability(self, k, n_runs=20):
        """
        Compute stability via multiple K-Means runs with different initializations.
        """
        all_labels = []
        
        for run in range(n_runs):
            km = KMeans(n_clusters=k, n_init=1, random_state=run)
            labels = km.fit_predict(self.text_embeddings)
            all_labels.append(labels)
        
        # Pairwise Adjusted Rand Index
        ari_scores = []
        for i, j in combinations(range(n_runs), 2):
            ari = adjusted_rand_score(all_labels[i], all_labels[j])
            ari_scores.append(ari)
        
        return ari_scores
    
    def _normalize_metric(self, values, higher_is_better=True):
        """Normalize metric to 0-1 scale."""
        values = np.array(values)
        if len(set(values)) == 1:
            return np.ones_like(values) * 0.5
        
        normalized = (values - values.min()) / (values.max() - values.min())
        if not higher_is_better:
            normalized = 1 - normalized
        return normalized
    
    def find_optimal_k(self):
        """
        Find optimal k using normalized composite score.
        """
        if not self._computed:
            raise ValueError("Run compute_all_metrics() first")
        
        # Normalize all metrics (higher = better after normalization)
        sil_norm = self._normalize_metric(self.results['silhouette'], higher_is_better=True)
        db_norm = self._normalize_metric(self.results['davies_bouldin'], higher_is_better=False)
        ch_norm = self._normalize_metric(self.results['calinski_harabasz'], higher_is_better=True)
        stab_norm = self._normalize_metric(self.results['stability_mean'], higher_is_better=True)
        
        # Composite score (equal weights, adjustable)
        composite = (sil_norm + db_norm + ch_norm + stab_norm) / 4
        
        best_idx = np.argmax(composite)
        self.best_k = self.results['k'][best_idx]
        
        return {
            'optimal_k': self.best_k,
            'composite_scores': composite,
            'normalized_metrics': {
                'silhouette': sil_norm,
                'davies_bouldin': db_norm,
                'calinski_harabasz': ch_norm,
                'stability': stab_norm
            }
        }
    def get_final_clusters(self, k=None):
        """Get final clustering with specified or best k"""
        if self.text_embeddings is None:
            raise ValueError("Run evaluate_clusters first")
        print("k is:", k)
        k = k or self.best_k
        if k is None:
            raise ValueError("No k specified and no best_k found")
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(self.text_embeddings)
        
        return labels, kmeans
    def plot_dashboard(self, figsize=(16, 12), save_path=None):
        """
        Create comprehensive visualization dashboard.
        """
        if not self._computed:
            raise ValueError("Run compute_all_metrics() first")
        
        # Get optimal k
        optimal_info = self.find_optimal_k()
        optimal_k = optimal_info['optimal_k']
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        
        # Define grid: 3 rows, 2 columns
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
        
        ax1 = fig.add_subplot(gs[0, 0])  # Silhouette
        ax2 = fig.add_subplot(gs[0, 1])  # Davies-Bouldin
        ax3 = fig.add_subplot(gs[1, 0])  # Calinski-Harabasz
        ax4 = fig.add_subplot(gs[1, 1])  # Elbow
        ax5 = fig.add_subplot(gs[2, 0])  # Stability
        ax6 = fig.add_subplot(gs[2, 1])  # Composite / Summary
        
        k_vals = self.results['k']
        
        # Color scheme
        colors = {
            'silhouette': '#2ecc71',
            'davies_bouldin': '#e74c3c', 
            'calinski_harabasz': '#3498db',
            'inertia': '#9b59b6',
            'stability': '#f39c12',
            'composite': '#1abc9c'
        }
        
        # 1. Silhouette Score (higher is better)
        ax1.plot(k_vals, self.results['silhouette'], 'o-', color=colors['silhouette'], 
                 linewidth=2, markersize=8)
        ax1.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
        best_sil_k = k_vals[np.argmax(self.results['silhouette'])]
        ax1.axvline(x=best_sil_k, color=colors['silhouette'], linestyle=':', alpha=0.7, 
                    label=f'Best Silhouette k={best_sil_k}')
        ax1.set_xlabel('Clusters (k)')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_title('Silhouette Score (↑ higher is better)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=9)
        ax1.set_xticks(k_vals)
        
        # 2. Davies-Bouldin Score (lower is better)
        ax2.plot(k_vals, self.results['davies_bouldin'], 'o-', color=colors['davies_bouldin'],
                 linewidth=2, markersize=8)
        ax2.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
        best_db_k = k_vals[np.argmin(self.results['davies_bouldin'])]
        ax2.axvline(x=best_db_k, color=colors['davies_bouldin'], linestyle=':', alpha=0.7,
                    label=f'Best DB k={best_db_k}')
        ax2.set_xlabel('Clusters (k)')
        ax2.set_ylabel('Davies-Bouldin Score')
        ax2.set_title('Davies-Bouldin Score (↓ lower is better)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=9)
        ax2.set_xticks(k_vals)
        
        # 3. Calinski-Harabasz Score (higher is better)
        ax3.plot(k_vals, self.results['calinski_harabasz'], 'o-', color=colors['calinski_harabasz'],
                 linewidth=2, markersize=8)
        ax3.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
        best_ch_k = k_vals[np.argmax(self.results['calinski_harabasz'])]
        ax3.axvline(x=best_ch_k, color=colors['calinski_harabasz'], linestyle=':', alpha=0.7,
                    label=f'Best CH k={best_ch_k}')
        ax3.set_xlabel('Clusters (k)')
        ax3.set_ylabel('Calinski-Harabasz Score')
        ax3.set_title('Calinski-Harabasz Score (↑ higher is better)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best', fontsize=9)
        ax3.set_xticks(k_vals)
        
        # 4. Elbow Plot (Inertia)
        ax4.plot(k_vals, self.results['inertia'], 'o-', color=colors['inertia'],
                 linewidth=2, markersize=8)
        ax4.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
        ax4.set_xlabel('Clusters (k)')
        ax4.set_ylabel('Inertia (Within-cluster SS)')
        ax4.set_title('Elbow Plot (look for bend)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='best', fontsize=9)
        ax4.set_xticks(k_vals)
        
        # 5. Stability Analysis
        stab_mean = self.results['stability_mean']
        stab_std = self.results['stability_std']
        ax5.errorbar(k_vals, stab_mean, yerr=stab_std, fmt='o-', color=colors['stability'],
                     linewidth=2, markersize=8, capsize=4, capthick=2)
        ax5.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
        best_stab_k = k_vals[np.argmax(stab_mean)]
        ax5.axvline(x=best_stab_k, color=colors['stability'], linestyle=':', alpha=0.7,
                    label=f'Most stable k={best_stab_k}')
        ax5.axhline(y=0.7, color='gray', linestyle=':', alpha=0.5, label='Stability threshold')
        ax5.set_xlabel('Clusters (k)')
        ax5.set_ylabel('Adjusted Rand Index')
        ax5.set_title('Stability Analysis (↑ higher is better)', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend(loc='best', fontsize=9)
        ax5.set_xticks(k_vals)
        ax5.set_ylim([0, 1.05])
        
        # 6. Normalized Composite Score
        norm_metrics = optimal_info['normalized_metrics']
        
        ax6.plot(k_vals, norm_metrics['silhouette'], 's--', color=colors['silhouette'],
                 alpha=0.7, label='Silhouette (norm)', markersize=6)
        ax6.plot(k_vals, norm_metrics['davies_bouldin'], '^--', color=colors['davies_bouldin'],
                 alpha=0.7, label='Davies-Bouldin (norm)', markersize=6)
        ax6.plot(k_vals, norm_metrics['calinski_harabasz'], 'd--', color=colors['calinski_harabasz'],
                 alpha=0.7, label='Calinski-Harabasz (norm)', markersize=6)
        ax6.plot(k_vals, norm_metrics['stability'], 'p--', color=colors['stability'],
                 alpha=0.7, label='Stability (norm)', markersize=6)
        ax6.plot(k_vals, optimal_info['composite_scores'], 'o-', color=colors['composite'],
                 linewidth=3, markersize=10, label='Composite Score')
        ax6.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax6.scatter([optimal_k], [optimal_info['composite_scores'][k_vals.index(optimal_k)]],
                    color='red', s=200, zorder=5, marker='*', label=f'Optimal k={optimal_k}')
        ax6.set_xlabel('Number of Clusters (k)')
        ax6.set_ylabel('Normalized Score')
        ax6.set_title('Normalized Metrics & Composite Score', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        ax6.set_xticks(k_vals)
        ax6.set_ylim([-0.05, 1.1])
        
        # Main title
        fig.suptitle(f'Clustering Metrics: Finding k for KMeans model\n'
                     f'Optimal k = {optimal_k} (based on composite score)',
                     fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"\n Dashboard saved to: {save_path}")
        
        plt.show()
        
        return fig,optimal_k
    
    def get_summary_table(self):
        """
        Return a summary table of all metrics.
        """
        if not self._computed:
            raise ValueError("Run compute_all_metrics() first")
        
        # optimal_info = self.find_optimal_k()
        # Validate all lists have same length
        k_list = list(self.results['k'])
        n_k = len(k_list)
        
        for key in ['silhouette', 'davies_bouldin', 'calinski_harabasz', 'stability_mean', 'stability_std']:
            if len(self.results[key]) != n_k:
                raise ValueError(f"Length mismatch: k has {n_k} values but {key} has {len(self.results[key])}. "
                                 f"Re-run compute_all_metrics().")
        
        optimal_info = self.find_optimal_k()
        composite_scores = list(optimal_info['composite_scores'])
        print("\n" + "="*80)
        print("CLUSTERING METRICS SUMMARY")
        print("="*80)
        print(f"{'k':<5} {'Silhouette':<12} {'Davies-Bouldin':<15} {'Calinski-H':<12} "
              f"{'Stability':<15} {'Composite':<10}")
        print("-"*80)
        
        for i, k in enumerate(self.results['k']):
            marker = " ← OPTIMAL" if k == optimal_info['optimal_k'] else ""
            print(f"{k:<5} {self.results['silhouette'][i]:<12.4f} "
                  f"{self.results['davies_bouldin'][i]:<15.4f} "
                  f"{self.results['calinski_harabasz'][i]:<12.1f} "
                  f"{self.results['stability_mean'][i]:.3f}±{self.results['stability_std'][i]:.3f}    "
                  f"{optimal_info['composite_scores'][i]:<10.4f}{marker}")
        
        
        print("="*80)
        
        # Individual metric recommendations
        print("\nIndividual Metric Recommendations:")
        print(f"  • Silhouette (max):        k = {self.results['k'][np.argmax(self.results['silhouette'])]}")
        print(f"  • Davies-Bouldin (min):    k = {self.results['k'][np.argmin(self.results['davies_bouldin'])]}")
        print(f"  • Calinski-Harabasz (max): k = {self.results['k'][np.argmax(self.results['calinski_harabasz'])]}")
        print(f"  • Stability (max):         k = {self.results['k'][np.argmax(self.results['stability_mean'])]}")
        print(f"\n  ★ Composite Optimal:       k = {optimal_info['optimal_k']}")
        
        return self.results
