from themes_with_kmeans import *
from answerability import *
from scope import *
from relevance import *

# composite score calculation is the weighted average of relevance, scope and answerability scores which has a maximum value of 1.0
def combined_score(relevance_score: float, scope_score: float, answerability_score: float,
                    relevance_weight: float = 1.0, scope_weight: float = 1.0, answerability_weight: float = 1.0) -> float:
    """
    Calculate composite score based on relevance, scope, and answerability scores with given weights.
    
    Args:
        relevance_score (float): Relevance score.
        scope_score (float): Scope score.
        answerability_score (float): Answerability score.
        relevance_weight (float): Weight for relevance score.
        scope_weight (float): Weight for scope score.
        answerability_weight (float): Weight for answerability score.

    Returns:
        float: Composite score.
    """
    total_weight = relevance_weight + scope_weight + answerability_weight
    composite = ((relevance_score * relevance_weight) + 
                 (scope_score * scope_weight) + 
                 (answerability_score * answerability_weight)) / total_weight
    return composite