"""Stable forest interface: Reference, V2 and V2.5 preserve scalar search semantics."""
from models.forest_reference import (
    Cell, ReferenceSemanticForest, _gather, catalan, compose_tree,
    masked_log_softmax, tree_for_length,
)
from models.forest_score_first import SemanticForest as ScoreFirstSemanticForest
from models.forest_lazy import LazyForestMixin
from models.forest_tensorized import TensorizedForestMixin,TENSORIZED_IMPLEMENTATIONS


class SemanticForest(TensorizedForestMixin, LazyForestMixin, ScoreFirstSemanticForest):
    def __init__(self, composer, k=4, max_compositions=12, delta=8.0, exploration=0.1,
                 implementation="score_first"):
        super().__init__(composer,k,max_compositions,delta,exploration,
                         "score_first" if implementation in ("lazy_semantic",)+TENSORIZED_IMPLEMENTATIONS else implementation)
        self.implementation = implementation
