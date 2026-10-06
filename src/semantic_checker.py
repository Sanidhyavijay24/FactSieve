"""
@file semantic_checker.py
@description Semantic similarity fact verifier using Sentence-Transformers and cosine similarity
@module src/semantic_checker
"""

from typing import Any, Dict, List, Literal, Optional, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from src.config import AppConfig, load_config
from src.models import ClaimTriplet, ExtractedClaim, ExtractedSentence, VerificationUnitResult


class SemanticSimilarityChecker:
    """Evaluates factual consistency via semantic embedding similarity."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        threshold: Optional[float] = None,
        device: Optional[str] = None,
        config: Optional[AppConfig] = None,
    ):
        """Initialize SemanticSimilarityChecker.

        Args:
            model_name (Optional[str]): SentenceTransformers model name.
            threshold (Optional[float]): Cosine similarity threshold for Entailment.
            device (Optional[str]): 'cuda', 'cpu', or 'auto'.
            config (Optional[AppConfig]): Optional application config.
        """
        self.config = config or load_config()
        self.model_name = model_name or self.config.verifiers.semantic_similarity.model_name
        self.threshold = threshold if threshold is not None else self.config.verifiers.semantic_similarity.threshold

        # Device determination
        target_device = device or self.config.verifiers.semantic_similarity.device
        if target_device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = target_device

        self.model = SentenceTransformer(self.model_name, device=self.device)

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity score between two texts in range [-1.0, 1.0].

        Args:
            text_a (str): Premise / reference.
            text_b (str): Hypothesis / claim text.

        Returns:
            float: Cosine similarity score.
        """
        embeddings = self.model.encode([text_a, text_b], convert_to_numpy=True, normalize_embeddings=True)
        # Cosine similarity of normalized vectors is the dot product
        similarity = float(np.dot(embeddings[0], embeddings[1]))
        return similarity

    def verify_units(
        self,
        sample_id: str,
        units: List[Union[ExtractedSentence, ExtractedClaim, ClaimTriplet]],
        evidence: str,
    ) -> List[VerificationUnitResult]:
        """Verify units against reference evidence using semantic embedding similarity.

        Args:
            sample_id (str): Sample identifier.
            units (List[Union[ExtractedSentence, ExtractedClaim, ClaimTriplet]]): List of units.
            evidence (str): Premise / reference text.

        Returns:
            List[VerificationUnitResult]: Structured unit results.
        """
        if not units:
            return []

        hypotheses: List[str] = []
        unit_types: List[str] = []
        unit_ids: List[str] = []

        for idx, u in enumerate(units):
            if isinstance(u, ExtractedSentence):
                hypotheses.append(u.text)
                unit_types.append("sentence")
                unit_ids.append(f"{sample_id}_sent_{u.sentence_index}")
            elif isinstance(u, ExtractedClaim):
                hypotheses.append(u.claim_text)
                unit_types.append("claim")
                unit_ids.append(f"{sample_id}_claim_{u.claim_index}")
            elif isinstance(u, ClaimTriplet):
                hypotheses.append(u.get_verbalized())
                unit_types.append("triplet")
                unit_ids.append(f"{sample_id}_triplet_{u.triplet_index}")
            else:
                hypotheses.append(str(u))
                unit_types.append("sentence")
                unit_ids.append(f"{sample_id}_unit_{idx}")

        # Compute embeddings in batch
        all_texts = [evidence] + hypotheses
        embeddings = self.model.encode(all_texts, convert_to_numpy=True, normalize_embeddings=True)
        evidence_vec = embeddings[0]
        hyp_vecs = embeddings[1:]

        results: List[VerificationUnitResult] = []

        for u_id, u_type, hyp_text, hyp_vec in zip(unit_ids, unit_types, hypotheses, hyp_vecs):
            sim_score = float(np.dot(evidence_vec, hyp_vec))

            # Multi-tier decision threshold
            if sim_score >= self.threshold:
                label: Literal["Entailment", "Neutral", "Contradiction"] = "Entailment"
                probs = {"Entailment": sim_score, "Neutral": 1.0 - sim_score, "Contradiction": 0.0}
            elif sim_score < 0.35:
                label = "Contradiction"
                probs = {"Entailment": 0.0, "Neutral": sim_score, "Contradiction": 1.0 - sim_score}
            else:
                label = "Neutral"
                probs = {"Entailment": (1.0 - sim_score) / 2, "Neutral": sim_score, "Contradiction": (1.0 - sim_score) / 2}

            results.append(
                VerificationUnitResult(
                    unit_id=u_id,
                    sample_id=sample_id,
                    unit_type=u_type,  # type: ignore
                    unit_text=hyp_text,
                    verifier=f"semantic_similarity_{self.model_name.split('/')[-1]}",
                    label=label,
                    probabilities=probs,
                    metadata={"similarity_score": round(sim_score, 4), "threshold": self.threshold},
                )
            )

        return results
