"""
@file nli_checker.py
@description Natural Language Inference (NLI) verifier using Hugging Face DeBERTa/RoBERTa
@module src/nli_checker
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.config import AppConfig, load_config
from src.models import ClaimTriplet, ExtractedClaim, ExtractedSentence, VerificationUnitResult


class NLIChecker:
    """Performs 3-way Natural Language Inference (Entailment, Neutral, Contradiction)."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 16,
        config: Optional[AppConfig] = None,
    ):
        """Initialize NLIChecker with model, tokenizer, and target inference device.

        Args:
            model_name (Optional[str]): HuggingFace checkpoint name.
            device (Optional[str]): 'cuda', 'cpu', or 'auto'.
            batch_size (int): Batch size for inference.
            config (Optional[AppConfig]): Optional application config.
        """
        self.config = config or load_config()
        self.model_name = model_name or self.config.verifiers.nli.model_name
        self.batch_size = batch_size or self.config.verifiers.nli.batch_size

        # Resolve device
        target_device = device or self.config.verifiers.nli.device
        if target_device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(target_device)

        # Load tokenizer and sequence classification model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # Build label index mapping
        self._label_map = self._build_label_mapping()

    def _build_label_mapping(self) -> Dict[int, str]:
        """Normalize HuggingFace model id2label to canonical labels: Entailment, Neutral, Contradiction."""
        id2label = getattr(self.model.config, "id2label", {})
        label_map: Dict[int, str] = {}

        for idx, lbl in id2label.items():
            lbl_lower = str(lbl).lower()
            if "entail" in lbl_lower:
                label_map[int(idx)] = "Entailment"
            elif "contra" in lbl_lower:
                label_map[int(idx)] = "Contradiction"
            elif "neut" in lbl_lower:
                label_map[int(idx)] = "Neutral"
            else:
                label_map[int(idx)] = str(lbl)

        # Fallback if id2label is missing or uninformative
        if len(label_map) != 3:
            # Common default for MNLI models: 0 -> contradiction, 1 -> neutral, 2 -> entailment
            # OR 0 -> entailment, 1 -> neutral, 2 -> contradiction
            label_map = {0: "Contradiction", 1: "Neutral", 2: "Entailment"}

        return label_map

    def predict_batch(
        self, premises: List[str], hypotheses: List[str]
    ) -> List[Tuple[Literal["Entailment", "Neutral", "Contradiction"], Dict[str, float]]]:
        """Run batched sequence classification inference.

        Args:
            premises (List[str]): Evidence or reference texts.
            hypotheses (List[str]): Claims, triplets, or sentences being verified.

        Returns:
            List[Tuple[str, Dict[str, float]]]: Predicted label and normalized class probabilities.
        """
        if not premises or not hypotheses:
            return []

        results: List[Tuple[Literal["Entailment", "Neutral", "Contradiction"], Dict[str, float]]] = []

        for i in range(0, len(premises), self.batch_size):
            batch_prem = premises[i : i + self.batch_size]
            batch_hyp = hypotheses[i : i + self.batch_size]

            inputs = self.tokenizer(
                batch_prem,
                batch_hyp,
                padding=True,
                truncation=True,
                max_length=self.config.verifiers.nli.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            for prob_row in probs:
                prob_dict: Dict[str, float] = {"Entailment": 0.0, "Neutral": 0.0, "Contradiction": 0.0}
                for idx, p in enumerate(prob_row):
                    canon_label = self._label_map.get(idx, f"Label_{idx}")
                    if canon_label in prob_dict:
                        prob_dict[canon_label] = float(p)

                # Determine predicted label with highest probability
                pred_label = max(prob_dict.items(), key=lambda x: x[1])[0]  # type: ignore
                results.append((pred_label, prob_dict))  # type: ignore

        return results

    def verify_units(
        self,
        sample_id: str,
        units: List[Union[ExtractedSentence, ExtractedClaim, ClaimTriplet]],
        evidence: str,
    ) -> List[VerificationUnitResult]:
        """Verify an arbitrary list of representation units against reference evidence.

        Args:
            sample_id (str): Sample identifier.
            units (List[Union[ExtractedSentence, ExtractedClaim, ClaimTriplet]]): Units to verify.
            evidence (str): Premise or ground truth reference text.

        Returns:
            List[VerificationUnitResult]: Structured unit verification outcomes.
        """
        if not units:
            return []

        premises: List[str] = []
        hypotheses: List[str] = []
        unit_types: List[str] = []
        unit_ids: List[str] = []

        for idx, u in enumerate(units):
            premises.append(evidence)
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

        predictions = self.predict_batch(premises, hypotheses)
        results: List[VerificationUnitResult] = []

        for unit_id, u_type, hyp_text, (label, probs) in zip(unit_ids, unit_types, hypotheses, predictions):
            results.append(
                VerificationUnitResult(
                    unit_id=unit_id,
                    sample_id=sample_id,
                    unit_type=u_type,  # type: ignore
                    unit_text=hyp_text,
                    verifier=f"nli_{self.model_name.split('/')[-1]}",
                    label=label,
                    probabilities=probs,
                    metadata={"evidence_length": len(evidence)},
                )
            )

        return results
