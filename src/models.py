"""
@file models.py
@description Core domain data models and Pydantic schemas for RefChecker-Mini
@module src/models
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class BenchmarkSample(BaseModel):
    """Represents a benchmark test sample."""
    model_config = ConfigDict(extra="forbid")
    id: str = Field(..., description="Unique sample identifier, e.g. sample_001")
    question: str = Field(..., description="Input query or prompt")
    context: List[str] = Field(default_factory=list, description="Grounding context / passages")
    reference: str = Field(..., description="Gold reference or ground-truth answer")
    setting: Literal["zero_context", "noisy_context", "accurate_context"] = Field(
        ..., description="Benchmark setting"
    )
    generator: Optional[str] = Field(default=None, description="Model used to generate the test response")
    response: Optional[str] = Field(default=None, description="LLM response being evaluated")
    ground_truth_label: Optional[Literal["Entailment", "Neutral", "Contradiction", "Hallucinated", "Factual"]] = Field(
        default=None, description="Human or gold evaluation label if available"
    )


class ExtractedSentence(BaseModel):
    """Represents a sentence segmented from a response."""
    model_config = ConfigDict(extra="forbid")
    sample_id: str
    sentence_index: int
    text: str


class ExtractedClaim(BaseModel):
    """Represents an atomic factual statement extracted from text."""
    model_config = ConfigDict(extra="forbid")
    sample_id: str
    claim_index: int
    claim_text: str
    source_sentence: Optional[str] = None


class ClaimTriplet(BaseModel):
    """Represents a structured knowledge claim triplet (subject, relation, object)."""
    model_config = ConfigDict(extra="forbid")
    sample_id: str
    triplet_index: int
    subject: str
    relation: str
    object: str
    verbalized_claim: Optional[str] = None

    def get_verbalized(self) -> str:
        """Returns verbalized natural language sentence for the triplet."""
        if self.verbalized_claim and self.verbalized_claim.strip():
            return self.verbalized_claim
        return f"{self.subject} {self.relation} {self.object}."


class VerificationUnitResult(BaseModel):
    """Result of verifying a single unit (sentence, claim, or triplet) against reference."""
    model_config = ConfigDict(extra="forbid")
    unit_id: str
    sample_id: str
    unit_type: Literal["sentence", "claim", "triplet"]
    unit_text: str
    verifier: str  # e.g., 'nli_deberta', 'llm_judge', 'semantic_similarity'
    label: Literal["Entailment", "Neutral", "Contradiction"]
    probabilities: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResponseEvaluationResult(BaseModel):
    """Aggregate response-level factual consistency and hallucination decision."""
    model_config = ConfigDict(extra="forbid")
    sample_id: str
    setting: str
    verifier: str
    granularity: Literal["sentence", "claim", "triplet"]
    unit_results: List[VerificationUnitResult]
    response_label: Literal["Entailment", "Neutral", "Contradiction"]
    is_hallucinated: bool
    contradiction_count: int
    neutral_count: int
    entailment_count: int
    score: float = Field(default=0.0, description="Normalized factual score [0.0 - 1.0]")
