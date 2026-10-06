"""
@file llm_checker.py
@description LLM Judge verifier with structured 3-way classification and persistent caching
@module src/llm_checker
"""

import os
import re
import time
from typing import Any, Dict, List, Literal, Optional, Union
from dotenv import load_dotenv
from google import genai
from src.cache import DiskCache
from src.config import AppConfig, load_config
from src.models import ClaimTriplet, ExtractedClaim, ExtractedSentence, VerificationUnitResult


LLM_JUDGE_PROMPT = """You are an objective fact verification judge.
Compare the Claim against the provided Evidence and determine the logical relationship.

Evidence:
"{evidence}"

Claim / Triplet:
"{claim}"

Task:
Determine whether the Claim is ENTAILMENT, CONTRADICTION, or NEUTRAL relative to the Evidence:
- ENTAILMENT: The claim is directly supported by or logically follows from the evidence.
- CONTRADICTION: The claim is directly refuted by or conflicts with the evidence.
- NEUTRAL: The claim is not mentioned or cannot be verified solely from the evidence.

Output Format:
Output ONLY one single word: ENTAILMENT, CONTRADICTION, or NEUTRAL.
No punctuation, no explanations.
"""


class LLMJudgeChecker:
    """Evaluates factual consistency using Gemini with few-shot judge prompting and disk caching."""

    def __init__(self, config: Optional[AppConfig] = None):
        """Initialize LLMJudgeChecker with config, Gemini client, and disk cache.

        Args:
            config (Optional[AppConfig]): Application configuration.
        """
        load_dotenv()
        self.config = config or load_config()
        self.cache = DiskCache(self.config.verifiers.llm_judge.cache_file)
        self.prompt_version = self.config.verifiers.llm_judge.prompt_version
        self.model_name = self.config.llm.judge_model

        api_key = os.getenv("GEMINI_API_KEY") or self.config.llm.api_key
        self.client = genai.Client(api_key=api_key) if api_key else None

    def _parse_judge_output(self, text: str) -> Literal["Entailment", "Neutral", "Contradiction"]:
        """Parse raw LLM response into canonical 3-way label.

        Args:
            text (str): Raw model output.

        Returns:
            Literal["Entailment", "Neutral", "Contradiction"]: Normalized label.
        """
        cleaned = text.strip().upper()
        if "CONTRADICT" in cleaned:
            return "Contradiction"
        elif "ENTAIL" in cleaned or "SUPPORT" in cleaned:
            return "Entailment"
        else:
            return "Neutral"

    def verify_single(
        self,
        evidence: str,
        claim_text: str,
        sample_id: str = "sample",
        overwrite_cache: bool = False,
    ) -> Literal["Entailment", "Neutral", "Contradiction"]:
        """Evaluate a single (evidence, claim) pair.

        Args:
            evidence (str): Reference evidence.
            claim_text (str): Claim or triplet to verify.
            sample_id (str): Sample identifier.
            overwrite_cache (bool): If True, bypass cache.

        Returns:
            Literal["Entailment", "Neutral", "Contradiction"]: Judgment outcome.
        """
        cache_payload = {
            "task": "llm_judge_verification",
            "prompt_version": self.prompt_version,
            "evidence": evidence.strip(),
            "claim": claim_text.strip(),
        }
        cache_key = DiskCache.compute_key(cache_payload)

        if not overwrite_cache:
            cached_val = self.cache.get(cache_key)
            if cached_val is not None:
                return self._parse_judge_output(str(cached_val))

        if not self.client:
            # Fallback simple string matching heuristic if no API key
            if any(word.lower() in evidence.lower() for word in claim_text.split() if len(word) > 4):
                return "Entailment"
            return "Neutral"

        prompt = LLM_JUDGE_PROMPT.format(evidence=evidence.strip(), claim=claim_text.strip())
        max_retries = 3

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                raw_text = response.text.strip() if response.text else "NEUTRAL"
                parsed_label = self._parse_judge_output(raw_text)
                self.cache.set(cache_key, parsed_label, metadata={"sample_id": sample_id})
                time.sleep(2.0)  # Gentle rate pacing
                return parsed_label
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait_time = 12.0 * attempt
                    print(f" [RateLimit 429] Sample {sample_id}: Backing off {wait_time:.0f}s (Attempt {attempt}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"[LLMJudge Warning] Call failed for {sample_id}: {e}. Defaulting to Neutral.")
                    return "Neutral"

        return "Neutral"

    def verify_units(
        self,
        sample_id: str,
        units: List[Union[ExtractedSentence, ExtractedClaim, ClaimTriplet]],
        evidence: str,
    ) -> List[VerificationUnitResult]:
        """Verify units against reference evidence using the LLM Judge.

        Args:
            sample_id (str): Sample identifier.
            units (List[Union[ExtractedSentence, ExtractedClaim, ClaimTriplet]]): List of units.
            evidence (str): Premise or ground truth text.

        Returns:
            List[VerificationUnitResult]: Structured unit verdicts.
        """
        results: List[VerificationUnitResult] = []

        for idx, u in enumerate(units):
            if isinstance(u, ExtractedSentence):
                hyp_text = u.text
                u_type = "sentence"
                unit_id = f"{sample_id}_sent_{u.sentence_index}"
            elif isinstance(u, ExtractedClaim):
                hyp_text = u.claim_text
                u_type = "claim"
                unit_id = f"{sample_id}_claim_{u.claim_index}"
            elif isinstance(u, ClaimTriplet):
                hyp_text = u.get_verbalized()
                u_type = "triplet"
                unit_id = f"{sample_id}_triplet_{u.triplet_index}"
            else:
                hyp_text = str(u)
                u_type = "sentence"
                unit_id = f"{sample_id}_unit_{idx}"

            label = self.verify_single(evidence=evidence, claim_text=hyp_text, sample_id=sample_id)
            probs = {"Entailment": 0.0, "Neutral": 0.0, "Contradiction": 0.0}
            probs[label] = 1.0

            results.append(
                VerificationUnitResult(
                    unit_id=unit_id,
                    sample_id=sample_id,
                    unit_type=u_type,  # type: ignore
                    unit_text=hyp_text,
                    verifier=f"llm_judge_{self.model_name}",
                    label=label,
                    probabilities=probs,
                    metadata={"evidence_length": len(evidence)},
                )
            )

        return results
