"""
@file extract_claims.py
@description Atomic claim extraction module using structured LLM prompting and disk caching
@module src/extract_claims
"""

import json
import os
import re
import time
from typing import List, Optional
from dotenv import load_dotenv
from google import genai
from src.cache import DiskCache
from src.config import AppConfig, load_config
from src.extract_sentences import SentenceExtractor
from src.models import ExtractedClaim


CLAIM_EXTRACTION_PROMPT = """You are an expert NLP system specialized in factual claim decomposition.
Given an input text, decompose it into a list of standalone atomic factual claims.
Each claim must:
1. Express exactly one independent, verifiable fact.
2. Be fully self-contained and decontextualized (replace pronouns with actual nouns).
3. Do not add any facts that are not present in the original text.

Input Text:
"{text}"

Output Format:
Return ONLY a valid JSON array of strings, with no markdown code fences and no surrounding explanation.
Example: ["The Eiffel Tower was completed in 1889.", "The Eiffel Tower is located in Paris."]
"""


class ClaimExtractor:
    """Decomposes text responses into discrete atomic factual claims."""

    def __init__(self, config: Optional[AppConfig] = None):
        """Initialize ClaimExtractor with config, Gemini client, and cache.

        Args:
            config (Optional[AppConfig]): Application configuration.
        """
        load_dotenv()
        self.config = config or load_config()
        self.cache = DiskCache(self.config.extractors.cache_file)
        self.sentence_extractor = SentenceExtractor(self.config.extractors.spacy_model)
        self.prompt_version = self.config.extractors.claim_prompt_version
        self.model_name = self.config.llm.generator_model

        api_key = os.getenv("GEMINI_API_KEY") or self.config.llm.api_key
        self.client = genai.Client(api_key=api_key) if api_key else None

    def _fallback_extract(self, text: str, sample_id: str) -> List[ExtractedClaim]:
        """Heuristic fallback using sentence segmentation when LLM is unavailable.

        Args:
            text (str): Input text.
            sample_id (str): Sample identifier.

        Returns:
            List[ExtractedClaim]: List of fallback claims.
        """
        sents = self.sentence_extractor.extract(text, sample_id)
        claims = []
        for idx, s in enumerate(sents):
            # Split on coordinating conjunctions (e.g., " and ", "; ") if long
            parts = re.split(r";\s+|\s+and\s+(?=[A-Z0-9])", s.text)
            for part in parts:
                cleaned = part.strip()
                if cleaned:
                    if not cleaned.endswith("."):
                        cleaned += "."
                    claims.append(
                        ExtractedClaim(
                            sample_id=sample_id,
                            claim_index=len(claims),
                            claim_text=cleaned,
                            source_sentence=s.text,
                        )
                    )
        return claims

    def _parse_llm_output(self, raw_text: str) -> List[str]:
        """Parse raw LLM string output into a list of claim strings.

        Args:
            raw_text (str): Output from LLM.

        Returns:
            List[str]: Parsed claim statements.
        """
        cleaned = raw_text.strip()
        # Strip markdown fences if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return [str(c).strip() for c in parsed if str(c).strip()]
            if isinstance(parsed, dict) and "claims" in parsed:
                return [str(c).strip() for c in parsed["claims"] if str(c).strip()]
        except json.JSONDecodeError:
            # Fallback regex extraction for json array items
            matches = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', cleaned)
            if matches:
                return [m.strip() for m in matches if len(m.strip()) > 3]

        return []

    def extract(
        self, text: str, sample_id: str = "sample", overwrite_cache: bool = False
    ) -> List[ExtractedClaim]:
        """Extract atomic claims from text.

        Args:
            text (str): Input text response.
            sample_id (str): Sample identifier.
            overwrite_cache (bool): If True, bypass cache.

        Returns:
            List[ExtractedClaim]: List of validated ExtractedClaim instances.
        """
        if not text or not text.strip():
            return []

        cache_payload = {
            "task": "claim_extraction",
            "prompt_version": self.prompt_version,
            "text": text.strip(),
        }
        cache_key = DiskCache.compute_key(cache_payload)

        if not overwrite_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None and isinstance(cached_data, list):
                return [
                    ExtractedClaim(
                        sample_id=sample_id,
                        claim_index=idx,
                        claim_text=c,
                    )
                    for idx, c in enumerate(cached_data)
                ]

        if not self.client:
            return self._fallback_extract(text, sample_id)

        prompt = CLAIM_EXTRACTION_PROMPT.format(text=text.strip())
        claims_list: List[str] = []
        max_retries = 3

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                raw_output = response.text if response.text else ""
                claims_list = self._parse_llm_output(raw_output)
                time.sleep(2.0)  # Gentle rate pacing (15 RPM limit)
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait_time = 12.0 * attempt
                    print(f" [RateLimit 429] Sample {sample_id}: Backing off {wait_time:.0f}s (Attempt {attempt}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"[ClaimExtractor Warning] LLM call failed for sample {sample_id}: {e}. Using fallback.")
                    return self._fallback_extract(text, sample_id)

        if not claims_list:
            return self._fallback_extract(text, sample_id)

        # Cache valid list
        self.cache.set(cache_key, claims_list, metadata={"sample_id": sample_id})

        return [
            ExtractedClaim(
                sample_id=sample_id,
                claim_index=idx,
                claim_text=c,
            )
            for idx, c in enumerate(claims_list)
        ]
