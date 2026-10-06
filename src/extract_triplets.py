"""
@file extract_triplets.py
@description Structured (subject, relation, object) knowledge triplet extraction with disk caching and verbalizer
@module src/extract_triplets
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from google import genai
from src.cache import DiskCache
from src.config import AppConfig, load_config
from src.extract_sentences import SentenceExtractor
from src.models import ClaimTriplet


TRIPLET_EXTRACTION_PROMPT = """You are an expert NLP system that extracts knowledge triplets from text.
Deconstruct the following text into atomic factual triplets of the form: [Subject, Relation, Object].

Guidelines:
1. Each triplet must represent one precise factual proposition asserted in the text.
2. Subject and Object must be explicit named entities, concepts, or values (no pronouns like 'it', 'he', 'they').
3. Relation should be a clear relational verb or predicate phrase (e.g., 'is located in', 'was completed in', 'wrote', 'is').
4. Do NOT hallucinate facts not present in the input text.

Input Text:
"{text}"

Output Format:
Return ONLY a valid JSON array of objects with keys "subject", "relation", "object".
No markdown fences, no explanatory text.
Example:
[
  {{"subject": "The Eiffel Tower", "relation": "was completed in", "object": "1889"}},
  {{"subject": "The Eiffel Tower", "relation": "is located in", "object": "London"}}
]
"""


class TripletExtractor:
    """Extracts structured [subject, relation, object] triplets from response text."""

    def __init__(self, config: Optional[AppConfig] = None):
        """Initialize TripletExtractor with config, Gemini client, and cache.

        Args:
            config (Optional[AppConfig]): Application configuration.
        """
        load_dotenv()
        self.config = config or load_config()
        self.cache = DiskCache(self.config.extractors.cache_file)
        self.sentence_extractor = SentenceExtractor(self.config.extractors.spacy_model)
        self.prompt_version = self.config.extractors.triplet_prompt_version
        self.model_name = self.config.llm.generator_model

        api_key = os.getenv("GEMINI_API_KEY") or self.config.llm.api_key
        self.client = genai.Client(api_key=api_key) if api_key else None

    def _fallback_extract(self, text: str, sample_id: str) -> List[ClaimTriplet]:
        """Heuristic rule-based fallback triplet extraction using spaCy noun chunks & verbs.

        Args:
            text (str): Input text.
            sample_id (str): Sample identifier.

        Returns:
            List[ClaimTriplet]: Extracted fallback triplets.
        """
        sents = self.sentence_extractor.extract(text, sample_id)
        triplets: List[ClaimTriplet] = []

        for sent in sents:
            doc = self.sentence_extractor.nlp(sent.text)
            subj = ""
            verb = ""
            obj = ""

            for token in doc:
                if "subj" in token.dep_:
                    subj = " ".join([t.text for t in token.subtree if not t.is_punct])
                elif token.pos_ == "VERB" or token.dep_ == "ROOT":
                    verb = token.text
                elif "obj" in token.dep_ or "attr" in token.dep_ or token.dep_ == "acomp":
                    obj = " ".join([t.text for t in token.subtree if not t.is_punct])

            if subj and (verb or obj):
                rel = verb or "is"
                target_obj = obj or "true"
                triplets.append(
                    ClaimTriplet(
                        sample_id=sample_id,
                        triplet_index=len(triplets),
                        subject=subj.strip(),
                        relation=rel.strip(),
                        object=target_obj.strip(),
                        verbalized_claim=f"{subj.strip()} {rel.strip()} {target_obj.strip()}.",
                    )
                )
            else:
                # Direct sentence fallback
                triplets.append(
                    ClaimTriplet(
                        sample_id=sample_id,
                        triplet_index=len(triplets),
                        subject=sent.text.strip(),
                        relation="is stated as",
                        object="factual",
                        verbalized_claim=sent.text.strip(),
                    )
                )
        return triplets

    def _parse_llm_output(self, raw_text: str) -> List[Dict[str, str]]:
        """Parse and sanitize raw LLM JSON response into structured triplet dicts.

        Args:
            raw_text (str): Output from LLM.

        Returns:
            List[Dict[str, str]]: List of triplet dictionaries with subject, relation, object.
        """
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                valid_triplets = []
                for item in parsed:
                    if isinstance(item, dict) and "subject" in item and "relation" in item and "object" in item:
                        valid_triplets.append({
                            "subject": str(item["subject"]).strip(),
                            "relation": str(item["relation"]).strip(),
                            "object": str(item["object"]).strip(),
                        })
                return valid_triplets
            if isinstance(parsed, dict) and "triplets" in parsed:
                return [
                    {
                        "subject": str(t.get("subject", "")).strip(),
                        "relation": str(t.get("relation", "")).strip(),
                        "object": str(t.get("object", "")).strip(),
                    }
                    for t in parsed["triplets"]
                    if t.get("subject") and t.get("relation") and t.get("object")
                ]
        except json.JSONDecodeError:
            pass

        return []

    def extract(
        self, text: str, sample_id: str = "sample", overwrite_cache: bool = False
    ) -> List[ClaimTriplet]:
        """Extract structured claim triplets from text and verbalize them.

        Args:
            text (str): Input text response.
            sample_id (str): Sample identifier.
            overwrite_cache (bool): If True, bypass cache.

        Returns:
            List[ClaimTriplet]: List of validated ClaimTriplet instances.
        """
        if not text or not text.strip():
            return []

        cache_payload = {
            "task": "triplet_extraction",
            "prompt_version": self.prompt_version,
            "text": text.strip(),
        }
        cache_key = DiskCache.compute_key(cache_payload)

        if not overwrite_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None and isinstance(cached_data, list):
                return [
                    ClaimTriplet(
                        sample_id=sample_id,
                        triplet_index=idx,
                        subject=item["subject"],
                        relation=item["relation"],
                        object=item["object"],
                        verbalized_claim=f"{item['subject']} {item['relation']} {item['object']}.",
                    )
                    for idx, item in enumerate(cached_data)
                ]

        if not self.client:
            return self._fallback_extract(text, sample_id)

        prompt = TRIPLET_EXTRACTION_PROMPT.format(text=text.strip())
        triplets_raw: List[Dict[str, str]] = []
        max_retries = 3

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                raw_output = response.text if response.text else ""
                triplets_raw = self._parse_llm_output(raw_output)
                time.sleep(2.0)  # Gentle rate pacing (15 RPM limit)
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait_time = 12.0 * attempt
                    print(f" [RateLimit 429] Sample {sample_id}: Backing off {wait_time:.0f}s (Attempt {attempt}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"[TripletExtractor Warning] LLM call failed for sample {sample_id}: {e}. Using fallback.")
                    return self._fallback_extract(text, sample_id)

        if not triplets_raw:
            return self._fallback_extract(text, sample_id)

        # Cache valid list
        self.cache.set(cache_key, triplets_raw, metadata={"sample_id": sample_id})

        return [
            ClaimTriplet(
                sample_id=sample_id,
                triplet_index=idx,
                subject=item["subject"],
                relation=item["relation"],
                object=item["object"],
                verbalized_claim=f"{item['subject']} {item['relation']} {item['object']}.",
            )
            for idx, item in enumerate(triplets_raw)
        ]
