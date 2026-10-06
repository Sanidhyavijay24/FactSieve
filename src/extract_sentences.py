"""
@file extract_sentences.py
@description Sentence segmentation module using spaCy with fallback heuristics
@module src/extract_sentences
"""

import re
from typing import List, Optional
import spacy
from src.models import ExtractedSentence


class SentenceExtractor:
    """Extracts clean sentence-level units from response text using spaCy."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Initialize SentenceExtractor with specified spaCy model.

        Args:
            spacy_model (str): Name of spaCy language model.
        """
        try:
            self.nlp = spacy.load(spacy_model)
        except Exception:
            # Fallback to a blank English model if the pipeline is unavailable
            self.nlp = spacy.blank("en")
            self.nlp.add_pipe("sentencizer")

    def extract(self, text: str, sample_id: str = "sample") -> List[ExtractedSentence]:
        """Split text into cleaned, non-empty ExtractedSentence objects.

        Args:
            text (str): Input response string.
            sample_id (str): Identifier of the parent sample.

        Returns:
            List[ExtractedSentence]: List of extracted sentences.
        """
        if not text or not text.strip():
            return []

        doc = self.nlp(text.strip())
        sentences: List[ExtractedSentence] = []
        sent_idx = 0

        for sent in doc.sents:
            cleaned = sent.text.strip()
            # Clean bullet points and leading numbering
            cleaned = re.sub(r"^(\d+[\.\)]|\-|\*)\s+", "", cleaned).strip()
            if cleaned and len(cleaned) > 2:
                sentences.append(
                    ExtractedSentence(
                        sample_id=sample_id,
                        sentence_index=sent_idx,
                        text=cleaned,
                    )
                )
                sent_idx += 1

        # Fallback if no sentences extracted (e.g., single fragment without punctuation)
        if not sentences and text.strip():
            sentences.append(
                ExtractedSentence(
                    sample_id=sample_id,
                    sentence_index=0,
                    text=text.strip(),
                )
            )

        return sentences
