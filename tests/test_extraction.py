"""
@file test_extraction.py
@description Unit tests for Sentence, Claim, and Triplet extraction engines
@module tests/test_extraction
"""

import pytest
from src.extract_sentences import SentenceExtractor
from src.extract_claims import ClaimExtractor
from src.extract_triplets import TripletExtractor
from src.models import ClaimTriplet, ExtractedClaim, ExtractedSentence


def test_sentence_extractor_basic():
    """Verify standard multi-sentence splitting and bullet cleaning."""
    extractor = SentenceExtractor()
    text = "1. The Eiffel Tower is in Paris. It was completed in 1889. - Another fact."
    sentences = extractor.extract(text, sample_id="s1")

    assert len(sentences) >= 2
    assert all(isinstance(s, ExtractedSentence) for s in sentences)
    assert "The Eiffel Tower is in Paris." in [s.text for s in sentences]


def test_sentence_extractor_empty():
    """Verify graceful handling of empty or whitespace text."""
    extractor = SentenceExtractor()
    assert extractor.extract("", sample_id="s1") == []
    assert extractor.extract("   ", sample_id="s1") == []


def test_claim_extractor_parsing():
    """Verify parsing of valid JSON, markdown fences, and fallback extraction."""
    extractor = ClaimExtractor()
    
    # Valid JSON array
    raw_json = '["The Eiffel Tower was built in 1889.", "It is located in Paris."]'
    parsed = extractor._parse_llm_output(raw_json)
    assert len(parsed) == 2
    assert parsed[0] == "The Eiffel Tower was built in 1889."

    # Markdown fenced JSON
    fenced_json = '```json\n["Fact A.", "Fact B."]\n```'
    parsed_fenced = extractor._parse_llm_output(fenced_json)
    assert len(parsed_fenced) == 2
    assert parsed_fenced[1] == "Fact B."


def test_claim_extractor_fallback():
    """Verify rule-based fallback extraction when LLM is unavailable."""
    extractor = ClaimExtractor()
    text = "Water boils at 100 degrees Celsius and freezes at 0 degrees Celsius."
    claims = extractor._fallback_extract(text, sample_id="s_fallback")

    assert len(claims) >= 1
    assert all(isinstance(c, ExtractedClaim) for c in claims)
    assert claims[0].sample_id == "s_fallback"


def test_triplet_extractor_parsing():
    """Verify parsing of triplet JSON structures."""
    extractor = TripletExtractor()
    raw_json = """
    [
        {"subject": "Eiffel Tower", "relation": "located in", "object": "Paris"},
        {"subject": "Eiffel Tower", "relation": "completed in", "object": "1889"}
    ]
    """
    triplets = extractor._parse_llm_output(raw_json)
    assert len(triplets) == 2
    assert triplets[0]["subject"] == "Eiffel Tower"
    assert triplets[0]["relation"] == "located in"
    assert triplets[0]["object"] == "Paris"


def test_triplet_verbalization():
    """Verify verbalization generates coherent sentence representation."""
    triplet = ClaimTriplet(
        sample_id="test_01",
        triplet_index=0,
        subject="Alexander Fleming",
        relation="discovered",
        object="penicillin in 1928",
    )
    verbalized = triplet.get_verbalized()
    assert verbalized == "Alexander Fleming discovered penicillin in 1928."


def test_triplet_extractor_fallback():
    """Verify heuristic fallback triplet extraction produces non-empty triplets."""
    extractor = TripletExtractor()
    text = "The Titanic sank in 1912."
    triplets = extractor._fallback_extract(text, sample_id="s_titanic")

    assert len(triplets) >= 1
    assert all(isinstance(t, ClaimTriplet) for t in triplets)
    assert triplets[0].sample_id == "s_titanic"
    assert len(triplets[0].get_verbalized()) > 5
