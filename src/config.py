"""
@file config.py
@description Central configuration schema and loader using Pydantic and PyYAML
@module src/config
"""

import os
from pathlib import Path
from typing import List, Literal
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "factsieve"
    version: str = "1.0.0"
    random_seed: int = 42


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    generated_dir: str = "data/generated"
    benchmark_file: str = "data/processed/benchmark_subset.jsonl"
    samples_per_setting: int = 20
    settings: List[str] = Field(default_factory=lambda: ["zero_context", "noisy_context", "accurate_context"])


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    generator_model: str = "gemini-2.5-flash"
    judge_model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    max_output_tokens: int = 1024
    cache_file: str = "data/generated/llm_cache.jsonl"
    api_key: str = Field(default="")


class ExtractorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spacy_model: str = "en_core_web_sm"
    claim_prompt_version: str = "v1"
    triplet_prompt_version: str = "v1"
    cache_file: str = "data/generated/extraction_cache.jsonl"


class NLIConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    batch_size: int = 16
    max_length: int = 512
    device: str = "auto"


class SemanticSimilarityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    threshold: float = 0.75
    batch_size: int = 32
    device: str = "auto"


class LLMJudgeVerifierConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt_version: str = "v1"
    cache_file: str = "data/generated/judge_cache.jsonl"


class VerifiersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nli: NLIConfig = Field(default_factory=NLIConfig)
    semantic_similarity: SemanticSimilarityConfig = Field(default_factory=SemanticSimilarityConfig)
    llm_judge: LLMJudgeVerifierConfig = Field(default_factory=LLMJudgeVerifierConfig)


class ScoringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    aggregation_rule: Literal["strict", "soft"] = "strict"
    labels: List[str] = Field(default_factory=lambda: ["Entailment", "Neutral", "Contradiction"])


class ResultsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plots_dir: str = "results/plots"
    tables_dir: str = "results/tables"


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    extractors: ExtractorConfig = Field(default_factory=ExtractorConfig)
    verifiers: VerifiersConfig = Field(default_factory=VerifiersConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    results: ResultsConfig = Field(default_factory=ResultsConfig)


def load_config(config_path: str = "configs/config.yaml") -> AppConfig:
    """Load configuration from YAML file and blend with environment variables.

    Args:
        config_path (str): Relative or absolute path to YAML config file.

    Returns:
        AppConfig: Validated application configuration instance.
    """
    # Load .env file if present
    load_dotenv()

    config_file = Path(config_path)
    config_dict = {}
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}

    # Read environment overrides
    api_key = os.getenv("GEMINI_API_KEY", "")
    if "llm" in config_dict:
        config_dict["llm"]["api_key"] = api_key
    else:
        config_dict["llm"] = {"api_key": api_key}

    return AppConfig(**config_dict)
