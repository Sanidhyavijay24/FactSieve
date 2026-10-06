"""
@file generate_responses.py
@description Deterministic Gemini response generator with zero-waste local disk caching and rate limiting
@module src/generate_responses
"""

import argparse
import os
import time
from typing import List, Optional
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError
from src.cache import DiskCache
from src.config import AppConfig, load_config
from src.data_loader import create_default_benchmark_dataset, load_benchmark_samples, save_benchmark_samples
from src.models import BenchmarkSample


class ResponseGenerator:
    """Handles LLM response generation with strict local caching and free-tier rate pacing."""

    def __init__(self, config: Optional[AppConfig] = None):
        """Initialize ResponseGenerator with configuration and disk cache.

        Args:
            config (Optional[AppConfig]): Validated application config.
        """
        load_dotenv()
        self.config = config or load_config()
        self.cache = DiskCache(self.config.llm.cache_file)
        
        api_key = os.getenv("GEMINI_API_KEY") or self.config.llm.api_key
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or configuration.")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = self.config.llm.generator_model
        self.prompt_version = "v1"

    def format_prompt(self, sample: BenchmarkSample) -> str:
        """Construct structured prompt for LLM based on benchmark setting.

        Args:
            sample (BenchmarkSample): Benchmark test instance.

        Returns:
            str: Formatted prompt string.
        """
        if sample.context and len(sample.context) > 0:
            joined_context = "\n".join(sample.context)
            return (
                f"Context Information:\n{joined_context}\n\n"
                f"Question: {sample.question}\n\n"
                "Please provide a direct, concise, and factual answer to the question based on the provided context."
            )
        else:
            return (
                f"Question: {sample.question}\n\n"
                "Please provide a direct, concise, and factual answer to the question."
            )

    def generate(self, sample: BenchmarkSample, overwrite_cache: bool = False, max_retries: int = 3) -> str:
        """Generate or retrieve cached LLM response for a benchmark sample.

        Args:
            sample (BenchmarkSample): Benchmark test instance.
            overwrite_cache (bool): If True, bypass cache and re-query the API.
            max_retries (int): Maximum number of retry attempts upon rate limit.

        Returns:
            str: Generated text response.
        """
        prompt = self.format_prompt(sample)
        cache_payload = {
            "model": self.model_name,
            "prompt_version": self.prompt_version,
            "temperature": self.config.llm.temperature,
            "sample_id": sample.id,
            "prompt": prompt,
        }
        cache_key = DiskCache.compute_key(cache_payload)

        if not overwrite_cache:
            cached_val = self.cache.get(cache_key)
            if cached_val is not None:
                return str(cached_val)

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                response_text = response.text.strip() if response.text else ""

                # Store in cache
                self.cache.set(
                    key=cache_key,
                    value=response_text,
                    metadata={"sample_id": sample.id, "setting": sample.setting},
                )
                # Rate pacing sleep to preserve free quota
                time.sleep(4.0)
                return response_text

            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait_time = 15.0 * attempt
                    print(f"[RateLimit] 429 encountered for sample {sample.id}. Backing off for {wait_time}s (Attempt {attempt}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"[API Error] Failed to generate for sample {sample.id}: {e}")
                    # Fallback to existing sample response if available
                    if sample.response:
                        return sample.response
                    raise e

        # If all retries exhausted, fallback to existing sample response
        if sample.response:
            return sample.response
        raise RuntimeError(f"Exhausted retries for sample {sample.id}")

    def generate_all(
        self, samples: List[BenchmarkSample], overwrite_cache: bool = False
    ) -> List[BenchmarkSample]:
        """Generate responses for all samples in the list and update instances.

        Args:
            samples (List[BenchmarkSample]): List of samples to process.
            overwrite_cache (bool): If True, force re-generation.

        Returns:
            List[BenchmarkSample]: Updated benchmark samples containing responses.
        """
        updated_samples: List[BenchmarkSample] = []
        for i, sample in enumerate(samples, start=1):
            print(f"[{i}/{len(samples)}] Processing sample {sample.id} ({sample.setting})...")
            resp_text = self.generate(sample, overwrite_cache=overwrite_cache)
            sample_dict = sample.model_dump()
            sample_dict["generator"] = self.model_name
            sample_dict["response"] = resp_text
            updated_samples.append(BenchmarkSample(**sample_dict))
        return updated_samples


def main():
    """CLI runner to generate responses for the benchmark subset."""
    parser = argparse.ArgumentParser(description="Generate and cache LLM responses for RefChecker benchmark")
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite cached responses")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    benchmark_file = cfg.data.benchmark_file

    try:
        samples = load_benchmark_samples(benchmark_file)
    except FileNotFoundError:
        print(f"Benchmark file not found at {benchmark_file}. Creating default dataset...")
        samples = create_default_benchmark_dataset(benchmark_file)

    print(f"Loaded {len(samples)} benchmark samples.")
    generator = ResponseGenerator(cfg)
    print(f"Generating / loading responses using model: {cfg.llm.generator_model}...")

    updated_samples = generator.generate_all(samples, overwrite_cache=args.overwrite)
    save_benchmark_samples(updated_samples, benchmark_file)
    print(f"Successfully saved {len(updated_samples)} updated samples to {benchmark_file}.")
    print(f"Total entries in cache: {len(generator.cache)}")


if __name__ == "__main__":
    main()
