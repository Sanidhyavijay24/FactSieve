"""
@file data_loader.py
@description Benchmark data loader and dataset curator across Zero, Noisy, and Accurate Context settings
@module src/data_loader
"""

import json
from pathlib import Path
from typing import List, Optional
from src.models import BenchmarkSample


def create_default_benchmark_dataset(
    output_path: str = "data/processed/benchmark_subset.jsonl",
    samples_per_setting: int = 10,
) -> List[BenchmarkSample]:
    """Creates and persists a curated benchmark dataset covering Zero, Noisy, and Accurate Context settings.

    Each setting includes ground-truth reference texts, context passages (or distractors),
    and gold evaluation targets for both factual and hallucinated responses.

    Args:
        output_path (str): Filepath to save the benchmark dataset JSONL.
        samples_per_setting (int): Target number of samples per setting.

    Returns:
        List[BenchmarkSample]: The curated list of benchmark samples.
    """
    raw_samples = [
        # ==========================================
        # 1. ACCURATE CONTEXT SETTING (Dolly-15k style)
        # ==========================================
        {
            "id": "acc_001",
            "question": "When was the Eiffel Tower completed and where is it located?",
            "context": [
                "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
                "Constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair, it was designed by Gustave Eiffel."
            ],
            "reference": "The Eiffel Tower is located on the Champ de Mars in Paris, France, and was completed in 1889.",
            "setting": "accurate_context",
            "response": "The Eiffel Tower was completed in 1889 and is located in Paris, France.",
            "ground_truth_label": "Entailment",
        },
        {
            "id": "acc_002",
            "question": "What is the capital and largest city of Australia?",
            "context": [
                "Canberra is the capital city of Australia, founded following the federation of the colonies of Australia. "
                "Sydney is the most populous city in Australia and the capital of New South Wales."
            ],
            "reference": "Canberra is the capital of Australia, while Sydney is the largest city.",
            "setting": "accurate_context",
            "response": "Sydney is the capital city of Australia and its largest city.",
            "ground_truth_label": "Contradiction",
        },
        {
            "id": "acc_003",
            "question": "Who developed the theory of general relativity and when was it published?",
            "context": [
                "Albert Einstein presented his theory of general relativity to the Prussian Academy of Sciences in November 1915."
            ],
            "reference": "Albert Einstein developed general relativity and published it in November 1915.",
            "setting": "accurate_context",
            "response": "Albert Einstein published the theory of general relativity in 1915.",
            "ground_truth_label": "Entailment",
        },
        {
            "id": "acc_004",
            "question": "What is the primary chemical element in diamonds and graphite?",
            "context": [
                "Both diamond and graphite are allotropes of pure carbon. Diamond has a tetrahedral crystal structure."
            ],
            "reference": "The primary chemical element in both diamonds and graphite is carbon.",
            "setting": "accurate_context",
            "response": "Diamonds and graphite are allotropes composed entirely of carbon.",
            "ground_truth_label": "Entailment",
        },
        {
            "id": "acc_005",
            "question": "Who was the first person to step on the Moon and during which mission?",
            "context": [
                "On July 20, 1969, Neil Armstrong became the first human to step onto the lunar surface during the Apollo 11 mission. "
                "Buzz Aldrin joined him 19 minutes later."
            ],
            "reference": "Neil Armstrong was the first person to step on the Moon during the Apollo 11 mission in 1969.",
            "setting": "accurate_context",
            "response": "Buzz Aldrin was the first person to walk on the Moon during Apollo 13.",
            "ground_truth_label": "Contradiction",
        },
        {
            "id": "acc_006",
            "question": "What is the boiling point of water at standard sea level atmospheric pressure?",
            "context": [
                "At standard atmospheric pressure (1 atm or 101.3 kPa), pure water boils at 100 degrees Celsius (212 degrees Fahrenheit)."
            ],
            "reference": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
            "setting": "accurate_context",
            "response": "At standard sea level pressure, water boils at 100 degrees Celsius.",
            "ground_truth_label": "Entailment",
        },
        {
            "id": "acc_007",
            "question": "Which organ in the human body produces insulin?",
            "context": [
                "Insulin is a peptide hormone produced by the beta cells of the pancreatic islets in the pancreas."
            ],
            "reference": "The pancreas produces insulin.",
            "setting": "accurate_context",
            "response": "Insulin is synthesized and secreted by the liver in humans.",
            "ground_truth_label": "Contradiction",
        },
        {
            "id": "acc_008",
            "question": "In what year did the Titanic sink and where was it traveling to?",
            "context": [
                "RMS Titanic sank in the North Atlantic Ocean on 15 April 1912 after colliding with an iceberg during her maiden voyage from Southampton to New York City."
            ],
            "reference": "The Titanic sank on April 15, 1912, while traveling to New York City.",
            "setting": "accurate_context",
            "response": "The Titanic sank in 1912 on its maiden voyage heading to New York City.",
            "ground_truth_label": "Entailment",
        },
        {
            "id": "acc_009",
            "question": "Who wrote the play 'Romeo and Juliet'?",
            "context": [
                "Romeo and Juliet is a tragedy written by William Shakespeare early in his career about two Italian star-crossed lovers."
            ],
            "reference": "William Shakespeare wrote Romeo and Juliet.",
            "setting": "accurate_context",
            "response": "The play Romeo and Juliet was written by Christopher Marlowe in 1595.",
            "ground_truth_label": "Contradiction",
        },
        {
            "id": "acc_010",
            "question": "What is the speed of light in vacuum?",
            "context": [
                "The speed of light in vacuum, commonly denoted c, is a universal physical constant exactly equal to 299,792,458 metres per second."
            ],
            "reference": "The speed of light in vacuum is approximately 299,792,458 meters per second.",
            "setting": "accurate_context",
            "response": "The speed of light in vacuum is exactly 299,792,458 meters per second.",
            "ground_truth_label": "Entailment",
        },

        # ==========================================
        # 2. NOISY CONTEXT SETTING (MS MARCO style: context with distractors)
        # ==========================================
        {
            "id": "noisy_001",
            "question": "What is the chemical formula for table salt?",
            "context": [
                "Distractor 1: Baking soda, known chemically as sodium bicarbonate (NaHCO3), is used for cooking and cleaning.",
                "Target Passage: Sodium chloride, commonly known as table salt, is an ionic compound with the chemical formula NaCl.",
                "Distractor 2: Potassium chloride (KCl) is a metal halide salt composed of potassium and chlorine used in fertilizer."
            ],
            "reference": "The chemical formula for table salt (sodium chloride) is NaCl.",
            "setting": "noisy_context",
            "response": "Table salt has the chemical formula NaCl.",
            "ground_truth_label": "Entailment",
        },
        {
            "id": "noisy_002",
            "question": "Who invented the telephone?",
            "context": [
                "Distractor 1: Thomas Edison invented the phonograph and developed the practical incandescent light bulb.",
                "Target Passage: Alexander Graham Bell was awarded the first US patent for the telephone in 1876.",
                "Distractor 2: Guglielmo Marconi is known for his pioneering work on long-distance radio transmission."
            ],
            "reference": "Alexander Graham Bell was awarded the patent for the invention of the telephone.",
            "setting": "noisy_context",
            "response": "Thomas Edison invented the telephone and patented it in 1876.",
            "ground_truth_label": "Contradiction",
        },
        {
            "id": "noisy_003",
            "question": "What is the longest river in the world?",
            "context": [
                "Target Passage: The Nile River in northeastern Africa is traditionally considered the longest river in the world, stretching approximately 6,650 km.",
                "Distractor 1: The Amazon River in South America is the largest river by discharge volume of water in the world.",
                "Distractor 2: The Yangtze River is the longest river in Asia and the third longest in the world."
            ],
            "reference": "The Nile River is traditionally recognized as the longest river in the world.",
            "setting": "noisy_context",
            "response": "The Nile River is recognized as the longest river in the world.",
            "ground_truth_label": "Entailment",
        },
        {
            "id": "noisy_004",
            "question": "What planet is known as the Red Planet?",
            "context": [
                "Distractor 1: Venus is often called Earth's twin due to its similar size and mass, but has an extreme greenhouse effect.",
                "Distractor 2: Jupiter is the largest planet in our solar system and features the Great Red Spot storm.",
                "Target Passage: Mars is the fourth planet from the Sun, often referred to as the Red Planet due to iron oxide on its surface."
            ],
            "reference": "Mars is known as the Red Planet due to iron oxide on its surface.",
            "setting": "noisy_context",
            "response": "Jupiter is known as the Red Planet because of its prominent iron oxide surface.",
            "ground_truth_label": "Contradiction",
        },
        {
            "id": "noisy_005",
            "question": "What gas do plants absorb during photosynthesis?",
            "context": [
                "Target Passage: During photosynthesis, green plants absorb carbon dioxide from the atmosphere and release oxygen.",
                "Distractor 1: Plants undergo cellular respiration where they consume oxygen and release carbon dioxide.",
                "Distractor 2: Nitrogen makes up roughly 78% of Earth's atmosphere and is fixed by soil bacteria."
            ],
            "reference": "Plants absorb carbon dioxide during photosynthesis.",
            "setting": "noisy_context",
            "response": "Plants take in carbon dioxide during photosynthesis and release oxygen.",
            "ground_truth_label": "Entailment",
        },
        {
            "id": "noisy_006",
            "question": "What year did World War II end?",
            "context": [
                "Distractor 1: World War I concluded on November 11, 1918, with the signing of the armistice.",
                "Target Passage: World War II ended in 1945 with the unconditional surrender of the Axis powers.",
                "Distractor 2: The United Nations was officially founded on October 24, 1945, after the war ended."
            ],
            "reference": "World War II ended in 1945.",
            "setting": "noisy_context",
            "response": "World War II came to an end in 1918 following the signing of the armistice.",
            "ground_truth_label": "Contradiction",
        },
        {
            "id": "noisy_007",
            "question": "What is the hardest known natural mineral?",
            "context": [
                "Distractor 1: Corundum has a hardness of 9 on the Mohs scale, second only to diamond.",
                "Target Passage: Diamond is the hardest known natural mineral, rated 10 on the Mohs scale of mineral hardness.",
                "Distractor 2: Quartz ranks 7 on the Mohs hardness scale and is abundant in Earth's continental crust."
            ],
            "reference": "Diamond is the hardest natural mineral on the Mohs scale.",
            "setting": "noisy_context",
            "response": "Diamond is the hardest naturally occurring mineral with a Mohs rating of 10.",
            "ground_truth_label": "Entailment",
        },
        {
            "id": "noisy_008",
            "question": "Where are the ancient Pyramids of Giza located?",
            "context": [
                "Distractor 1: The Mayan pyramids of Chichen Itza are situated on the Yucatan Peninsula in Mexico.",
                "Target Passage: The Giza pyramid complex is located on the Giza Plateau near Cairo, Egypt.",
                "Distractor 2: Ancient Mesopotamian ziggurats were built in modern-day Iraq and Iran."
            ],
            "reference": "The Pyramids of Giza are located near Cairo, Egypt.",
            "setting": "noisy_context",
            "response": "The Pyramids of Giza are ancient architectural structures located in Mexico.",
            "ground_truth_label": "Contradiction",
        },
        {
            "id": "noisy_009",
            "question": "Who discovered penicillin?",
            "context": [
                "Distractor 1: Louis Pasteur invented pasteurization and developed rabies and anthrax vaccines.",
                "Target Passage: Scottish physician Alexander Fleming discovered penicillin in 1928 at St Mary's Hospital, London.",
                "Distractor 2: Edward Jenner developed the smallpox vaccine in 1796."
            ],
            "reference": "Alexander Fleming discovered penicillin in 1928.",
            "setting": "noisy_context",
            "response": "Alexander Fleming discovered penicillin in 1928.",
            "ground_truth_label": "Entailment",
        },
        {
            "id": "noisy_010",
            "question": "What is the smallest prime number?",
            "context": [
                "Distractor 1: 1 is neither prime nor composite by modern mathematical definition.",
                "Target Passage: 2 is the smallest prime number and the only even prime number.",
                "Distractor 2: 3 is the smallest odd prime number."
            ],
            "reference": "2 is the smallest prime number.",
            "setting": "noisy_context",
            "response": "The number 1 is the smallest prime number.",
            "ground_truth_label": "Contradiction",
        },

        # ==========================================
        # 3. ZERO CONTEXT SETTING (NaturalQuestions style: closed-book QA)
        # ==========================================
        {
            "id": "zero_001",
            "question": "Who was the first President of the United States?",
            "context": [],
            "reference": "George Washington was the first President of the United States, serving from 1789 to 1797.",
            "setting": "zero_context",
            "response": "George Washington served as the first President of the United States.",
            "ground_truth_label": "Entailment",
        },
        {
            "id": "zero_002",
            "question": "What is the largest mammal in the world?",
            "context": [],
            "reference": "The blue whale is the largest known mammal and animal to have ever lived.",
            "setting": "zero_context",
            "response": "The African elephant is the largest living mammal on Earth.",
            "ground_truth_label": "Contradiction",
        },
        {
            "id": "zero_003",
            "question": "Which country hosted the 2016 Summer Olympics?",
            "context": [],
            "reference": "Brazil hosted the 2016 Summer Olympic Games in Rio de Janeiro.",
            "setting": "zero_context",
            "response": "The 2016 Summer Olympics were hosted by Brazil in Rio de Janeiro.",
            "ground_truth_label": "Entailment",
        },
        {
            "id": "zero_004",
            "question": "What is the chemical symbol for gold?",
            "context": [],
            "reference": "The chemical symbol for gold is Au, derived from the Latin word aurum.",
            "setting": "zero_context",
            "response": "The chemical symbol for gold is Ag.",
            "ground_truth_label": "Contradiction",
        },
        {
            "id": "zero_005",
            "question": "Who painted the Mona Lisa?",
            "context": [],
            "reference": "Leonardo da Vinci painted the Mona Lisa during the Italian Renaissance.",
            "setting": "zero_context",
            "response": "Leonardo da Vinci painted the Mona Lisa.",
            "ground_truth_label": "Entailment",
        },
        {
            "id": "zero_006",
            "question": "What is the largest ocean on Earth?",
            "context": [],
            "reference": "The Pacific Ocean is the largest and deepest ocean on Earth.",
            "setting": "zero_context",
            "response": "The Atlantic Ocean is the largest and deepest ocean in the world.",
            "ground_truth_label": "Contradiction",
        },
        {
            "id": "zero_007",
            "question": "In what year did man first land on the moon?",
            "context": [],
            "reference": "Humans first landed on the Moon in 1969 during the Apollo 11 mission.",
            "setting": "zero_context",
            "response": "The first human moon landing occurred in 1969.",
            "ground_truth_label": "Entailment",
        },
        {
            "id": "zero_008",
            "question": "What is the capital of Japan?",
            "context": [],
            "reference": "Tokyo is the capital and largest city of Japan.",
            "setting": "zero_context",
            "response": "Kyoto is the official capital city of Japan.",
            "ground_truth_label": "Contradiction",
        },
        {
            "id": "zero_009",
            "question": "How many continents are there on Earth?",
            "context": [],
            "reference": "There are seven continents on Earth: Asia, Africa, North America, South America, Antarctica, Europe, and Australia.",
            "setting": "zero_context",
            "response": "There are seven continents on Earth.",
            "ground_truth_label": "Entailment",
        },
        {
            "id": "zero_010",
            "question": "What currency is used in the United Kingdom?",
            "context": [],
            "reference": "The currency used in the United Kingdom is the British Pound Sterling (GBP).",
            "setting": "zero_context",
            "response": "The official currency of the United Kingdom is the Euro.",
            "ground_truth_label": "Contradiction",
        },
    ]

    samples: List[BenchmarkSample] = [BenchmarkSample(**item) for item in raw_samples]

    # Save to JSONL
    save_benchmark_samples(samples, output_path)
    return samples


def save_benchmark_samples(samples: List[BenchmarkSample], filepath: str) -> None:
    """Save benchmark samples list to JSONL format.

    Args:
        samples (List[BenchmarkSample]): List of validated BenchmarkSample objects.
        filepath (str): Destination file path.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(sample.model_dump_json() + "\n")


def load_benchmark_samples(filepath: str) -> List[BenchmarkSample]:
    """Load benchmark samples from a JSONL file.

    Args:
        filepath (str): Path to JSONL file.

    Returns:
        List[BenchmarkSample]: List of parsed BenchmarkSample instances.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found at: {filepath}")

    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                samples.append(BenchmarkSample(**data))
            except Exception as e:
                raise ValueError(f"Error parsing line {line_num} in {filepath}: {e}")
    return samples
