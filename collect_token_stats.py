#!/usr/bin/env python3
"""
Collect token sampling statistics without training.
This script runs the tokenizer sampling process and tracks token frequencies.
"""

import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import copy
import numpy as np

from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)

from zett.model import HypernetArgs
from zett.collator import Collator
from zett.dataset import TrainDataset
from zett.tokenizer_converters import convert_to_byte_level

# Import the argument classes from train.py
sys.path.insert(0, os.path.dirname(__file__))
from train import ModelArguments, DataArguments, TrainingArguments

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, HypernetArgs)
    )

    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, hn_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        print("Usage: python collect_token_stats.py <config.json>")
        print("Additional arguments:")
        print("  --num_batches N    Number of batches to sample (default: 1000)")
        print("  --output_file PATH Output JSON file (default: token_frequencies.json)")
        sys.exit(1)

    # Parse additional arguments
    num_batches = 30000
    output_file = "token_frequencies.json"

    if "--num_batches" in sys.argv:
        idx = sys.argv.index("--num_batches")
        num_batches = int(sys.argv[idx + 1])

    if "--output_file" in sys.argv:
        idx = sys.argv.index("--output_file")
        output_file = sys.argv[idx + 1]

    logger.info(f"Collecting token statistics for {num_batches} batches")
    logger.info(f"Output will be saved to: {output_file}")

    # Parse language configuration
    lang_probs = None
    if data_args.langs.endswith(".txt"):
        lang_data = [x.strip() for x in open(data_args.langs).readlines()]
        if "," in lang_data[0]:
            langs = [x.split(",")[0].strip() for x in lang_data]
            lang_probs = np.array([float(x.split(",")[1].strip()) for x in lang_data])
            lang_probs = lang_probs / lang_probs.sum()
        else:
            langs = lang_data
    else:
        langs = data_args.langs.split(" ")

    # Filter to only code languages (skip 'en' if data not prepared)
    logger.info(f"Original languages: {langs}")
    code_langs = [lang for lang in langs if lang != 'en']
    logger.info(f"Filtering to code languages only: {code_langs}")

    if len(code_langs) == 0:
        logger.error("No code languages found!")
        sys.exit(1)

    # Adjust language probabilities if needed
    if lang_probs is not None and len(code_langs) < len(langs):
        # Find indices of code languages
        code_indices = [i for i, lang in enumerate(langs) if lang in code_langs]
        lang_probs = lang_probs[code_indices]
        lang_probs = lang_probs / lang_probs.sum()
        logger.info(f"Adjusted language probabilities: {dict(zip(code_langs, lang_probs))}")

    data_args.langs = code_langs

    if len(data_args.langs) == 1:
        lang_probs = np.array([1.0])

    set_seed(training_args.seed)

    # Add hn_surface_maxlen to data_args (required by collator)
    data_args.hn_surface_maxlen = hn_args.hn_surface_maxlen

    # Load tokenizer configuration
    reference = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path
    )

    if reference.pad_token is None:
        reference.pad_token = reference.eos_token

    # Set up hypernetwork tokenizer
    if hn_args.hn_embed_using_source_embeddings:
        hn_tokenizer = copy.deepcopy(reference)
    elif hn_args.hn_model_name_or_path is not None:
        hn_tokenizer = AutoTokenizer.from_pretrained(hn_args.hn_model_name_or_path)
    else:
        hn_tokenizer = None

    if hn_tokenizer is not None:
        hn_tokenizer, _ = convert_to_byte_level(hn_tokenizer)

    # Create datasets
    logger.info("Creating datasets...")
    train_batch_size = training_args.train_batch_size

    train_datasets = [
        TrainDataset(
            [lang],
            data_args.train_directory,
            np.array([1.0]),
            train_batch_size,
            data_args.block_size,
            do_sequence_packing=data_args.do_sequence_packing,
            eos_token=reference.eos_token,
        )
        for lang in data_args.langs
    ]

    initial_texts = {
        lang_code: dset.get_texts(data_args.tokenizer_batch_size)
        for lang_code, dset in zip(data_args.langs, train_datasets)
    }

    # Create collators
    logger.info("Creating collators...")
    train_collators = [
        Collator(
            reference,
            hn_tokenizer,
            data_args,
            batch_size=train_batch_size,
            tokenizer_name=data_args.target_tokenizer_name,
            initial_texts={lang_code: initial_texts[lang_code]},
            with_consistent_whitespace=not data_args.use_passthrough_hypernet,
        )
        for lang_code in data_args.langs
    ]

    # Dictionary to track token frequencies
    token_frequencies = defaultdict(int)

    # Collect statistics
    logger.info(f"Collecting token statistics from {num_batches} batches...")

    for batch_idx in tqdm(range(num_batches)):
        # Sample a language
        lang_idx = np.random.choice(len(data_args.langs), p=lang_probs)
        lang_code = data_args.langs[lang_idx]

        # Get batch from dataset
        dataset = train_datasets[lang_idx]
        collator = train_collators[lang_idx]

        # Sample texts
        texts = dataset.get_texts(train_batch_size)

        # Sample tokenizer and get tokens
        try:
            # The collator's sample_tokenizer method returns 5 elements:
            # (tokenizer, special_ids_map, surface_forms, priors, byte_lengths)
            tokenizer, _, _, _, _ = collator.sample_tokenizer(
                texts, collator.samplers[lang_code][0]
            )

            # Extract all tokens from the sampled tokenizer vocabulary
            tokens = tokenizer.convert_ids_to_tokens(range(len(tokenizer)))

            # Update frequency dictionary
            for token in tokens:
                token_frequencies[token] += 1

        except Exception as e:
            logger.warning(f"Error at batch {batch_idx} for language {lang_code}: {e}")
            continue

    # Sort by frequency (descending)
    sorted_frequencies = dict(
        sorted(token_frequencies.items(), key=lambda x: x[1], reverse=True)
    )

    # Save to JSON
    logger.info(f"Saving {len(sorted_frequencies)} unique tokens to {output_file}")

    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_frequencies, f, ensure_ascii=False, indent=2)

    # Print statistics
    total_tokens = sum(sorted_frequencies.values())
    unique_tokens = len(sorted_frequencies)

    logger.info("=" * 60)
    logger.info("TOKEN SAMPLING STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total batches processed: {num_batches}")
    logger.info(f"Total token occurrences: {total_tokens}")
    logger.info(f"Unique tokens sampled: {unique_tokens}")
    logger.info(f"Average tokens per batch: {total_tokens / num_batches:.2f}")
    logger.info(f"Languages: {', '.join(data_args.langs)}")
    logger.info("=" * 60)

    # Show top 20 most frequent tokens
    logger.info("\nTop 20 most frequent tokens:")
    for i, (token, freq) in enumerate(list(sorted_frequencies.items())[:20], 1):
        # Escape special characters for display
        display_token = repr(token)
        logger.info(f"{i:2d}. {display_token:30s} : {freq:6d} ({freq/total_tokens*100:.2f}%)")

    logger.info(f"\nFull results saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
