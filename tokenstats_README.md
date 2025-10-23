# Token Sampling Statistics Collection

This document explains how to collect token sampling statistics from the hypernetwork training process without actually training the model.

## Overview

The `collect_token_stats.py` script allows you to:
- Sample tokenizers across multiple batches (simulating the training process)
- Track which tokens appear in each sampled vocabulary
- Count the frequency of each token across all batches
- Output results to a JSON file sorted by frequency

This is useful for understanding:
- What tokens are being sampled during training
- Token distribution across different programming languages
- Most common substrings in your training corpus

## Prerequisites

1. **Prepared Data**: You must have prepared your training data using `data/prepare_code.py`
2. **Dependencies**: All ZeTT dependencies must be installed (see main README)
3. **Rust Utils**: The `rust_utils` module must be compiled (`cd rust_utils && maturin develop --release`)

## Quick Start

### 1. Prepare Code Data (if not done already)

```bash
python data/prepare_code.py \
    --max_train_pages_per_language 2000000 \
    --out_train_dir /mnt/disks/persist/train \
    --out_valid_dir /mnt/disks/persist/valid \
    --include_langs python javascript java cpp go github-issues-filtered-structured
```

### 2. Run Token Statistics Collection

```bash
# Basic usage (1000 batches, default output file)
python collect_token_stats.py configs/token_stats_code_only.json

# Custom number of batches
python collect_token_stats.py configs/token_stats_code_only.json --num_batches 5000

# Custom output file
python collect_token_stats.py configs/token_stats_code_only.json \
    --num_batches 5000 \
    --output_file my_token_frequencies.json
```

## Configuration Files

### Using the Provided Config

The repository includes `configs/token_stats_code_only.json` which is configured for code-only data:
- Uses all 6 code languages (Python, JavaScript, Java, C++, Go, GitHub issues)
- Same hyperparameters as the full training config
- No English data (since you haven't prepared it)

### Creating a Custom Config

If you want to use different parameters or data paths, create a new config based on `configs/zeroshot/v7:tinyllama_en+code:lw=0.5_long.json`:

```json
{
    "train_directory": "/path/to/your/train",
    "valid_directory": "/path/to/your/valid",
    "langs": "artifacts/code_only.txt",
    "model_name_or_path": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "tokenizer_sample_mean": 32768,
    "tokenizer_sample_max": 32768,
    "tokenizer_batch_size": 2048,
    "train_batch_size": 128,
    "hn_model_name_or_path": "roberta-base",
    "hn_surface_maxlen": 7,
    "do_tokenizer_sampling": true,
    ...
}
```

**Important Parameters:**
- `tokenizer_sample_mean/max`: Vocabulary size for sampled tokenizers (32768 tokens)
- `tokenizer_batch_size`: Number of texts used to sample each tokenizer (2048)
- `train_batch_size`: Batch size (128)
- `tokenizer_noise_mean/std`: Noise added to token frequencies during sampling

## Output Format

The script produces a JSON file with the following structure:

```json
{
  "token1": 5432,
  "token2": 4891,
  "token3": 3456,
  ...
}
```

Where:
- Keys are token strings (sampled substrings)
- Values are the number of times each token appeared across all batches
- Tokens are sorted by frequency (descending)

### Example Output

```json
{
  "e": 15234,
  "t": 14567,
  "a": 13890,
  " ": 12345,
  "in": 8765,
  "def": 7654,
  "class": 6543,
  "function": 5432,
  ...
}
```

## Understanding the Results

### Statistics Printed to Console

The script prints:
- Total batches processed
- Total token occurrences (sum of all frequencies)
- Unique tokens sampled
- Average tokens per batch
- Languages included
- Top 20 most frequent tokens

### Example Console Output

```
======================================================================
TOKEN SAMPLING STATISTICS
======================================================================
Total batches processed: 1000
Total token occurrences: 32768000
Unique tokens sampled: 156789
Average tokens per batch: 32768.00
Languages: python, javascript, java, cpp, go, github-issues-filtered-structured
======================================================================

Top 20 most frequent tokens:
 1. ' '                          :  15234 (0.05%)
 2. 'e'                          :  14567 (0.04%)
 3. 't'                          :  13890 (0.04%)
 4. 'a'                          :  12345 (0.04%)
 5. 'in'                         :   8765 (0.03%)
...
```

## How It Works

The script simulates the training process:

1. **Load Configuration**: Parses your config file
2. **Setup Datasets**: Creates dataset loaders for each language
3. **Initialize Collators**: Sets up the tokenizer sampling mechanism
4. **Sample Batches**: For each batch:
   - Randomly selects a language (based on language probabilities)
   - Samples texts from that language
   - Runs Algorithm 1 to sample a tokenizer vocabulary
   - Collects all tokens from the sampled vocabulary
5. **Aggregate Statistics**: Counts frequency of each unique token
6. **Output Results**: Saves sorted results to JSON

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `<config.json>` | Configuration file (required) | - |
| `--num_batches` | Number of batches to sample | 1000 |
| `--output_file` | Output JSON file path | `token_frequencies.json` |

## Performance Notes

- **Time**: Approximately 1-5 minutes per 1000 batches (depends on hardware)
- **Memory**: Requires ~4-8GB RAM for loading models and datasets
- **Storage**: Output JSON file size depends on unique tokens (~10-50MB for 1000 batches)

## Troubleshooting

### "No code languages found!"
- Check that `artifacts/code_only.txt` exists and contains valid language codes
- Ensure language codes match your prepared data files

### "FileNotFoundError: /mnt/disks/persist/train/python.parquet"
- Update `train_directory` in your config to point to where you prepared the data
- Verify that all language parquet files exist

### "Import Error: No module named 'rust_utils'"
- Run `cd rust_utils && maturin develop --release` to compile the Rust extension

### Script runs but produces empty results
- Check that `do_tokenizer_sampling` is `true` in your config
- Verify that `tokenizer_sample_mean` and `tokenizer_sample_max` are set correctly

## Differences from Training

This script does **NOT**:
- Initialize or load the language model
- Create the hypernetwork
- Perform any forward passes or compute losses
- Update any model parameters
- Use GPUs or JAX

It **ONLY**:
- Loads tokenizers
- Creates data loaders
- Runs the tokenizer sampling algorithm (Algorithm 1 from the paper)
- Collects statistics

This means:
- ✅ You can run it on CPU-only machines
- ✅ It's much faster than actual training
- ✅ It doesn't interfere with the training codebase
- ✅ You can analyze the tokenizer sampling process independently

## Next Steps

After collecting statistics, you can:

1. **Analyze Token Distribution**: Use the JSON output to understand what substrings are most common
2. **Compare Languages**: Run separately for each language to see differences
3. **Tune Sampling Parameters**: Adjust `tokenizer_noise_std` and other parameters to change token distribution
4. **Validate Training Setup**: Ensure the tokenizer sampling is working as expected before full training

## Related Files

- `collect_token_stats.py` - Main script
- `configs/token_stats_code_only.json` - Configuration for code-only data
- `artifacts/code_only.txt` - Language list (code languages only)
- `zett/collator.py` - Contains the tokenizer sampling logic
- `rust_utils/src/lib.rs` - Rust implementation of sampling algorithm

---

**Note**: This tool is designed to help you understand the tokenizer sampling process without running the full training pipeline. It's useful for debugging, analysis, and parameter tuning.
