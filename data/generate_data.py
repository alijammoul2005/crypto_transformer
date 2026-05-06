"""
Data generation script for substitution cipher cryptanalysis.

Downloads WikiText-103, generates substitution cipher pairs, and saves
training/validation/test datasets as JSON files.
"""

import json
import random
import string
import re
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

# Set random seeds for reproducibility
random.seed(42)

# Constants
TRAIN_SIZE = 300000
VAL_SIZE = 25000
TEST_SIZE = 25000
CHUNK_SIZE = 200
OUTPUT_DIR = Path("/kaggle/working/data")


def clean_text(text):
    """
    Clean text by:
    - Converting to lowercase
    - Keeping only a-z and spaces
    - Removing special characters and numbers

    Args:
        text: Raw text string

    Returns:
        Cleaned text string
    """
    # Lowercase
    text = text.lower()
    # Keep only a-z and spaces
    text = re.sub(r'[^a-z ]', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def generate_key():
    """
    Generate a random substitution cipher key.

    Returns:
        Dictionary mapping each letter a-z to a unique other letter (bijective)
    """
    alphabet = list(string.ascii_lowercase)
    shuffled = alphabet.copy()
    random.shuffle(shuffled)
    return {alphabet[i]: shuffled[i] for i in range(26)}


def encrypt(text, key):
    """
    Encrypt plaintext using substitution cipher key.

    Args:
        text: Plaintext string (lowercase a-z and spaces)
        key: Substitution cipher key dictionary

    Returns:
        Encrypted ciphertext string
    """
    return ''.join(key.get(char, char) for char in text)


def decrypt(text, key):
    """
    Decrypt ciphertext using substitution cipher key.

    Args:
        text: Ciphertext string
        key: Substitution cipher key dictionary

    Returns:
        Decrypted plaintext string
    """
    # Create inverse key
    inverse_key = {v: k for k, v in key.items()}
    return ''.join(inverse_key.get(char, char) for char in text)


def load_and_clean_corpus():
    """
    Load WikiText-103 dataset and clean it.

    Returns:
        Single string containing all cleaned text
    """
    print("Loading WikiText-103 dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

    # Combine all splits
    all_text = []
    for split in ['train', 'validation', 'test']:
        for item in dataset[split]:
            text = item['text']
            if text.strip():  # Skip empty lines
                all_text.append(text)

    # Join and clean
    corpus = ' '.join(all_text)
    print(f"Raw corpus size: {len(corpus)} characters")

    cleaned_corpus = clean_text(corpus)
    print(f"Cleaned corpus size: {len(cleaned_corpus)} characters")

    return cleaned_corpus


def generate_samples(corpus, num_samples, chunk_size=CHUNK_SIZE):
    """
    Generate substitution cipher sample pairs.

    Each sample:
    - Random chunk of text from corpus
    - Fresh random substitution key
    - Plaintext and corresponding ciphertext

    Args:
        corpus: Cleaned text corpus
        num_samples: Number of samples to generate
        chunk_size: Length of each text chunk

    Returns:
        List of dictionaries with 'plaintext' and 'ciphertext' keys
    """
    samples = []
    corpus_len = len(corpus)

    # Ensure corpus is long enough
    if corpus_len < chunk_size:
        raise ValueError(f"Corpus too small ({corpus_len}) for chunk size {chunk_size}")

    for _ in tqdm(range(num_samples), desc="Generating samples"):
        # Random starting position
        start_idx = random.randint(0, corpus_len - chunk_size)
        plaintext = corpus[start_idx:start_idx + chunk_size]

        # Generate fresh random key for this sample
        key = generate_key()
        ciphertext = encrypt(plaintext, key)

        samples.append({
            'plaintext': plaintext,
            'ciphertext': ciphertext
        })

    return samples


def save_dataset(samples, filename):
    """
    Save samples to JSON file.

    Args:
        samples: List of sample dictionaries
        filename: Output filename
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / filename

    with open(filepath, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"Saved {len(samples)} samples to {filepath}")


def main():
    """
    Main data generation pipeline.
    """
    print("="*80)
    print("SUBSTITUTION CIPHER DATA GENERATION")
    print("="*80)

    # Load and clean corpus
    corpus = load_and_clean_corpus()

    # Generate datasets
    print("\n" + "="*80)
    print("Generating training set...")
    train_samples = generate_samples(corpus, TRAIN_SIZE)
    save_dataset(train_samples, 'train.json')

    print("\n" + "="*80)
    print("Generating validation set...")
    val_samples = generate_samples(corpus, VAL_SIZE)
    save_dataset(val_samples, 'val.json')

    print("\n" + "="*80)
    print("Generating test set...")
    test_samples = generate_samples(corpus, TEST_SIZE)
    save_dataset(test_samples, 'test.json')

    print("\n" + "="*80)
    print("DATA GENERATION COMPLETE!")
    print("="*80)
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print(f"Total samples: {len(train_samples) + len(val_samples) + len(test_samples)}")

    # Show example
    print("\n" + "="*80)
    print("Example sample:")
    print("="*80)
    example = train_samples[0]
    print(f"Plaintext:  {example['plaintext'][:100]}...")
    print(f"Ciphertext: {example['ciphertext'][:100]}...")


if __name__ == '__main__':
    main()
