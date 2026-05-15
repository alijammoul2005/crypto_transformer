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
# Use relative path for local data generation
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "generated"


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


def key_to_string(key):
    """
    Convert key dictionary to 26-character string.

    The string representation is: position i contains the plaintext letter
    that cipher letter i decrypts to.

    Args:
        key: Dictionary mapping plaintext -> ciphertext

    Returns:
        26-character string where key_string[i] is the plaintext letter
        that chr(ord('a') + i) decrypts to
    """
    # Create inverse mapping: ciphertext -> plaintext
    inverse_key = {v: k for k, v in key.items()}

    # Build key string: for each cipher letter a-z, what plaintext letter does it map to?
    key_string = ''
    for i in range(26):
        cipher_char = chr(ord('a') + i)
        plaintext_char = inverse_key.get(cipher_char, cipher_char)
        key_string += plaintext_char

    return key_string


def generate_samples(corpus, num_samples, chunk_size=CHUNK_SIZE):
    """
    Generate plaintext samples for substitution cipher training.

    Each sample contains only plaintext - keys will be generated fresh
    during training in the dataset __getitem__ method.

    Args:
        corpus: Cleaned text corpus
        num_samples: Number of samples to generate
        chunk_size: Length of each text chunk

    Returns:
        List of dictionaries with 'plaintext' key only
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

        samples.append({
            'plaintext': plaintext
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
    print(f"Plaintext: {example['plaintext'][:100]}...")
    print("\nNote: Keys will be generated fresh during training for maximum diversity")

    # Demonstrate key format
    print("\n" + "="*80)
    print("Key format example:")
    print("="*80)
    demo_key = generate_key()
    demo_key_string = key_to_string(demo_key)
    demo_cipher = encrypt(example['plaintext'][:50], demo_key)
    print(f"Example key string: {demo_key_string}")
    print(f"(26 chars, position i = plaintext for cipher letter a+i)")
    print(f"\nOriginal: {example['plaintext'][:50]}")
    print(f"Encrypted: {demo_cipher}")


if __name__ == '__main__':
    main()
