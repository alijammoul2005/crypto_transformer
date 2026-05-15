"""
PyTorch Dataset class for substitution cipher KEY prediction.

NEW APPROACH: Predict the 26-letter substitution key, then apply it deterministically.

Input: ciphertext (200 chars)
Output: substitution key (26 chars) representing the mapping

Vocabulary: a-z + <SOS> + <EOS> + <PAD> = 29 tokens (no space in key output)
"""

import json
import torch
from torch.utils.data import Dataset
import string
import random


class CipherDataset(Dataset):
    """
    Dataset class for substitution cipher KEY prediction.

    Generates fresh random keys on-the-fly for each access to maximize diversity.

    Input sequence: ciphertext (encrypted with fresh random key)
    Output sequence: 26-letter key string

    Vocabulary:
        - a-z (26 letters)
        - <PAD> (padding token)
        - <SOS> (start of sequence)
        - <EOS> (end of sequence)
        Total: 29 tokens
    """

    def __init__(self, json_path, max_src_len=201, seed=None):
        """
        Initialize dataset.

        Args:
            json_path: Path to JSON file with plaintext samples
            max_src_len: Maximum source length (200 chars + EOS)
            seed: Random seed for reproducibility (None for random)
        """
        self.max_src_len = max_src_len
        self.key_len = 28  # 26 letters + SOS + EOS

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)

        # Load data (plaintext only)
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        # Build vocabulary (a-z only, no space in keys)
        self.chars = list(string.ascii_lowercase)  # a-z
        self.PAD_TOKEN = '<PAD>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'

        # Create char to index mapping
        self.char2idx = {char: idx for idx, char in enumerate(self.chars)}
        self.char2idx[' '] = 26  # Space for ciphertext input
        self.char2idx[self.PAD_TOKEN] = 27
        self.char2idx[self.SOS_TOKEN] = 28
        self.char2idx[self.EOS_TOKEN] = 29

        # Create index to char mapping
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}

        # Store special token indices
        self.PAD_IDX = self.char2idx[self.PAD_TOKEN]
        self.SOS_IDX = self.char2idx[self.SOS_TOKEN]
        self.EOS_IDX = self.char2idx[self.EOS_TOKEN]

        self.vocab_size = len(self.char2idx)  # 30

    def generate_key(self):
        """
        Generate a random substitution cipher key.

        Returns:
            Dictionary mapping each letter a-z to a unique other letter (bijective)
        """
        alphabet = list(string.ascii_lowercase)
        shuffled = alphabet.copy()
        random.shuffle(shuffled)
        return {alphabet[i]: shuffled[i] for i in range(26)}

    def key_to_string(self, key):
        """
        Convert key dictionary to 26-character string.

        The string at position i contains the plaintext letter that
        cipher letter chr(ord('a') + i) decrypts to.

        Args:
            key: Dictionary mapping plaintext -> ciphertext

        Returns:
            26-character string representing the decryption key
        """
        # Create inverse mapping: ciphertext -> plaintext
        inverse_key = {v: k for k, v in key.items()}

        # Build key string
        key_string = ''
        for i in range(26):
            cipher_char = chr(ord('a') + i)
            plaintext_char = inverse_key.get(cipher_char, cipher_char)
            key_string += plaintext_char

        return key_string

    def encrypt(self, text, key):
        """
        Encrypt plaintext using substitution cipher key.

        Args:
            text: Plaintext string (lowercase a-z and spaces)
            key: Substitution cipher key dictionary

        Returns:
            Encrypted ciphertext string
        """
        return ''.join(key.get(char, char) for char in text)

    def encode(self, text):
        """
        Convert text to token indices.

        Args:
            text: String to encode

        Returns:
            List of token indices
        """
        return [self.char2idx.get(char, self.PAD_IDX) for char in text]

    def decode(self, tokens):
        """
        Convert token indices back to text.

        Args:
            tokens: List or tensor of token indices

        Returns:
            Decoded string
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()

        chars = []
        for idx in tokens:
            if idx == self.EOS_IDX:
                break
            if idx not in [self.PAD_IDX, self.SOS_IDX]:
                chars.append(self.idx2char.get(idx, ''))

        return ''.join(chars)

    def __len__(self):
        """Return number of samples in dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single sample with FRESH random key generation.

        Each access generates a new random key and encrypts the plaintext,
        ensuring maximum diversity across epochs.

        Returns:
            src: Source sequence (ciphertext + EOS), padded to max_src_len
            tgt: Target sequence (SOS + key + EOS), fixed length 28
            plaintext: Original plaintext for evaluation (not used in training)

        src shape: (max_src_len,)
        tgt shape: (28,) = SOS + 26 key letters + EOS
        """
        sample = self.data[idx]
        plaintext = sample['plaintext']

        # Generate FRESH random key for this access
        key = self.generate_key()
        key_string = self.key_to_string(key)

        # Encrypt plaintext with fresh key
        ciphertext = self.encrypt(plaintext, key)

        # Encode ciphertext: add EOS at end
        src_tokens = self.encode(ciphertext) + [self.EOS_IDX]

        # Encode key: add SOS at start and EOS at end
        tgt_tokens = [self.SOS_IDX] + self.encode(key_string) + [self.EOS_IDX]

        # Pad source to max_src_len
        src_tokens = src_tokens[:self.max_src_len]
        src_tokens += [self.PAD_IDX] * (self.max_src_len - len(src_tokens))

        # Target is fixed length (SOS + 26 letters + EOS = 28)
        # No padding needed for key

        # Convert to tensors
        src = torch.tensor(src_tokens, dtype=torch.long)
        tgt = torch.tensor(tgt_tokens, dtype=torch.long)

        return src, tgt


def test_dataset():
    """
    Test function for dataset class.
    """
    # Create a small test file (plaintext only)
    test_data = [
        {'plaintext': 'the quick brown fox jumps over the lazy dog' + ' ' * 155},  # Pad to 200
        {'plaintext': 'hello world this is a test of the substitution cipher key prediction system' + ' ' * 124}
    ]

    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_path = f.name

    try:
        # Test dataset
        dataset = CipherDataset(temp_path, max_src_len=201, seed=42)

        print(f"Vocabulary size: {dataset.vocab_size}")
        print(f"PAD_IDX: {dataset.PAD_IDX}")
        print(f"SOS_IDX: {dataset.SOS_IDX}")
        print(f"EOS_IDX: {dataset.EOS_IDX}")
        print(f"Dataset size: {len(dataset)}")

        # Test encoding/decoding
        src, tgt = dataset[0]
        print(f"\nSample 0 (first access):")
        print(f"Source shape: {src.shape} (ciphertext)")
        print(f"Target shape: {tgt.shape} (key: SOS + 26 letters + EOS)")
        print(f"Decoded source (cipher): {dataset.decode(src)[:50]}...")
        print(f"Decoded target (key):    {dataset.decode(tgt)}")

        # Test fresh key generation
        src2, tgt2 = dataset[0]
        print(f"\nSample 0 (second access - should have different key):")
        print(f"Decoded source (cipher): {dataset.decode(src2)[:50]}...")
        print(f"Decoded target (key):    {dataset.decode(tgt2)}")
        print(f"Keys are different: {not torch.equal(tgt, tgt2)}")

    finally:
        os.unlink(temp_path)


if __name__ == '__main__':
    test_dataset()
