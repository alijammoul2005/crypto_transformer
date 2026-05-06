"""
PyTorch Dataset class for substitution cipher data.

Handles character-level tokenization with vocabulary:
a-z + space + <SOS> + <EOS> + <PAD> = 30 tokens
"""

import json
import torch
from torch.utils.data import Dataset
import string


class CipherDataset(Dataset):
    """
    Dataset class for substitution cipher cryptanalysis.

    Loads plaintext-ciphertext pairs and converts them to token sequences.

    Vocabulary:
        - a-z (26 letters)
        - space (1)
        - <PAD> (padding token)
        - <SOS> (start of sequence)
        - <EOS> (end of sequence)
        Total: 30 tokens
    """

    def __init__(self, json_path, max_len=202):
        """
        Initialize dataset.

        Args:
            json_path: Path to JSON file with plaintext/ciphertext pairs
            max_len: Maximum sequence length (200 chars + SOS + EOS)
        """
        self.max_len = max_len

        # Load data
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        # Build vocabulary
        self.chars = list(string.ascii_lowercase) + [' ']  # a-z + space
        self.PAD_TOKEN = '<PAD>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'

        # Create char to index mapping
        self.char2idx = {char: idx for idx, char in enumerate(self.chars)}
        self.char2idx[self.PAD_TOKEN] = len(self.chars)      # index 27
        self.char2idx[self.SOS_TOKEN] = len(self.chars) + 1  # index 28
        self.char2idx[self.EOS_TOKEN] = len(self.chars) + 2  # index 29

        # Create index to char mapping
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}

        # Store special token indices
        self.PAD_IDX = self.char2idx[self.PAD_TOKEN]
        self.SOS_IDX = self.char2idx[self.SOS_TOKEN]
        self.EOS_IDX = self.char2idx[self.EOS_TOKEN]

        self.vocab_size = len(self.char2idx)  # 30

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
        Get a single sample.

        Returns:
            src: Source sequence (ciphertext + EOS), padded to max_len
            tgt: Target sequence (SOS + plaintext + EOS), padded to max_len

        Both returned as torch tensors of shape (max_len,)
        """
        sample = self.data[idx]
        plaintext = sample['plaintext']
        ciphertext = sample['ciphertext']

        # Encode ciphertext: add EOS at end
        src_tokens = self.encode(ciphertext) + [self.EOS_IDX]

        # Encode plaintext: add SOS at start and EOS at end
        tgt_tokens = [self.SOS_IDX] + self.encode(plaintext) + [self.EOS_IDX]

        # Pad sequences to max_len
        src_tokens = src_tokens[:self.max_len]  # Truncate if too long
        tgt_tokens = tgt_tokens[:self.max_len]

        # Pad with PAD_IDX
        src_tokens += [self.PAD_IDX] * (self.max_len - len(src_tokens))
        tgt_tokens += [self.PAD_IDX] * (self.max_len - len(tgt_tokens))

        # Convert to tensors
        src = torch.tensor(src_tokens, dtype=torch.long)
        tgt = torch.tensor(tgt_tokens, dtype=torch.long)

        return src, tgt


def test_dataset():
    """
    Test function for dataset class.
    """
    # Create a small test file
    test_data = [
        {'plaintext': 'hello world', 'ciphertext': 'mjqqt btsqa'},
        {'plaintext': 'the quick brown fox', 'ciphertext': 'xmj zfnhp dstbu ktc'}
    ]

    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_path = f.name

    try:
        # Test dataset
        dataset = CipherDataset(temp_path, max_len=202)

        print(f"Vocabulary size: {dataset.vocab_size}")
        print(f"PAD_IDX: {dataset.PAD_IDX}")
        print(f"SOS_IDX: {dataset.SOS_IDX}")
        print(f"EOS_IDX: {dataset.EOS_IDX}")
        print(f"Dataset size: {len(dataset)}")

        # Test encoding/decoding
        src, tgt = dataset[0]
        print(f"\nSample 0:")
        print(f"Source shape: {src.shape}")
        print(f"Target shape: {tgt.shape}")
        print(f"Decoded source: {dataset.decode(src)}")
        print(f"Decoded target: {dataset.decode(tgt)}")

    finally:
        os.unlink(temp_path)


if __name__ == '__main__':
    test_dataset()
