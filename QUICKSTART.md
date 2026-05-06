# Quick Start Guide

## What You Have

A complete, production-ready deep learning project for breaking substitution ciphers using Transformers.

## Project Components

✅ **Data Pipeline**
- `data/generate_data.py` - Downloads WikiText-103 and generates 350k cipher pairs
- `data/dataset.py` - PyTorch Dataset with character-level tokenization

✅ **Model Architecture** (built from scratch, no pre-trained models)
- `model/attention.py` - Multi-head attention mechanism
- `model/transformer.py` - Full encoder-decoder Transformer (3.5M parameters)

✅ **Training System**
- `training/scheduler.py` - Warmup learning rate scheduler
- `training/train.py` - Complete training loop with early stopping

✅ **Evaluation Suite**
- `evaluation/metrics.py` - 4 metrics (char accuracy, word accuracy, BLEU, edit distance)
- `evaluation/evaluate.py` - Test set evaluation script

✅ **Inference Engine**
- `inference/decrypt.py` - Greedy + beam search decoding with 10 demo examples

✅ **Ready for Kaggle**
- `run_all.ipynb` - End-to-end notebook with all steps
- `requirements.txt` - All dependencies
- `README.md` - Comprehensive documentation

## Fastest Way to Run

### On Kaggle:

1. Upload the entire `crypto_transformer` folder to Kaggle
2. Open `run_all.ipynb`
3. Settings → Accelerator → **GPU T4 x2**
4. Run all cells (Click "Run All")

**Total time:** ~3-5 hours (mostly training)

### Locally:

```bash
cd crypto_transformer

# Install dependencies
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"

# Run everything
python data/generate_data.py      # ~15 min
python training/train.py           # ~2-4 hours on GPU
python evaluation/evaluate.py     # ~10 min
python inference/decrypt.py       # ~2 min
```

## What to Expect

### Training Output:
```
EPOCH 1/20
  Step 100: loss=2.1234, acc=45.67%, lr=0.000123
  Step 200: loss=1.8765, acc=58.32%, lr=0.000156
  ...
  Val Loss: 1.2345, Val Acc: 87.65%
  ✓ Best model saved

EPOCH 2/20
  ...
```

### Evaluation Results:
```
Character Accuracy: 95.34%
Word Accuracy:      86.21%
BLEU Score:         0.9123
Mean Edit Distance: 7.45
```

### Demo Examples:
```
Example 1 [Typical] - Character Accuracy: 98.50%
Ciphertext: xli xvergjavqiv piergw xa hilvcx...
Predicted:  the transformer learns to decrypt...
True:       the transformer learns to decrypt...
BLEU Score: 0.9850
```

## File Sizes

- Training data: ~150 MB (JSON)
- Model checkpoint: ~50 MB
- Total disk space needed: ~500 MB

## GPU Memory

- Training: ~4-6 GB VRAM (batch size 32)
- Inference: ~2 GB VRAM

Works on: T4, V100, A100, RTX 3090, etc.

## Reproducibility

All random seeds are fixed (seed=42):
- Python random
- NumPy random
- PyTorch random
- CUDA random (if available)

Running twice should give identical results.

## Common Issues

**Issue:** "No module named 'datasets'"
**Fix:** Run `pip install datasets`

**Issue:** "CUDA out of memory"
**Fix:** Reduce batch size to 16 in `training/train.py`

**Issue:** "File not found: /kaggle/working/data/train.json"
**Fix:** Run `data/generate_data.py` first

**Issue:** Training is very slow
**Fix:** Ensure GPU is enabled with `torch.cuda.is_available()`

## Next Steps

After successful run:

1. **Experiment with hyperparameters:**
   - Increase d_model to 512 for better accuracy
   - Try beam_size=10 in inference
   - Add more training data

2. **Extend the model:**
   - Encrypt spaces too (harder task)
   - Try polyalphabetic ciphers
   - Multi-language support

3. **Visualize attention:**
   - Add attention weight visualization
   - Analyze what patterns the model learns

4. **Deploy:**
   - Create web API with Flask/FastAPI
   - Build interactive demo

## Success Criteria

✅ Data generated: 350k samples in `/kaggle/working/data/`
✅ Model trained: checkpoint saved to `/kaggle/working/checkpoints/best_model.pt`
✅ Character accuracy: >90% (target: 95%+)
✅ Demo runs: 10 examples with predictions

## Support

All code is heavily commented. Each file has:
- Module docstring explaining purpose
- Function docstrings with args/returns
- Inline comments for complex logic

Read the README.md for detailed architecture explanation.

---

**Ready to decrypt!** 🔓

Run `run_all.ipynb` and watch the Transformer learn to break ciphers!
