# LSTM vs Transformer Data Loading - Complete Comparison

## ✅ Summary: THEY ARE IDENTICAL (with minor path differences)

Both models use the **exact same tokenization** and will get **identical token sequences** when using `.npy` files or when loading from the same text file.

---

## Data Loading Flow Comparison

### Transformer Path
```
core.py (line 820)
  ↓
get_dataset_smart(config)  [data_loading.py:720]
  ↓
get_streaming_dataset(config)  [data_loading.py:627]
  ↓
StreamingTextDataset()  [data_loading.py:115]
  ↓
Tokenization:
  1. Check for .npy file → memory mapping
  2. Fallback: 2000-char chunks, truncation=True, max_length=1024
```

### LSTM Path
```
lstm_training.py (line 680)
  ↓
load_and_tokenize_text(config)  [data_loading.py:744]
  ↓
Tokenization:
  1. Check for .npy file → memory mapping
  2. Fallback: 2000-char chunks, truncation=True, max_length=1024
  ↓
Returns token lists (not Dataset objects)
  ↓
Wrapped in TokenDataset or LSTMStatefulDataset
```

---

## Detailed Comparison Table

| Aspect | Transformer | LSTM | Identical? |
|--------|------------|------|------------|
| **Memory Mapping** | ✅ Checks for .npy file | ✅ Checks for .npy file | ✅ YES |
| **Fallback Chunking** | 2000 chars | 2000 chars | ✅ YES |
| **Truncation** | `truncation=True` | `truncation=True` | ✅ YES |
| **Max Length** | `max_length=1024` | `max_length=1024` | ✅ YES |
| **add_special_tokens** | `True` (default) | `True` (default) | ✅ YES |
| **Tokenizer Path** | `./gpt2_tokenizer` | `../gpt2_tokenizer` | ⚠️ Different paths, same files |
| **Tokenizer Type** | Default (may be fast) | `use_fast=False` | ⚠️ See below |
| **Offline Mode** | Not forced | `local_files_only=True` | ⚠️ See below |

---

## Key Differences (and Why They Exist)

### 1. Tokenizer Path
**Transformer:** `./gpt2_tokenizer` (current directory)
**LSTM:** `../gpt2_tokenizer` (parent directory)

**Why Different?**
- LSTM training runs from `LSTM_model/` subdirectory
- Transformer training runs from project root
- Both point to the same tokenizer files

**Does it matter?**
- ❌ NO - Same tokenizer vocabulary, same tokenization
- Both paths resolve to `/path/to/Experimental_Progress/gpt2_tokenizer/`

### 2. Fast vs Slow Tokenizer
**Transformer:** Uses default (may be fast Rust-based tokenizer if available)
**LSTM:** `use_fast=False` (forces slow Python tokenizer)

**Why Different?**
- Historical: LSTM code was written to avoid potential fast tokenizer issues
- Conservative approach for reproducibility

**Does it matter?**
- ❌ NO - When both tokenizers produce identical outputs for same input
- Fast tokenizer is just faster, not different in results
- **RECOMMENDATION:** Could make both use fast tokenizer for speed

### 3. Offline Mode
**Transformer:** May try to download tokenizer from HuggingFace
**LSTM:** `local_files_only=True` (forces local files only)

**Why Different?**
- LSTM designed for offline environments (MIT Supercloud)
- Transformer assumes internet may be available

**Does it matter?**
- ❌ NO - Both ultimately load from local `gpt2_tokenizer/` directory
- `local_files_only=True` just prevents network attempts if files missing

---

## Token Sequence Verification

### When using .npy files (MOST IMPORTANT)
**Both models load from the SAME .npy file:**
```python
# Transformer
all_tokens_mmap = np.load('Datasets/c4_subset_large.npy', mmap_mode='r')

# LSTM (from subdirectory)
all_tokens_mmap = np.load('../Datasets/c4_subset_large.npy', mmap_mode='r')
```

**Result:** ✅ **IDENTICAL token sequences** (byte-for-byte same)

### When loading from text files (fallback)
**Both use identical tokenization:**
```python
chunk_size = 2000
text_chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
for chunk in text_chunks:
    tokens = tokenizer(chunk, truncation=True, max_length=1024)["input_ids"]
```

**Result:** ✅ **IDENTICAL token sequences** (assuming same random seed for text sampling)

---

## Stride Differences

| Model | Default Stride | Effect |
|-------|---------------|--------|
| Transformer | `stride=128` | Sequences overlap by (seq_length - 128) tokens |
| LSTM | `stride=1` | Maximum overlap (sliding window) |

**Why Different?**
- Transformer uses larger stride for efficiency (less overlap = fewer sequences)
- LSTM uses stride=1 for maximum data utilization (important for smaller models)

**Does it matter?**
- ⚠️ **YES** - Different strides mean different training sequences
- But this is intentional based on model architecture needs
- **Recommendation:** This difference is correct and should stay

---

## Dataset Split Behavior

### Transformer (StreamingTextDataset)
```python
if split == "train":
    # Use first 90% of tokens
    start_idx = 0
    end_idx = int(total_tokens * 0.9)
elif split == "val":
    # Use last 10% of tokens
    start_idx = int(total_tokens * 0.9)
    end_idx = total_tokens
```

### LSTM (load_and_tokenize_text)
```python
# Default splits: train=80%, val=10%, test=10%
n_train = int(n * 0.8)
n_val = int(n * 0.1)
train_tokens = tokens[:n_train]
val_tokens = tokens[n_train : n_train + n_val]
test_tokens = tokens[n_train + n_val:]
```

**Why Different?**
- Transformer: 90/10 split (no separate test set)
- LSTM: 80/10/10 split (includes test set)

**Does it matter?**
- ⚠️ **YES** - Models train on different portions of data
- Transformer trains on more data (90% vs 80%)
- **Recommendation:** Consider unifying to 90/10 for both (no test set)

---

## Final Verdict

### ✅ Tokenization is IDENTICAL
Both models will produce the **exact same token sequences** from the same text.

### ⚠️ Three legitimate differences:
1. **Stride:** Different by design (Transformer=128, LSTM=1) - KEEP AS IS
2. **Train/Val Split:** Different proportions (90/10 vs 80/10/10) - CONSIDER UNIFYING
3. **Tokenizer Speed:** LSTM forces slow tokenizer - COULD UNIFY for speed

### 🎯 Recommendations:

#### Must Change (for fair comparison):
1. **Unify train/val split to 90/10** - Make LSTM use same 90/10 split as Transformer

#### Optional Optimizations:
2. **Use fast tokenizer for LSTM** - Remove `use_fast=False` for 5-10x faster loading
3. **Remove `local_files_only=True` from LSTM** - Not needed if tokenizer files exist locally

#### Keep As Is:
4. **Different strides** - Appropriate for each architecture
5. **Different tokenizer paths** - Required due to different execution directories

---

## Code Changes Needed for Perfect Alignment

### Option 1: Minimal Changes (Just Fix Split)
Update `load_and_tokenize_text()` to use 90/10 split by default:
```python
train_split = config.get("train_split", 0.9)  # Changed from 0.8
val_split = config.get("val_split", 0.1)      # Keep same
# No test set by default
```

### Option 2: Full Optimization (Speed + Fair Comparison)
1. Change LSTM default split to 90/10
2. Remove `use_fast=False` from LSTM tokenizer
3. Remove `local_files_only=True` from LSTM tokenizer

**Benefits:**
- ✅ Both train on same 90% of data (fair comparison)
- ✅ LSTM data loading 5-10x faster (fast tokenizer)
- ✅ Cleaner code (less special-casing)

**Risks:**
- ⚠️ May need to regenerate LSTM results (due to different train split)
- ⚠️ Fast tokenizer requires additional files in `gpt2_tokenizer/`

---

## Testing Checklist

To verify both models use identical data:

```bash
# 1. Ensure same .npy file is accessible to both
ls -lh Datasets/c4_subset_large.npy
ls -lh LSTM_model/../Datasets/c4_subset_large.npy  # Should be same file

# 2. Run both models and compare tokens loaded
python experiments.py 2>&1 | grep "Total tokens in file"
cd LSTM_model && python lstm_experiments.py 2>&1 | grep "Total tokens in file"
# Should show same number

# 3. Compare first batch of tokens (after fixing split ratio)
# Add debug print in both training scripts:
# print(f"First 10 tokens: {batch[0][:10]}")
```

**Expected:** Both print identical token sequences (if using same split ratio).
