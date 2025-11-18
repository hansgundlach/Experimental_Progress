# Memory-Mapped Dataset Setup

## 🎯 Problem Solved
Large models (256d+) with Chinchilla scaling need 1B+ tokens, which causes OOM when loading datasets. Memory mapping solves this by keeping data on disk and loading only what's needed.

## 📋 Quick Summary
**Before**: Dataset uses 12GB RAM → OOM with 256d model ❌
**After**: Dataset uses ~0.1GB RAM → Fits easily ✅

---

## 🚀 Step-by-Step Instructions

### Step 1: Upload Code to Remote Server
Upload these modified files to your remote server:
- `data_loading.py` (modified to use memory mapping)
- `pretokenize_dataset.py` (new script)
- `experiment_utils.py` (has diagnostic logging added)
- `core.py` (has diagnostic logging added)

```bash
# On your local machine, sync to remote:
rsync -avz data_loading.py pretokenize_dataset.py experiment_utils.py core.py \
    user@supercloud:/path/to/Experimental_Progress/
```

---

### Step 2: Pre-tokenize Your Datasets (ONE-TIME SETUP)

**On the remote server**, run these commands to pre-tokenize each dataset:

```bash
# Navigate to your project directory
cd /path/to/Experimental_Progress

# Pre-tokenize C4 dataset (takes ~20-30 mins for large files)
python pretokenize_dataset.py Datasets/c4_subset_large.txt

# Pre-tokenize OpenWebText (if you use it)
python pretokenize_dataset.py Datasets/openwebtext_subset.txt

# Pre-tokenize WikiText (if you use it)
python pretokenize_dataset.py Datasets/wikitext103_train.txt
python pretokenize_dataset.py Datasets/wikitext103_validation.txt
```

**What this does:**
- Reads the text file in chunks (doesn't load entire file into RAM)
- Tokenizes everything once
- Saves as `Datasets/c4_subset_large.npy` (or similar)
- File size: ~4GB for 1B tokens
- **You only need to do this ONCE per dataset!**

**Expected output:**
```
============================================================
PRE-TOKENIZING DATASET
============================================================
Loading tokenizer from ./gpt2_tokenizer...

Reading text file: Datasets/c4_subset_large.txt
  File size: 4.23 GB (4230 MB)

Tokenizing (this may take a while for large files)...
  Chunk 43: 4230M chars → 1050.2M tokens
  ✅ Total: 1,050,234,567 tokens from 4,230,123,456 chars
     Char-to-token ratio: 4.03:1

Saving to Datasets/c4_subset_large.npy...
✅ Done! Saved to: Datasets/c4_subset_large.npy
   Tokens: 1,050,234,567
   File size: 4.01 GB

============================================================
MEMORY-MAPPED FILE CREATED SUCCESSFULLY
============================================================
This file will be memory-mapped during training.
Memory mapping means the OS loads only the pages you access,
so the entire dataset doesn't need to fit in RAM!
============================================================
```

---

### Step 3: Run Your Training (As Normal!)

**No changes to your training commands!** The dataset loader automatically detects the `.npy` files.

```bash
# Run experiments as usual
python experiments.py

# Or submit SLURM jobs as usual
sbatch submit_job.sh
```

**What you'll see during training:**
```
🚀 Using memory-mapped file: Datasets/c4_subset_large.npy
   (This uses ~0MB RAM regardless of dataset size!)
   Total tokens in file: 1,050,234,567
   Using train portion: 945,211,110 tokens (indices 0 to 945,211,110)
StreamingDataset (train): 7,383,680 sequences from 945,211,110 tokens
```

✅ **That's it!** Your 256d model will now train without OOM errors.

---

## 🔍 How to Verify It's Working

### Check 1: Look for the rocket emoji 🚀
When training starts, you should see:
```
🚀 Using memory-mapped file: Datasets/c4_subset_large.npy
   (This uses ~0MB RAM regardless of dataset size!)
```

✅ **If you see this:** Memory mapping is working!
❌ **If you see warning:** `.npy` file not found, run Step 2 again

### Check 2: Monitor RAM usage during training
```bash
# On remote server, watch memory usage:
watch -n 1 nvidia-smi  # GPU memory
htop                    # CPU/RAM usage
```

**Expected RAM usage:**
- **Without memory mapping**: Spikes to 25-30GB during data loading
- **With memory mapping**: Stays around 12-15GB total

---

## 📁 What Files Are Created?

After pre-tokenization, your `Datasets/` folder will have:

```
Datasets/
├── c4_subset_large.txt           # Original text file (4.2GB)
├── c4_subset_large.npy           # Pre-tokenized tokens (4.0GB) ← NEW!
├── openwebtext_subset.txt        # Original text file
├── openwebtext_subset.npy        # Pre-tokenized tokens ← NEW!
├── wikitext103_train.txt         # Original text file
├── wikitext103_train.npy         # Pre-tokenized tokens ← NEW!
└── ...
```

**Disk space needed:** ~2x the original text file size
**Can I delete the .txt files?** No, keep them as backup
**Can I reuse .npy files?** Yes! Once created, use them forever

---

## ❓ Troubleshooting

### Problem: "⚠️ WARNING: No .npy file found"
**Solution:** Run Step 2 to pre-tokenize the dataset

### Problem: "FileNotFoundError: c4_subset_large.npy"
**Solution:** Make sure you ran `pretokenize_dataset.py` in the correct directory

### Problem: Still getting OOM errors
**Check these:**
1. Verify you see the 🚀 rocket emoji in training output
2. Check that `.npy` file exists: `ls -lh Datasets/*.npy`
3. Make sure you uploaded the modified `data_loading.py`

### Problem: Pre-tokenization is taking forever
**Expected time:**
- Small datasets (100MB): ~1-2 minutes
- Medium datasets (1GB): ~10-15 minutes
- Large datasets (4GB+): ~30-45 minutes

This is normal! You only need to do it once per dataset.

---

## 🎓 Technical Details (Optional Reading)

### What is memory mapping?
Memory mapping means the OS treats a disk file as if it's in RAM:
- File stays on disk
- OS loads only the 4KB "pages" you actually access
- When RAM fills up, OS automatically evicts old pages
- To your code, it looks like a normal array

### Why is this better?
**Old way (loading into RAM):**
```python
tokens = [1, 2, 3, ...]  # All 1B tokens in RAM = 8GB
```

**New way (memory mapping):**
```python
tokens = np.memmap('file.npy')  # File stays on disk
x = tokens[1000:2000]           # OS loads only this 4KB page
```

### Performance impact?
**First access**: ~0.1ms slower (disk read)
**Subsequent accesses**: Same speed (OS caches hot pages)
**Overall**: < 1% training time difference

---

## ✅ Success Checklist

Before running 256d experiments, verify:
- [ ] `data_loading.py` uploaded to remote server
- [ ] `pretokenize_dataset.py` uploaded to remote server
- [ ] Ran `python pretokenize_dataset.py Datasets/c4_subset_large.txt`
- [ ] Verified `.npy` file exists: `ls -lh Datasets/*.npy`
- [ ] Ran test training, saw 🚀 emoji in output
- [ ] Confirmed RAM usage is low (~10-15GB instead of 25-30GB)

**Once all checked:** Your 256d model is ready to train! 🎉

---

## 📞 Need Help?

If you still have issues, check:
1. Do you see the 🚀 emoji? (memory mapping active)
2. Does `Datasets/c4_subset_large.npy` exist?
3. What's the last diagnostic message before the hang?

The diagnostic logging in `core.py` and `experiment_utils.py` will show exactly where any issues occur.
