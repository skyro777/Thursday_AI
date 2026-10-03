# Thursday AI on Google Colab Free — Brief Tutorial

> **Honest scope:** Colab Free = T4 16GB GPU (same as Kaggle). It works for the **smoke test** (~3 hours). For real v0.1 (90k examples × 3 epochs = ~25 hours), it will hit the 12-hour session limit — use Lightning AI Studios instead.

---

## Why Colab Free is "okay but limited"

| Aspect | Kaggle | Colab Free |
|---|---|---|
| GPU | T4 16GB | T4 16GB (same) |
| Session limit | 12h hard cap | ~12h/day, **90-min idle disconnect** |
| Persistent storage | 20GB `/kaggle/working` | None (must mount Google Drive) |
| GPU availability | Guaranteed | Dynamic (sometimes you wait) |
| Background execution | No | No |

**Translation:** For the smoke test (3h), Colab works. For real v0.1 (25h), it doesn't — same problem as Kaggle.

---

## Step 1 — Open Colab (1 min)

1. Go to **https://colab.research.google.com**
2. Sign in with your Google account
3. Click **"New Notebook"** (or `File → New notebook`)

---

## Step 2 — Enable GPU (1 min)

1. Click **"Runtime"** in the top menu
2. Click **"Change runtime type"**
3. Set **"Hardware accelerator"** → **"T4 GPU"**
4. Click **"Save"**

**Gotcha:** Sometimes during peak hours, Colab says "GPU unavailable" — just wait a few minutes and try again.

---

## Step 3 — Add your HF token as a Secret (2 min)

1. Look at the **left sidebar** — click the **key icon** (Secrets)
2. Click **"Add new secret"**
3. **Name:** `HF_TOKEN` (exactly this, case-sensitive)
4. **Value:** paste your HF token from https://huggingface.co/settings/tokens (use "Write" or "Full Access" preset)
5. **Toggle:** Make sure **"Notebook access"** is **ON** (the toggle next to the secret)

**Gotcha:** If you don't toggle "Notebook access" ON, the notebook can't read the secret.

---

## Step 4 — Get the notebook (1 min)

You have two options:

**Option A — Upload the Colab notebook directly:**
1. Download from: https://github.com/skyro777/Thursday_AI/blob/main/training/thursday_ai_finetune_colab.ipynb
2. In Colab: `File → Upload notebook` → select the downloaded file

**Option B — Open directly from GitHub:**
1. In Colab: `File → Open notebook` → "GitHub" tab
2. Paste: `https://github.com/skyro777/Thursday_AI`
3. Find `training/thursday_ai_finetune_colab.ipynb` and click it

---

## Step 5 — Mount Google Drive (optional but recommended)

To persist your trained model between Colab sessions (otherwise you lose everything when the session disconnects):

Add a new cell at the top of the notebook and run:
```python
from google.colab import drive
drive.mount('/content/drive')
```
Authorize when prompted.

Then change these in the CONFIG cell (cell 0):
```python
'output_dir'  : '/content/drive/MyDrive/thursday-ai-lora',
'merged_dir'   : '/content/drive/MyDrive/thursday-ai-merged',
'gguf_path'    : '/content/drive/MyDrive/thursday-ai-v0.1-Q4_K_M.gguf',
```

Now even if Colab disconnects mid-training, your saved checkpoints survive in Google Drive.

---

## Step 6 — Run the notebook (3 hours for smoke)

1. **Cell 1** (deps install) — Run it. Expected output:
   ```
   torch           2.5.1+cu121
   transformers    4.46.x
   trl             0.11.4
   bitsandbytes    0.49.2
   GPU count: 1
     GPU 0: Tesla T4
   OK: all imports succeeded. Proceed to cell 2.
   ```

2. **Cell 2** (HF token check) — Run it. Expected:
   ```
   Token is valid. Logged in as: <your-HF-username>
   Will upload merged model to: <username>/thursday-ai-v0.1-merged
   Token has WRITE permission
   ```

3. **Cells 3-7** — Run all. ~5 min total. Model download ~2 min (cached after).

4. **Cell 8 (Train!)** — Run it. **~3 hours.** Watch for:
   ```
   Num examples = 5,000 | Num Epochs = 1 | Total steps = 625
   ```
   Loss should drop from ~2.0 to ~0.5-1.0 over the run.

5. **Cells 9-14** — Run all. ~15 min (save, merge, GGUF convert, upload, smoke test).

---

## Step 7 — Pull on your potato PC

After cell 13 (upload) succeeds:
```bash
ollama pull hf.co/<your-hf-username>/thursday-ai-v0.1-gguf:Q4_K_M
ollama run hf.co/<your-hf-username>/thursday-ai-v0.1-gguf:Q4_K_M
```

Test:
```
>>> Hey Thursday, set an alarm for 7 am tomorrow.
```

Expected:
```json
{"name": "set_alarm", "arguments": {"hour": 7, "minute": 0, "label": "tomorrow 7am"}}
```

---

## ⚠️ Colab Free gotchas to avoid

1. **90-min idle disconnect:** If you don't interact with the page for 90 minutes, Colab kills the session. **Keep the tab active** — click it every 30 min during training, or use a browser extension like "Colab Auto-Clicker" that pings the page periodically.

2. **Tab closing kills training:** Unlike Lightning, Colab Free has no background execution. If you close the tab, training dies. Leave the tab open.

3. **Dynamic GPU availability:** Sometimes during peak hours you'll get "GPU unavailable" when trying to start a session. Just wait or try again later.

4. **No persistent storage:** Without Google Drive mount, everything is lost when the session ends. Mount Drive (Step 5) if you want to keep checkpoints.

5. **12-hour daily cap:** Total GPU time per day is capped (~12h on free tier). If you burn through it, you're locked out for the day.

---

## For real v0.1 (later)

Once you confirm the smoke test works on Colab, switch to **Lightning AI Studios** for real v0.1:
- L4 GPU (2.5× faster than T4)
- 22 hours/month free quota (v0.1 takes ~8h)
- Persistent storage (no reinstall)
- No idle disconnect (background execution allowed)
- Tutorial: `training/LIGHTNING_TUTORIAL.md`

---

## TL;DR

```
1. Go to colab.research.google.com (1 min)
2. Runtime → Change runtime type → T4 GPU (1 min)
3. Left sidebar → key icon → Add secret HF_TOKEN (2 min)
4. Upload thursday_ai_finetune_colab.ipynb (1 min)
5. (Optional) Mount Google Drive for persistence (2 min)
6. Run All — ~3 hours for smoke test
7. Pull GGUF via Ollama on potato PC
```

Total: ~7 min setup + 3 hours training. Cost: $0.
