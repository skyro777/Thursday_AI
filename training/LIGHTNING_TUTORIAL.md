# Thursday AI on Lightning AI Studios — Quick Tutorial

> Brief, focused, no-fluff. Follow these steps in order. ~10 min to set up, then 8 hours of training.

---

## Step 1 — Sign up (2 min)

1. Go to **https://lightning.ai**
2. Click **"Sign Up"** (top right)
3. Sign up with Google OR GitHub OR email — any is fine, no credit card needed
4. Verify your email if you used email signup
5. You'll land on the **Studios dashboard** — that's your home base

**Gotcha:** If asked "What brings you here?" or similar onboarding, pick anything. Doesn't matter.

---

## Step 2 — Create a Studio (2 min)

1. Click **"New Studio"** (big button, top right)
2. Name it: `thursday-ai`
3. Description: leave blank (or whatever)
4. Click **"Create Studio"**
5. Wait ~30 sec for it to boot — you'll see a JupyterLab-like interface

**What you have now:** A persistent Linux environment with `/teamspace/studios/this_studio/` as your home dir. Files saved here stay between sessions — no reinstalling pip packages or re-downloading the model.

---

## Step 3 — Select the L4 GPU (30 sec)

1. Look at the **right sidebar** (might be collapsed — click the `>>` to expand)
2. Find the **"Hardware"** section
3. Click the dropdown that says "CPU" (default)
4. Select **"GPU"** → then **"L4"** (this is the free-tier GPU, 24 GB VRAM)
5. Click **"Start"** or "Apply"**
6. Wait ~1 min — it'll provision the GPU. Status shows "Running" when ready

**Gotcha:** The L4 is the free option. Don't pick A100, H100, or "Multi-GPU" — those cost money and will burn your credits.

**Cost check:** Free tier = 22 GPU-hours/month. One v0.1 training run = ~8 hours. You have plenty.

---

## Step 4 — Set your HF token as env var (1 min)

1. Open the **terminal** in your Studio (the black panel at the bottom, or `File → New → Terminal`)
2. Run this command (replace `hf_xxx` with your actual HF token from https://huggingface.co/settings/tokens):

   ```bash
   echo 'export HF_TOKEN=hf_your_token_here' >> ~/.bashrc
   source ~/.bashrc
   echo $HF_TOKEN  # should print your token
   ```

3. **Verify it worked:** `echo $HF_TOKEN` should print your token starting with `hf_`

**Gotcha:** If you skip this, cell 2 of the notebook will fail with `ERROR: HF_TOKEN not set`.

**Persistent benefit:** Because `~/.bashrc` persists across sessions, you only set this ONCE — even if you come back tomorrow, the token is still there.

---

## Step 5 — Download the Lightning notebook (1 min)

1. In the Studio, open a terminal (if not already open)
2. Run:

   ```bash
   cd /teamspace/studios/this_studio
   wget https://raw.githubusercontent.com/skyro777/Thursday_AI/main/training/thursday_ai_finetune_lightning.ipynb
   ```

3. The notebook will appear in your Studio's file browser (left sidebar, refresh if needed)

**Alternative:** Download from https://github.com/skyro777/Thursday_AI/blob/main/training/thursday_ai_finetune_lightning.ipynb to your local computer, then drag-and-drop into the Lightning Studio file browser.

---

## Step 6 — Open and run the notebook (8 hours)

1. Double-click `thursday_ai_finetune_lightning.ipynb` in the file browser
2. **Select kernel:** Top right of the notebook → pick the default Python 3 kernel (Lightning's pre-installed one)
3. **Run cell 1** (the deps install). Expected output:
   ```
   torch           2.5.1+cu121
   transformers    4.46.x
   trl             0.11.4
   peft            0.13.x
   accelerate      0.34.x
   bitsandbytes    0.45.3 (or 0.49.2)
   huggingface_hub 0.25.x
   datasets        2.20.x
   GPU count: 1
     GPU 0: NVIDIA L4
   OK: all imports succeeded. Proceed to cell 2.
   ```

4. **Run cell 2** (HF token check). Expected:
   ```
   Token is valid. Logged in as: <your-HF-username>
   Will upload merged model to: <username>/thursday-ai-v0.1-merged
   Will upload GGUF to:          <username>/thursday-ai-v0.1-gguf
   Token has WRITE permission
   ```

5. **Run cells 3-7** (repo clone, dataset gen, model load, LoRA attach, trainer setup). ~5 min total. The model load takes ~2 min the first time (downloading Qwen2.5-3B-Instruct ~6 GB), then it's cached.

6. **Run cell 8** (Train!). This is the big one. ~6-8 hours.
   - Expected first log lines:
     ```
     Num examples = 90,000 | Num Epochs = 3 | Total steps = ~16,875
     Batch size per device = 2 | Gradient accumulation steps = 8
     ```
   - You'll see a progress bar with `loss: X.XXX` decreasing
   - Loss should drop from ~2.0 to ~0.3-0.6 over the run

7. **Run cells 9-14** (save, merge, GGUF convert, upload, smoke test). ~15 min.

---

## Gotchas to avoid (we hit all of these on Kaggle)

1. **Don't close the browser tab during training.** Lightning keeps the Studio running even if you navigate away, but it's safest to leave the tab open. If you need to step away, just minimize — don't close.

2. **If you accidentally pick the wrong GPU (A100/H100),** you'll burn credits. Check the right sidebar — it should say `L4 (free)` or similar.

3. **Don't run out of disk space.** The model download + LoRA + merged model + GGUF = ~10 GB. Lightning free tier gives you 15 GB persistent storage. If you run multiple training attempts, delete the old `thursday-ai-merged` and `thursday-ai-lora` folders before re-running.

4. **If training fails midway,** don't restart from scratch. The notebook saves checkpoints every 500 steps to `thursday-ai-lora/checkpoint-XXXX`. You can resume by adding `resume_from_checkpoint=True` to `trainer.train()` in cell 8.

5. **Watch the GPU quota.** Right sidebar shows remaining GPU hours. If you're at 18h used out of 22h and training is at 50%, consider killing it and using a smaller dataset. Or pay $2 for Vast.ai spot instead.

6. **Internet is ON by default** on Lightning — no need to enable anything. Your HF Hub upload at the end will work.

---

## When training is done

After cell 13 (HF Hub upload), your GGUF will be live at:
```
https://huggingface.co/<your-hf-username>/thursday-ai-v0.1-gguf
```

On your potato PC:
```bash
ollama pull hf.co/<your-hf-username>/thursday-ai-v0.1-gguf:Q4_K_M
ollama run hf.co/<your-hf-username>/thursday-ai-v0.1-gguf:Q4_K_M
```

Then test:
```
>>> Hey Thursday, set an alarm for 7 am tomorrow.
```

Expected response (if training worked):
```json
{"name": "set_alarm", "arguments": {"hour": 7, "minute": 0, "label": "tomorrow 7am"}}
```

If you still get "I'm sorry, I don't have the capability..." — paste the training log (cells 8 and 14 output) and we'll diagnose.

---

## Quick troubleshooting

| Symptom | Fix |
|---|---|
| Cell 1 fails: bitsandbytes CUDA binary missing | Change `bitsandbytes==0.49.2` → `bitsandbytes==0.45.3` in the deps install line |
| Cell 2 fails: HF_TOKEN not set | Go back to Step 4 and set the env var properly |
| Cell 4 fails: OOM | Reduce `per_device_batch` from 2 to 1 in cell 0 CONFIG |
| Cell 8 takes longer than 10 hours | Check `smoke_dataset_size` and `epochs` in CONFIG — should be 90000 and 3 for full v0.1, or 5000 and 1 for smoke test |
| Cell 12 fails: make/CMake error | Make sure you're using the Lightning notebook (not the Kaggle one) — it has the CMake fix |
| Cell 13 fails: 403 Forbidden | Token doesn't have write permission — regenerate with "Write" or "Full Access" preset |

---

## TL;DR — the entire flow

```
1. Sign up lightning.ai (2 min)
2. New Studio → "thursday-ai" (2 min)
3. Right sidebar → Hardware → L4 GPU (1 min)
4. Terminal: echo 'export HF_TOKEN=hf_xxx' >> ~/.bashrc && source ~/.bashrc (1 min)
5. wget the lightning notebook (1 min)
6. Open notebook, Run All (8 hours)
7. Pull GGUF via Ollama on your potato PC
```

Total: ~10 min setup + 8 hours training. Cost: $0.
