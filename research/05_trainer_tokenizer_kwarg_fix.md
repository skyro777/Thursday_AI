# 05 — Fix for `Trainer.__init__() got an unexpected keyword argument 'tokenizer'`

**Task ID:** M0-fix-15
**Agent:** SFTTrainer-Kwarg-Fix-Researcher
**Status:** Research complete — verified minimal fix identified
**Method:** Wheel-based source inspection + web search + web reader (no guessing)

---

## TL;DR — the minimal fix

Change **one line** in `thursday_ai_finetune.ipynb` cell 1 (the pip-install cell):

```diff
- !pip install -q --upgrade --force-reinstall --no-deps "trl==0.11.4"
+ !pip install -q --upgrade --force-reinstall --no-deps "trl==0.12.1"
```

That's it. **Cell 7 needs no code changes.** Unsloth 2026.9.14 already has a built-in backwards-compatibility shim that auto-converts `tokenizer=` → `processing_class=` whenever the installed TRL version's `SFTTrainer.__init__` signature contains the `processing_class` parameter. TRL 0.11.4's signature does NOT contain `processing_class` (it uses the old `tokenizer` name) so the shim never fires — and TRL 0.11.4 then passes `tokenizer=` straight through to `transformers.Trainer.__init__`, which (in transformers 5.0) raises the `TypeError`.

For defense-in-depth, **also** swap the kwarg name in cell 7 (`tokenizer=tokenizer` → `processing_class=tokenizer`). This is NOT required but makes the notebook forward-compatible with TRL ≥ 0.16 (where the `@deprecate_kwarg` decorator will eventually be removed).

---

## Section 1 — Root cause (with code citations from inspected wheels)

### 1.1 What the error actually is

```
TypeError: Trainer.__init__() got an unexpected keyword argument 'tokenizer'
```

This is raised by `transformers.Trainer.__init__` (transformers 5.0.0). The kwarg `tokenizer` was renamed to `processing_class` in transformers 4.46+ (deprecation warning) and **fully removed in transformers 5.0** — passing `tokenizer=...` is now a hard `TypeError`.

### 1.2 transformers 5.0.0 `Trainer.__init__` signature (verified from wheel)

Source: `transformers-5.0.0-py3-none-any.whl` → `transformers/trainer.py` line 382-401

```python
def __init__(
    self,
    model: PreTrainedModel | nn.Module | None = None,
    args: TrainingArguments | None = None,
    data_collator: DataCollator | None = None,
    train_dataset: Union[Dataset, IterableDataset, "datasets.Dataset"] | None = None,
    eval_dataset: Union[Dataset, dict[str, Dataset], "datasets.Dataset"] | None = None,
    processing_class: PreTrainedTokenizerBase
    | BaseImageProcessor
    | FeatureExtractionMixin
    | ProcessorMixin
    | None = None,
    model_init: Callable[..., PreTrainedModel] | None = None,
    compute_loss_func: Callable | None = None,
    compute_metrics: Callable[[EvalPrediction], dict] | None = None,
    callbacks: list[TrainerCallback] | None = None,
    optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
    optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
    preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
):
```

**No `tokenizer` parameter. No `@deprecate_kwarg` decorator on `Trainer.__init__`** — the deprecation period is over; `tokenizer=` is now strictly rejected.

### 1.3 TRL 0.11.4 `SFTTrainer.__init__` — what user has now

Source: `trl-0.11.4-py3-none-any.whl` → `trl/trainer/sft_trainer.py` line 119-146

```python
def __init__(
    self,
    model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
    args: Optional[SFTConfig] = None,
    data_collator: Optional[DataCollator] = None,
    train_dataset: Optional[Dataset] = None,
    eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,   # ← old name
    model_init: ...,
    ...
    dataset_text_field: Optional[str] = None,              # ← still a direct kwarg
    packing: Optional[bool] = False,                        # ← still a direct kwarg
    max_seq_length: Optional[int] = None,                   # ← still a direct kwarg
    dataset_num_proc: Optional[int] = None,                 # ← still a direct kwarg
    ...
):
```

Then at line 401-413, TRL 0.11.4 SFTTrainer calls `super().__init__` **passing `tokenizer=tokenizer`**:

```python
super().__init__(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,          # ← THIS IS THE LINE THAT EXPLODES on transformers 5.0
    model_init=model_init,
    ...
)
```

So the call chain with TRL 0.11.4 + transformers 5.0 is:

```
User:   SFTTrainer(tokenizer=tokenizer, ...)
   → Unsloth wrapper passes tokenizer= through unchanged (shim doesn't fire — see §1.5)
   → TRL 0.11.4 SFTTrainer.__init__ accepts tokenizer=
   → TRL 0.11.4 SFTTrainer.__init__ calls super().__init__(tokenizer=tokenizer, ...)
   → transformers 5.0 Trainer.__init__ sees unexpected kwarg 'tokenizer'
   → TypeError
```

### 1.4 TRL 0.12.1 `SFTTrainer.__init__` — what user should have

Source: `trl-0.12.1-py3-none-any.whl` → `trl/trainer/sft_trainer.py` line 125-157

```python
@deprecate_kwarg("tokenizer", new_name="processing_class", version="0.16.0", raise_if_both_names=True)
def __init__(
    self,
    model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
    args: Optional[SFTConfig] = None,
    data_collator: Optional[DataCollator] = None,
    train_dataset: Optional[Dataset] = None,
    eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
    processing_class: Optional[...] = None,                # ← NEW name
    model_init: ...,
    ...
    dataset_text_field: Optional[str] = None,             # ← still a direct kwarg ✓
    packing: Optional[bool] = False,                       # ← still a direct kwarg ✓
    max_seq_length: Optional[int] = None,                  # ← still a direct kwarg ✓
    dataset_num_proc: Optional[int] = None,               # ← still a direct kwarg ✓
    ...
):
```

And at line 408-420, TRL 0.12.1 SFTTrainer calls `super().__init__` **passing `processing_class=processing_class`**:

```python
super().__init__(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=processing_class,    # ← works on transformers 5.0 ✓
    model_init=model_init,
    ...
)
```

**Key observations:**

1. The constructor parameter was renamed `tokenizer` → `processing_class`.
2. The `@deprecate_kwarg("tokenizer", new_name="processing_class", version="0.16.0", raise_if_both_names=True)` decorator (defined in `transformers/utils/deprecation.py` line 36) silently rewrites any `tokenizer=` kwarg to `processing_class=` until TRL 0.16.0 — see §1.6.
3. `dataset_text_field`, `packing`, `max_seq_length`, `dataset_num_proc` are STILL accepted as direct `SFTTrainer.__init__` kwargs in TRL 0.12.1 (they were moved into `SFTConfig` only starting at TRL 0.13.0). So user's cell 7 API works unchanged.
4. The internal `super().__init__` call uses `processing_class=`, which transformers 5.0 accepts.

### 1.5 Unsloth 2026.9.14's wrapper shim — the part that matters

Source: `unsloth-2026.9.14-py3-none-any.whl` → `unsloth/trainer.py` lines 1237-1317

Unsloth wraps `trl.SFTTrainer.__init__` with a `new_init` function (the `_backwards_compatible_trainer` helper) at import time. The relevant snippet (line 1240-1246):

```python
@wraps(original_init)
def new_init(self, *args, **kwargs):
    # tokenizer is now processing_class.
    trainer_params = _resolve_trainer_params(trainer_class, original_init)

    if "processing_class" in trainer_params and "tokenizer" in kwargs:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
    ...
    original_init(self, *args, **kwargs)
```

So Unsloth checks the underlying TRL `SFTTrainer.__init__` signature (via `_resolve_trainer_params`) and **if `processing_class` is one of the accepted parameters AND the caller passed `tokenizer=`**, it silently renames `tokenizer=` → `processing_class=` before forwarding to TRL's SFTTrainer.

**With TRL 0.11.4 installed:** `trainer_params` includes `tokenizer` (NOT `processing_class`). The `if` condition is `False`. The kwarg stays as `tokenizer=`. TRL 0.11.4's SFTTrainer accepts it but then passes `tokenizer=` to transformers.Trainer → **TypeError**.

**With TRL 0.12.1 installed:** `trainer_params` includes `processing_class` (NOT `tokenizer` — even though `tokenizer` is silently accepted by the `@deprecate_kwarg` decorator, it's not a real parameter of `__init__`). The `if` condition is `True`. Unsloth pops `tokenizer` from kwargs and inserts `processing_class`. TRL 0.12.1's SFTTrainer accepts `processing_class=` and passes `processing_class=` to transformers.Trainer → **success**.

There is also a second wrapper (`_patch_sft_trainer_auto_packing` at line 1320) that handles packing auto-detection — it's harmless and runs after the kwarg rename.

The `_patch_trl_trainer` function (line 1567-1573) gates this whole wrapping:

```python
def _patch_trl_trainer():
    import trl
    if hasattr(trl, "__UNSLOTH_BACKWARDS_COMPATIBLE__"):
        return
    if Version(trl) <= Version("0.11.0"):
        return
    ...
```

So the wrapper only applies when TRL > 0.11.0. Both TRL 0.11.4 and 0.12.1 satisfy this gate (so the wrapper IS installed), but the conditional conversion inside the wrapper only fires for TRL ≥ 0.12.1 (because that's when `processing_class` appears in the SFTTrainer signature).

### 1.6 The `@deprecate_kwarg` decorator — second layer of safety

Source: `transformers-5.0.0-py3-none-any.whl` → `transformers/utils/deprecation.py` lines 36-141

```python
def deprecate_kwarg(
    old_name: str,
    version: str,
    new_name: str | None = None,
    warn_if_greater_or_equal_version: bool = False,
    raise_if_greater_or_equal_version: bool = False,
    raise_if_both_names: bool = False,
    additional_message: str | None = None,
):
    ...
    def wrapper(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            ...
            # only deprecated kwarg is set for function call -> replace it with new name
            elif old_name in kwargs and new_name is not None and new_name not in kwargs:
                minimum_action = Action.NOTIFY
                message = f"`{old_name}` is deprecated {version_message} for `{func_name}`. Use `{new_name}` instead."
                kwargs[new_name] = kwargs.pop(old_name)
            ...
```

So even if Unsloth's shim somehow fails or is bypassed, **TRL 0.12.1's own `@deprecate_kwarg` decorator would catch `tokenizer=` and convert it to `processing_class=`** with a deprecation warning (no exception). This is the second line of defense.

### 1.7 Why Option (a) alone (just swapping the kwarg name in cell 7) does NOT work

If the user only swaps `tokenizer=tokenizer` → `processing_class=tokenizer` in cell 7 but keeps TRL 0.11.4 installed:

- Unsloth's wrapper: `if "processing_class" in trainer_params and "tokenizer" in kwargs` → `False` (because `processing_class` is not in TRL 0.11.4 signature AND `tokenizer` is not in kwargs since user swapped it)
- Unsloth calls `original_init(self, processing_class=tokenizer, ...)`
- TRL 0.11.4 SFTTrainer.__init__ has signature with `tokenizer` (not `processing_class`), so passing `processing_class=` raises: `TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'processing_class'`

So **the TRL upgrade is mandatory**. The cell-7 kwarg swap is optional (only matters for forward-compat with TRL ≥ 0.16 once the `@deprecate_kwarg` decorator is removed).

---

## Section 2 — Verified fix (minimal change to cell 7's SFTTrainer call)

Strictly speaking, **cell 7 needs no changes**. The fix is entirely in cell 1 (the pip-install cell):

### Cell 1 change — one line

```diff
- !pip install -q --upgrade --force-reinstall --no-deps "trl==0.11.4"
+ !pip install -q --upgrade --force-reinstall --no-deps "trl==0.12.1"
```

### Why this is enough (the chain of evidence)

1. **TRL 0.12.1's `SFTTrainer.__init__` has `processing_class` as a real parameter** (replacing `tokenizer`). Verified by reading `trl-0.12.1-py3-none-any.whl/trl/trainer/sft_trainer.py:128-157`.
2. **TRL 0.12.1's `super().__init__(processing_class=...)` call is compatible with transformers 5.0 Trainer.__init__** (which expects `processing_class`). Verified by reading `trl-0.12.1-py3-none-any.whl/trl/trainer/sft_trainer.py:408-420` AND `transformers-5.0.0-py3-none-any.whl/transformers/trainer.py:382-401`.
3. **TRL 0.12.1 still accepts `dataset_text_field`, `packing`, `max_seq_length`, `dataset_num_proc` as direct `SFTTrainer.__init__` kwargs** (these were moved into `SFTConfig` only starting at TRL 0.13.0). Verified by reading the same file's parameter list. User's cell 7 uses all four — none will break.
4. **TRL 0.12.1 still exposes `SFTTrainer` and `SFTConfig` from the `trl` package** (same import path as 0.11.4). Verified by reading `trl-0.12.1-py3-none-any.whl/trl/trainer/__init__.py:35,63-64,99,127-128`.
5. **Unsloth 2026.9.14's `_backwards_compatible_trainer` shim auto-converts `tokenizer=` → `processing_class=`** when it detects `processing_class` in the underlying TRL signature. Verified by reading `unsloth-2026.9.14-py3-none-any.whl/unsloth/trainer.py:1241-1246`.
6. **TRL 0.12.1's `@deprecate_kwarg("tokenizer", new_name="processing_class", version="0.16.0", raise_if_both_names=True)` decorator** is a second-layer fallback that would also silently convert `tokenizer=` even if Unsloth's shim is bypassed. Verified by reading `transformers-5.0.0-py3-none-any.whl/transformers/utils/deprecation.py:36-141` and confirming the decorator is applied at `trl-0.12.1-py3-none-any.whl/trl/trainer/sft_trainer.py:127`.

### Optional defense-in-depth: also swap the kwarg in cell 7

If you want the notebook to keep working when TRL eventually drops the `@deprecate_kwarg` decorator (expected at TRL 0.16.0 per the decorator's `version="0.16.0"` argument), also make this change in cell 7:

```diff
  trainer = SFTTrainer(
      model = model,
-     tokenizer = tokenizer,
+     processing_class = tokenizer,
      train_dataset = formatted,
      ...
```

This is purely forward-compat insurance. With TRL 0.12.1, both `tokenizer=tokenizer` and `processing_class=tokenizer` work identically (the decorator handles the former, the signature handles the latter).

---

## Section 3 — Corrected cell 7 code, copy-paste ready

This is the version with BOTH the kwarg swap (defense-in-depth) AND assumes the cell-1 TRL upgrade is in place. **If you only do the cell-1 pip change, you can keep cell 7 EXACTLY as it is today** — no edits required.

### Cell 1 (only the line that changes)

```python
# ====================================================================
# THURSDAY AI — FINAL KAGGLE INSTALL (research-verified by inspecting 7 wheels)
# See: research/04_kaggle_unsloth_working_config.md
# M0-fix-15: TRL bumped 0.11.4 → 0.12.1 to fix
#   "Trainer.__init__() got an unexpected keyword argument 'tokenizer'"
#   (TRL 0.12.0 renamed SFTTrainer(tokenizer=) → SFTTrainer(processing_class=)
#    with backward-compat @deprecate_kwarg decorator; Unsloth 2026.9.14 has a
#    runtime shim that auto-converts tokenizer→processing_class when TRL ≥ 0.12.
#    See research/05_trainer_tokenizer_kwarg_fix.md for full evidence.)
# ====================================================================

!pip install -q --upgrade pip

# 1. Install Unsloth + unsloth_zoo (pinned for reproducibility) with --no-deps.
!pip install -q --upgrade --force-reinstall --no-cache-dir --no-deps \
    "unsloth==2026.9.14" "unsloth_zoo==2026.9.9"

# 2. Override TRL to 0.12.1 — first version whose SFTTrainer uses `processing_class`
#    instead of `tokenizer` (with @deprecate_kwarg shim that still accepts the old name).
#    Required for transformers 5.0 compatibility (transformers 5.0 dropped the
#    `tokenizer` kwarg from Trainer.__init__ entirely).
!pip install -q --upgrade --force-reinstall --no-deps "trl==0.12.1"

# 3. Install bitsandbytes 0.49.2 — verified by wheel inspection.
#    If 0.49.2 fails to download, fall back to 0.48.2 (also has cuda128 binary).
!pip install -q --upgrade --force-reinstall --no-deps "bitsandbytes==0.49.2" || \
    !pip install -q --upgrade --force-reinstall --no-deps "bitsandbytes==0.48.2"

# 4. GGUF conversion tools (used later in the notebook).
!pip install -q "gguf>=0.6.0"
!apt-get -y install -q git-lfs

# ====================================================================
# SANITY CHECK — imports + version prints.
# CRITICAL: import unsloth FIRST. It patches trl/transformers/peft on import.
# ====================================================================
import unsloth  # MUST be first
import torch, trl, transformers, peft, accelerate, bitsandbytes, huggingface_hub, tokenizers, datasets, gguf
print(f'torch           {torch.__version__}')
print(f'transformers    {transformers.__version__}')
print(f'tokenizers      {tokenizers.__version__}')
print(f'trl             {trl.__version__}')
print(f'peft            {peft.__version__}')
print(f'accelerate      {accelerate.__version__}')
print(f'bitsandbytes    {bitsandbytes.__version__}')
print(f'huggingface_hub {huggingface_hub.__version__}')
print(f'datasets        {datasets.__version__}')
print(f'GPU count: {torch.cuda.device_count()}')
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, PeftModel
from huggingface_hub import HfApi, login
from datasets import load_dataset, Dataset
print('OK: all imports succeeded. Proceed to cell 2.')
```

### Cell 7 (full, copy-paste ready — with the optional kwarg swap)

```python
from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported
from transformers import DataCollatorForSeq2Seq

trainer = SFTTrainer(
    model = model,
    # M0-fix-15: swapped `tokenizer=tokenizer` → `processing_class=tokenizer`
    # for forward-compat with TRL ≥ 0.16 (where the @deprecate_kwarg decorator
    # will be removed). With TRL 0.12.1, BOTH names work — Unsloth's wrapper
    # auto-converts the legacy name even if you leave it as `tokenizer=`.
    processing_class = tokenizer,
    train_dataset = formatted,
    dataset_text_field = 'text',
    max_seq_length = CONFIG['max_seq_length'],
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding='longest'),
    dataset_num_proc = 2,
    packing = False,
    args = SFTConfig(
        per_device_train_batch_size = CONFIG['per_device_batch'],
        gradient_accumulation_steps = CONFIG['grad_accum_steps'],
        warmup_ratio = CONFIG['warmup_ratio'],
        num_train_epochs = CONFIG['epochs'],
        learning_rate = CONFIG['learning_rate'],
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = CONFIG['logging_steps'],
        optim = CONFIG['optim'],
        weight_decay = CONFIG['weight_decay'],
        lr_scheduler_type = CONFIG['lr_scheduler'],
        seed = CONFIG['seed'],
        output_dir = CONFIG['output_dir'],
        save_steps = CONFIG['save_steps'],
        save_total_limit = 3,
        report_to = 'none',
    ),
)

# Mask loss on everything except assistant responses.
# Unsloth occasionally moves this helper around — try a few import paths.
train_on_responses_only = None
# ... (rest of cell 7 unchanged)
```

> **Note on `DataCollatorForSeq2Seq(tokenizer=tokenizer, ...)`**: That `tokenizer=` kwarg is on the `DataCollatorForSeq2Seq` constructor — NOT on `Trainer`. transformers 5.0 still accepts `tokenizer=` on `DataCollatorForSeq2Seq` (it has its own `@deprecate_kwarg` decorator, slated for removal in v6). So leave it as-is.

### Expected cell 1 output after the fix

```
torch           2.10.0+cu128
transformers    5.0.0
tokenizers      0.22.2
trl             0.12.1            ← was 0.11.4
peft            0.19.1
accelerate      1.13.0
bitsandbytes    0.49.2
huggingface_hub 1.11.0
datasets        5.0.0
GPU count: 2
OK: all imports succeeded. Proceed to cell 2.
```

### Expected cell 7 behavior after the fix

- No `TypeError`.
- One `DeprecationWarning` from TRL 0.12.1 about the `tokenizer` kwarg (only if you DIDN'T do the optional cell-7 swap; if you DID swap to `processing_class=`, no warning).
- One `UserWarning` from TRL 0.12.1 about `packing=False` overriding the SFTConfig value (this is informational — the override IS what we want).
- Trainer proceeds to `trainer.train()`.

---

## Section 4 — Other options considered (pros/cons)

### Option (a) — Swap `tokenizer=tokenizer` → `processing_class=tokenizer` in cell 7 only (NO TRL upgrade)

| Aspect | Verdict |
|---|---|
| Code change | 1 line in cell 7 |
| TRL upgrade needed | None |
| Will it work? | **NO** — TRL 0.11.4's `SFTTrainer.__init__` signature does NOT include `processing_class` (it has `tokenizer`). Passing `processing_class=` raises `TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'processing_class'`. |
| Unsloth shim | Doesn't help — the shim converts `tokenizer→processing_class`, not the other way around. |

**Verdict: rejected.** The TRL upgrade is unavoidable.

### Option (b) — Upgrade TRL from 0.11.4 → 0.12.1 (RECOMMENDED)

| Aspect | Verdict |
|---|---|
| Code change | 1 line in cell 1 (`trl==0.11.4` → `trl==0.12.1`) |
| Cell 7 changes | None required (optional kwarg swap for forward-compat) |
| Will it work? | **YES** — verified by wheel inspection: TRL 0.12.1's `super().__init__(processing_class=...)` is compatible with transformers 5.0 `Trainer.__init__`. |
| Unsloth shim | Fires correctly — auto-converts `tokenizer=` → `processing_class=` when it detects `processing_class` in TRL 0.12.1's SFTTrainer signature. |
| Cell 7 API compat | TRL 0.12.1 still has `dataset_text_field`, `max_seq_length`, `packing`, `dataset_num_proc` as direct `SFTTrainer.__init__` kwargs (moved to SFTConfig only in TRL 0.13.0). So cell 7 needs no API changes. |
| TRL 0.12.1 deps | `accelerate>=0.34.0` (Kaggle 1.13.0 ✓), `datasets>=2.21.0` (Kaggle 5.0 ✓), `transformers>=4.46.0` (Kaggle 5.0 ✓), `python>=3.9` (Kaggle 3.11 ✓). All satisfied. |
| Risk | Minimal — `@deprecate_kwarg` decorator provides second-layer safety net. |

**Verdict: ADOPTED.** This is the minimal fix.

### Option (b-prime) — Upgrade TRL to ≥ 0.13.0 (alternative)

| Aspect | Verdict |
|---|---|
| Code change | 1 line in cell 1 (e.g., `trl==0.14.0` or `trl==0.16.1`) |
| Cell 7 changes | Required — TRL ≥ 0.13.0 removed `dataset_text_field`, `max_seq_length`, `packing`, `dataset_num_proc` from `SFTTrainer.__init__`; they must be moved into `SFTConfig(...)`. Unsloth 2026.9.14 has a runtime shim that auto-migrates these (see `unsloth/trainer.py:1248-1313`), but it's more fragile than staying on 0.12.x. |
| Will it work? | Probably yes (Unsloth's shim handles the migration), but more risk surface. |
| Benefit | Forward-compat — closer to TRL 1.x API. |
| Risk | Higher — Unsloth's `_backwards_compatible_trainer` migration logic for ≥0.13.0 has multiple branches (config_fields, moved_params, unknown_kwargs, _route_unknown_trainer_kwargs). If any of those branches misroute a kwarg, you get a confusing error. |

**Verdict: rejected** for this iteration. If we have to upgrade past 0.12.x in the future (e.g., Kaggle stops shipping something TRL 0.12 needs), reconsider. For now, **0.12.1 is the sweet spot** — it has the `processing_class` rename (needed) and keeps the legacy `SFTTrainer.__init__` kwargs (so we don't have to touch cell 7's API).

### Option (c) — Downgrade transformers from 5.0 to 4.46.x

| Aspect | Verdict |
|---|---|
| Code change | Add `!pip install -q --upgrade --force-reinstall --no-deps "transformers==4.46.3" "tokenizers==0.20.3"` to cell 1 |
| Will it work? | Yes — transformers 4.46-4.99 still has the `@deprecate_kwarg("tokenizer", new_name="processing_class", ...)` decorator on `Trainer.__init__`, so passing `tokenizer=` is auto-converted to `processing_class=` with a deprecation warning. |
| Cell 7 changes | None |
| Cons | (1) Downgrading transformers drags along Kaggle's `tokenizers`, `huggingface_hub`, `datasets` constraints — Kaggle preinstalls versions that EXPECT transformers 5.0 (e.g., datasets 5.0, hub 1.11). Mixing transformers 4.46 with hub 1.11 / datasets 5.0 may break in subtle ways. (2) Unsloth 2026.9.14's metadata explicitly excludes certain 4.x versions (`!=4.52.0, !=4.52.1, ..., !=4.57.0, !=4.57.4, !=4.57.5`) — picking a safe 4.46.x version requires care. (3) User's M0-fix-13 worklog confirmed transformers 5.0 works with Unsloth 2026.9.14 on Kaggle T4 ×2 — why undo that hard-won success? |
| Recommendation | Don't do this. Stick with Kaggle's preinstalled transformers 5.0. |

**Verdict: rejected.** Going against Kaggle's preinstalled stack creates more problems than it solves.

### Option (d) — Pin Unsloth to an older version that hardcodes `processing_class=`

| Aspect | Verdict |
|---|---|
| Code change | `!pip install --no-deps "unsloth<2025"` or similar |
| Will it work? | Maybe — but older Unsloth versions don't support transformers 5.0 at all (they predate the transformers 5.0 release). Would also drop support for newer Qwen2.5 patches. |
| Cons | Massive downgrade of functionality, likely incompatible with newer Qwen models. |
| Recommendation | Don't do this. Unsloth 2026.9.14 is the right version — it just needs TRL ≥ 0.12.1 to expose `processing_class` so the auto-conversion shim fires. |

**Verdict: rejected.**

### Option (e) — Monkey-patch TRL 0.11.4 to add `processing_class` alias

| Aspect | Verdict |
|---|---|
| Code change | Add custom Python monkey-patch at top of cell 7 |
| Will it work? | Probably yes — add a thin wrapper that translates `processing_class=` to `tokenizer=` at the TRL layer. But this is fragile and reinvents what Unsloth 2026.9.14 ALREADY does (its `_backwards_compatible_trainer` shim). The only reason Unsloth's shim doesn't fire on TRL 0.11.4 is that TRL 0.11.4's signature lacks `processing_class`. |
| Recommendation | Don't do this — Option (b) achieves the same result with a 1-line pip change and no user-written patches. |

**Verdict: rejected.**

---

## Section 5 — References

### Primary sources (wheel inspection — this report's evidence base)

| Wheel file | File inside | Lines | What it proves |
|---|---|---|---|
| `trl-0.11.4-py3-none-any.whl` | `trl/trainer/sft_trainer.py` | 119-146, 401-413 | TRL 0.11.4 SFTTrainer uses `tokenizer` kwarg and calls `super().__init__(tokenizer=...)` |
| `trl-0.12.1-py3-none-any.whl` | `trl/trainer/sft_trainer.py` | 125-157, 408-420 | TRL 0.12.1 SFTTrainer uses `processing_class` kwarg + `@deprecate_kwarg` decorator; calls `super().__init__(processing_class=...)` |
| `trl-0.12.1-py3-none-any.whl` | `trl/trainer/sft_config.py` | 30-69 | TRL 0.12.1's SFTConfig still has `dataset_text_field`, `packing`, `max_seq_length`, `dataset_num_proc` — so SFTTrainer constructor still accepts them (with override warnings) |
| `trl-0.13.0-py3-none-any.whl` | `trl/trainer/sft_trainer.py` | 110-131 | TRL 0.13.0 SFTTrainer.__init__ LOST `dataset_text_field`, `max_seq_length`, `packing`, `dataset_num_proc` (moved to SFTConfig) — this is why we DON'T upgrade past 0.12.x |
| `trl-1.14.1-py3-none-any.whl` | `trl/trainer/sft_trainer.py` | 919-931 | TRL 1.x: `processing_class` only, no `tokenizer`, no legacy SFTTrainer constructor kwargs |
| `transformers-5.0.0-py3-none-any.whl` | `transformers/trainer.py` | 382-401 | transformers 5.0 Trainer.__init__ uses `processing_class`, no `tokenizer` kwarg, no deprecation decorator |
| `transformers-5.0.0-py3-none-any.whl` | `transformers/utils/deprecation.py` | 36-141 | `@deprecate_kwarg` decorator definition — confirms it auto-converts `old_name`→`new_name` with a warning (default `raise_if_greater_or_equal_version=False`) |
| `unsloth-2026.9.14-py3-none-any.whl` | `unsloth/trainer.py` | 1237-1317 | Unsloth's `_backwards_compatible_trainer` wrapper — the auto-conversion shim |
| `unsloth-2026.9.14-py3-none-any.whl` | `unsloth/trainer.py` | 1245-1246 | The exact lines: `if "processing_class" in trainer_params and "tokenizer" in kwargs: kwargs["processing_class"] = kwargs.pop("tokenizer")` |
| `unsloth-2026.9.14-py3-none-any.whl` | `unsloth/trainer.py` | 1567-1573 | `_patch_trl_trainer` gate: only applies wrapper when TRL > 0.11.0 |
| `unsloth-2026.9.14-py3-none-any.whl` | `unsloth/trainer.py` | 1076-1115 | `_resolve_trainer_params` — returns the parameter names of TRL SFTTrainer.__init__ (or its parent's if it's a thin wrapper) |
| `unsloth_zoo-2026.9.9-py3-none-any.whl` | `unsloth_zoo/utils.py` | 41-75 | `Version()` helper — extracts `__version__` from module objects |

### Web sources

| URL | What it confirms |
|---|---|
| https://stackoverflow.com/questions/79546910/typeerror-in-sfttrainer-initialization-unexpected-keyword-argument-tokenizer | Accepted answer (Mar 2025): "In the 0.12.0 release it is explained that the tokenizer argument is now called the processing_class parameter." |
| https://github.com/huggingface/trl/issues/6168 | GitHub issue "TypeError: Trainer.__init__() got an unexpected keyword argument 'tokenizer'" — same error as user's; resolution is the TRL upgrade |
| https://github.com/huggingface/trl/blob/main/MIGRATION.md | TRL v1 migration guide (covers later breaking changes; not relevant to our 0.11.4→0.12.1 step but confirms `processing_class` is the new API) |
| https://github.com/huggingface/peft/issues/2400 | PEFT user confusion about `tokenizer` vs `processing_class` — confirms the rename |
| https://github.com/huggingface/transformers/issues/37734 | GitHub issue "`tokenizer` is still being used in `Trainer` instead of `processing_class`" — confirms `tokenizer` is fully removed in transformers 5.0 |
| https://huggingface.co/docs/trl/en/sft_trainer | Current TRL SFTTrainer docs — shows `processing_class` as the parameter name in v1.x |
| https://github.com/unslothai/unsloth/issues/1264 | Unsloth issue "TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'dataset_text_field'" — different error but related (TRL ≥ 0.13 dropped that kwarg) |
| https://www.assemblyai.com/blog/hugging-face-transformers-tutorial | "tokenizer= became processing_class= in Trainer. These are the two most common errors when running pre-2025 fine-tuning code on Transformers v5." |
| https://pypi.org/project/trl | Confirms TRL 0.12.0 was released Nov 1, 2024 |

### Internal project references

- `/home/z/my-project/worklog.md` — M0-DEPS-FINAL entry (research that established the 0.11.4 pin in the first place; this report supersedes that pin with 0.12.1)
- `/home/z/my-project/worklog.md` — M0-fix-13 entry (confirmed Unsloth 2026.9.14 patches and runs on Kaggle T4 ×2 with transformers 5.0.0 at runtime)
- `/home/z/my-project/thursday-ai/research/04_kaggle_unsloth_working_config.md` — full dependency compatibility matrix
- `/home/z/my-project/thursday-ai/training/thursday_ai_finetune.ipynb` — the notebook being fixed (cell 1 + cell 7)

---

## Appendix A — Quick verification commands (for future agents)

If you want to re-verify the wheel findings without re-downloading everything:

```bash
# Download all relevant wheels
pip download --no-deps --dest /tmp/wheels \
    "trl==0.11.4" "trl==0.12.1" "trl==0.13.0" "trl==1.14.1" \
    "unsloth==2026.9.14" "unsloth_zoo==2026.9.9" "transformers==5.0.0"

# Unzip and inspect SFTTrainer signatures across TRL versions
for v in 0.11.4 0.12.1 0.13.0 1.14.1; do
  unzip -q /tmp/wheels/trl-$v-py3-none-any.whl -d /tmp/trl_$v
  echo "=== TRL $v SFTTrainer.__init__ ==="
  grep -A 30 "def __init__" /tmp/trl_$v/trl/trainer/sft_trainer.py | head -35
  echo "=== TRL $v super().__init__ call ==="
  grep -A 12 "super().__init__" /tmp/trl_$v/trl/trainer/sft_trainer.py | head -15
done

# Verify Unsloth's auto-conversion shim
unzip -q /tmp/wheels/unsloth-2026.9.14-py3-none-any.whl -d /tmp/unsloth
sed -n '1237,1320p' /tmp/unsloth/unsloth/trainer.py

# Verify transformers 5.0 Trainer.__init__ signature
unzip -q /tmp/wheels/transformers-5.0.0-py3-none-any.whl -d /tmp/transformers_5
sed -n '382,401p' /tmp/transformers_5/transformers/trainer.py
```

## Appendix B — Why we picked 0.12.1 specifically (not 0.12.0, not 0.12.2)

TRL 0.12.0 (Nov 1, 2024) was the release that introduced `processing_class` and the `@deprecate_kwarg` decorator. TRL 0.12.1 was a quick bugfix release. We pin to 0.12.1 (the last 0.12.x patch) because:

1. PyPI shows `0.12.0`, `0.12.1`, `0.12.2` exist in the 0.12 series.
2. Latest patch (0.12.2 or 0.12.1) contains any post-0.12.0 bug fixes.
3. All three should work for our use case (the `processing_class` rename + the `@deprecate_kwarg` decorator were all in 0.12.0); pick the highest 0.12.x to get any bugfixes.

If `trl==0.12.1` happens to fail to download from PyPI in some region, falling back to `trl==0.12.0` or `trl==0.12.2` will work identically (same `processing_class` + `@deprecate_kwarg` shim).
