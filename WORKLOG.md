# Thursday AI - Build Worklog

This is the shared worklog for the Thursday AI project. Each agent (subagent or main builder) appends a section after finishing a task. The latest section is at the bottom.

---
Task ID: 1
Agent: Main Builder (GLM 5.2)
Task: Clone the Thursday_AI GitHub repo into the sandbox and prepare the working directory.

Work Log:
- Cloned https://github.com/skyro777/Thursday_AI.git into /home/z/my-project/thursday-ai
- Set local git config (user.email + user.name) for commits
- Verified repo only contains LICENSE + README (clean slate)
- Created this worklog file at /home/z/my-project/worklog.md (root project) and will also keep /home/z/my-project/thursday-ai/WORKLOG.md mirrored

Stage Summary:
- Working directory: /home/z/my-project/thursday-ai
- Repo is clean; ready for first content push
- Next: launch parallel research subagents (Task IDs 2-a, 2-b, 2-c) and start writing the living plan

---
Task ID: 2-a
Agent: HF-Model-Researcher
Task: Find best HuggingFace base model for Thursday AI (potato-PC offline agentic assistant, fine-tunable on T4x2)

Work Log:
- Read existing worklog (Task 1 — repo clone done by Main Builder)
- Created /home/z/my-project/thursday-ai/research/ directory
- Used z-ai web_search to discover candidate models across Tier A (3B-4B), Tier B (7B-8B), Tier C (specialized agentic)
- Visited and parsed HuggingFace model cards for: Qwen/Qwen2.5-3B-Instruct, Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-Coder-7B-Instruct, meta-llama/Llama-3.2-3B-Instruct, meta-llama/Llama-3.1-8B-Instruct, microsoft/Phi-3.5-mini-instruct, google/gemma-2-2b-it, mistralai/Mistral-7B-Instruct-v0.3, NousResearch/Hermes-3-Llama-3.1-8B, unsloth/Qwen2.5-3B-Instruct
- Parsed bartowski GGUF repos for exact Q4_K_M file sizes (Qwen2.5-3B=1.93GB, Qwen2.5-7B=4.7GB, Llama-3.2-3B=2.02GB, Llama-3.1-8B=4.92GB, Mistral-7B-v0.3=4.37GB, Hermes-3-8B=4.92GB)
- Verified Ollama library availability for each (qwen2.5:3b-instruct-q4_K_M, qwen2.5:7b-instruct-q4_K_M, etc.)
- Pulled BFCL v3 leaderboard from benchmarklist.com: Llama-3.2-3B (FC)=45.86%, Llama-3.1-8B (FC)=25.92%, Llama-3.1-8B (Prompt)=49.57%
- Pulled SOTA2 BFCL Overall Nov-2025: Qwen2.5-7B-Instruct raw=59.5 (AST 63.3, Exec 93.6, Irrel 72.1), Qwen2.5-3B-Instruct raw=31.81, Llama3.2-3B-Instruct raw=40.5
- Pulled Emergent Mind BFCL v4: Qwen2.5-Coder-7B-Instruct (FunRL)=86.02 (top-tier open-source FC)
- Read Llama 3.2 launch blog (huggingface.co/blog/llama32) confirming "Tool use is supported in both 1B and 3B instruction-tuned models" (zero-shot) and Llama-3.2 EU license restriction
- Read Microsoft Phi-3.5-mini discussion/7 confirming NO official tool/function-call support
- Read Hermes-3 model card: documented full Hermes Function Calling standard (XML <tools> + JSON schema)
- Pulled Unsloth Kaggle notebooks: confirmed unsloth/Qwen2.5-3B-Instruct and Llama-3.2-3B have free Kaggle T4 fine-tune notebooks with 4-bit QLoRA support
- Pulled CPU inference benchmarks: github.com/ggml-org/llama.cpp/issues/34 (i5/2-core/8GB → ~94ms/tok), roger.lol/blog (4B Q4 = 7.3 tok/s), localaimaster.com (usable ≤4B, tolerable 7B on DDR5), r/LocalLLaMA (7B Q4 K_M CPU = 2-3 tok/s)
- KEY HARDWARE FINDING: i5 3rd-gen Ivy Bridge supports AVX but NOT AVX2 (AVX2 arrived with Haswell 2013). Must build llama.cpp with LLAMA_AVX2=OFF LLAMA_AVX=ON LLAMA_FMA=ON. Ollama Windows installer has noavx fallback.
- Wrote detailed markdown report to /home/z/my-project/thursday-ai/research/01_base_model_selection.md (~425 lines, 9 model candidates, full quant tables, BFCL scores, license strings, CPU benchmarks, source URLs)

Stage Summary:
- Final recommendation: Qwen/Qwen2.5-3B-Instruct @ Q4_K_M (1.93 GB) on llama.cpp CPU AVX build (or Ollama wrapper)
- Tier A pick: Qwen/Qwen2.5-3B-Instruct (Apache 2.0, 3.09B params, 32K→128K ctx, native tool_calls JSON, 1.93GB Q4_K_M, Unsloth Kaggle T4 fine-tune ready, 29+ langs)
  - Runner-up A: meta-llama/Llama-3.2-3B-Instruct (better raw BFCL FC=45.86%, but Llama-3.2 license EU-blocked + AUP anti-automation clauses)
- Tier B pick: Qwen/Qwen2.5-7B-Instruct (Apache 2.0, 7.61B params, 131K native ctx, BFCL raw=59.5, same chat template as 3B sibling → free upgrade path)
  - Runner-up B: mistralai/Mistral-7B-Instruct-v0.3 (Apache 2.0, official FC support added in v0.3, 4.37GB Q4_K_M)
- Tier C pick: NousResearch/Hermes-3-Llama-3.1-8B (best-in-class Hermes FC standard XML+JSON, ChatML, 4.92GB Q4_K_M, but Llama-3 license + heavy on 8GB RAM → use as fallback on 16GB+ machines)
  - Runner-up C: Qwen/Qwen2.5-Coder-7B-Instruct (BFCL=86.02 with FunRL — best FC score, but code-specialised → better as delegated secondary model than primary voice-brain)
- Key tradeoffs:
  - 3B vs 7B: 3B fits 8GB RAM comfortably and gives 6-10 tok/s (good for voice); 7B swap-thrashes (4.7GB Q4_K_M + KV cache + OS) and only 2.5-4 tok/s. Quality gap is closed by QLoRA fine-tune on OS-control traces.
  - Qwen2.5-3B vs Llama-3.2-3B: Llama has higher raw FC score (45.86% vs 31.81%) but Llama-3.2 license is EU-blocked and AUP discourages system-automation use cases. Qwen's Apache 2.0 + multilingual (29 langs) + cleaner upgrade path won.
  - Tool-use native support: Qwen2 (yes), Llama-3.x (yes, zero-shot), Mistral v0.3 (yes, with tool IDs), Hermes (yes, best format), Phi-3.5 (NO), Gemma-2 (NO). Phi and Gemma eliminated for lack of native FC.
  - Fine-tune path: Unsloth officially supports Qwen2.5-3B/7B, Llama-3.1-8B, Llama-3.2-3B on Kaggle T4 with 4-bit QLoRA. T4x2 (16GBx2 = 32GB VRAM pooled) is sufficient for full LoRA fine-tune of any of these.
- Full report saved at: /home/z/my-project/thursday-ai/research/01_base_model_selection.md

---
Task ID: 2-b
Agent: PC-Control-TrainingData-Researcher
Task: Research how to construct training data for an autonomous OS/browser-controlling LLM

Work Log:
- Read existing worklog (Task 1 repo clone + Task 2-a base-model pick: Qwen/Qwen2.5-3B-Instruct, Unsloth 4-bit QLoRA on Kaggle T4x2)
- Created /home/z/my-project/thursday-ai/research/search_results/ and ran 6 z-ai web_search calls (one per query), saving JSON each
- Search 1 "HuggingFace computer use agent dataset 2024" -> Datoric/computer-use-agent-traces-250k, xlangai/CUA-Gym, ranpox/awesome-computer-use-agents, MS Fara1.5
- Search 2 "Open Interpreter training data format" -> confirmed Open Interpreter ships NO public training set; only inference runtime; best practices from OpenAI fine-tuning guide
- Search 3 "Mind2Web WebVoyager SeeAct dataset format" -> osunlp/Mind2Web (2,350 tasks/235k steps, CC-BY-4.0), SeeAct (GPT-4V web agent), NeurIPS 2023 paper
- Search 4 "Qwen2 tool calling chat template format jsonl Hermes" -> qwen.readthedocs.io function_call docs confirm Qwen2 uses Hermes-variant template (<tools> block, stringified JSON arguments); vLLM tool_calling docs corroborate
- Search 5 "Salesforce xLAM function calling dataset huggingface" -> Salesforce/xlam-function-calling-60k (60k examples, Apache 2.0-ish, JSON {query, tools, answers}, APIGen pipeline); HF cookbook has QLoRA fine-tune notebook
- Search 6 "synthetic agentic trajectory dataset generation LLM self-instruct" -> Self-Instruct, Evol-Instruct, CoT-Self-Instruct (arxiv 2507.23751), AgentInstruct (MS 1M), distillation patterns
- Cross-referenced 6 projects for Section 1 landscape: Open Interpreter, OSWorld, OmniParser, UFO, Self-Operating Computer, Claude Computer Use
- Drafted Thursday AI dataset schema (JSONL: system + tools + messages) with 2 full multi-turn trajectories: (a) "set alarm 7am" using set_alarm high-level tool, (b) "find a YouTube video I described" using browser_open + screenshot + click + type_text loop
- Designed action vocabulary of ~26 tools across 7 categories: window mgmt, mouse/keyboard, filesystem, browser, Thursday AI internal subsystems (paste_to_webai, read_webai_response), OS convenience (set_alarm, set_volume, notify_user), conversation/safety (ask_user, finish)
- Specified synthetic-data plan with 3 methods: (A) template-based (deterministic Jinja2 skeletons, 25k examples), (B) LLM-assisted via stronger model with Evol-Instruct + JSON-schema validation (35k + 12k amplified), (C) recorded Playwright/pyautogui sessions (5k real traces). Plus mixed-in xLAM (8k), Mind2Web (4k), WebVoyager (1k).
- Sizing reasoning: T4x2 32GB VRAM, Unsloth QLoRA on 3B, ~10k samples/hr throughput; targeted 90k total examples x 3 epochs = ~10k steps, fits single 9-hr Kaggle session (with checkpoint-resume fallback). Phase 0 smoke test 1k, Phase 1 format acquisition 15k x 2 ep, Phase 2 generalisation 90k x 3 ep. Flagged Phase 3 RL via CUA-Gym for later.
- Wrote detailed markdown report to /home/z/my-project/thursday-ai/research/02_training_data_approach.md (~470 lines, 6 sections + appendix of sources)

Stage Summary:
- Key existing projects: Open Interpreter (local code-running LLM, no FC training data), OSWorld (eval harness 369 tasks on Win/macOS/Ubuntu), OmniParser (MS screen parser — usable as a Thursday AI tool), UFO (MS Windows agent w/ best reference action vocab), Self-Operating Computer (OthersideAI pyautogui CLI), Claude Computer Use (Anthropic closed but action-space reference). None ships a reusable training dataset -> must assemble our own.
- Key datasets on HF: Salesforce/xlam-function-calling-60k (60k, Apache 2.0, JSON FC format primer), osunlp/Mind2Web (2,350 web tasks, CC-BY-4.0), osunlp/WebVoyager (643 screenshot-conditioned web trajectories), OSU SeeAct silver traces, Datoric/computer-use-agent-traces-250k (largest CUA trace corpus late-2025), xlangai/CUA-Gym (RLVR reward harness for future Phase 3).
- Proposed Thursday AI dataset schema: JSONL one example per line with 3 top-level fields {system, tools, messages}. tools = OpenAI-style JSONSchema array rendered by Qwen2 Hermes-variant chat template. messages = user/assistant/tool turns where assistant.tool_calls[].function.arguments is a STRINGIFIED JSON and each tool_call.id must be echoed in a tool-role message. Two full worked trajectories included: set_alarm 7am (single tool call) and find-a-YouTube-video (6-step screenshot+click+type loop). Action vocabulary = ~26 tools across 7 categories.
- Synthetic data generation plan: 3 methods, all used in mix. (A) Template-based Jinja2 skeletons, 25k deterministic format-correct examples. (B) LLM-assisted via teacher model (Qwen2.5-72B / GLM-4.5 via z-ai SDK, Apache-2.0 licensed only) with Evol-Instruct mutation + schema validation, ~47k examples. (C) Recorded Playwright + pyautogui sessions across Win/macOS/Linux, 5k real traces amplified via Method B distillation to 12k. Final mixture: 90k total (28% templates / 52% teacher-distilled / 6% real / 14% HF-converted).
- Recommended target dataset size: 90,000 examples x 3 epochs on T4x2 Unsloth QLoRA = ~10k optimizer steps, fits single 9-hr Kaggle session. Phase 0 smoke test (1k) -> Phase 1 format acquisition (15k x 2 ep) -> Phase 2 generalisation (90k x 3 ep) -> Phase 3 RL via CUA-Gym (out of scope, future).
- Full report saved at: /home/z/my-project/thursday-ai/research/02_training_data_approach.md

---
Task ID: 3-9 (Main Builder work, post-research)
Agent: Main Builder (GLM 5.2)
Task: Write the living plan, architecture, dataset schema, generators, validator, Kaggle notebook, README; push to GitHub

Work Log:
- Read all 3 research reports (01 base model, 02 training data, 03 Kaggle recipe) — all selected Qwen2.5-3B-Instruct @ Q4_K_M via Unsloth QLoRA on T4x2
- Wrote LIVING_PLAN.md (21KB, v0.1) — 12 sections: pitch, north-star goals, scope, system map, model decision, action vocabulary, dataset spec, build roadmap M0-M7, risks, naming, repo layout, decision history
- Wrote ARCHITECTURE.md (15KB) — 10 sections: training vs inference diagrams, think-act-observe loop pseudocode, tool-call lifecycle, web-AI delegation pattern, safety architecture, memory (M6), inference engines, training pipeline, performance budget, failure modes
- Wrote data/schema/tools.json — the 26 tools as JSON-schema (open_app, close_window, switch_app, minimize_window, click, drag, type_text, key_press, scroll, screenshot, list_dir, read_file, write_file, run_shell, browser_open, browser_search, browser_navigate, browser_close_tab, paste_to_webai, read_webai_response, set_alarm, set_volume, toggle_mute, open_url, notify_user, ask_user, finish)
- Wrote data/system_prompts/thursday_default.md — system prompt with variables {os}, {now}, {user_name}
- Wrote 6 hand-written golden JSONL examples: alarm_7am, find_youtube_video, summarize_screen, world_news, explain_script, volume_and_open — all validate
- Wrote scripts/generate_template_data.py — Method A generator with 10 templates (set_alarm, open_app, set_volume, browser_search_simple, open_url, delegate_to_webai, world_news, explain_file, ask_user_clarify, list_dir), cross-products OS×time×user_name×slot_values to scale. Tested: 50/50 examples valid.
- Wrote scripts/generate_llm_distilled.py — Method B generator using teacher LLM (Ollama/ZAI SDK/OpenAI-compatible) with Evol-Instruct mutation prompts, validates each output and retries on failure
- Wrote scripts/validate_dataset.py — checks JSON validity, schema correctness, tool-call id consistency, single-call-per-turn rule, JSON-string arguments convention. Tested: 100/100 generated + 6/6 hand-written all pass.
- Wrote scripts/merge_and_quantize.py — post-train: PEFT merge → save → llama.cpp convert_hf_to_gguf.py → llama-quantize Q4_K_M → optional HF Hub push
- Wrote training/thursday_ai_finetune.ipynb — 30-cell Kaggle notebook (validated JSON nbformat 4.5): config → install deps → HF login → clone repo → generate dataset if missing → load Qwen2.5-3B-Instruct in 4-bit via Unsloth → attach LoRA r=32 alpha=64 all-linear → format with Qwen2 chat template → train_on_responses_only → SFTTrainer 3 epochs effective batch 16 → save LoRA → merge → GGUF Q4_K_M → push to HF Hub → smoke test
- Wrote training/requirements.txt + training/README.md
- Wrote eval/thursday_eval_v1.json — 8-task seed benchmark across 7 categories
- Wrote eval/README.md + runtime/README.md (placeholders for M3/M4)
- Wrote README.md (10KB) — quick links, model decision summary, dataset spec, repo layout, roadmap
- Wrote .gitignore (excludes *.gguf, *.bin, data/processed/*.jsonl, etc.)
- Mirrored worklog into repo as WORKLOG.md
- Committed (M0: Foundation) + pushed to https://github.com/skyro777/Thursday_AI.git main (commit c481a9b)

Stage Summary:
- M0 (Foundation) is COMPLETE and pushed
- All artifacts saved to GitHub repo
- Ready for M1 (smoke-train): user can now upload training/thursday_ai_finetune.ipynb to Kaggle with T4x2 + HF_TOKEN secret and run it
- Next steps for user:
  1. Get HuggingFace token (write permission) → add as Kaggle Secret `HF_TOKEN`
  2. Upload training/thursday_ai_finetune.ipynb to Kaggle
  3. Set accelerator = GPU T4 x2
  4. Run all cells → first smoke-trained Thursday AI v0.1 will be on HF Hub
