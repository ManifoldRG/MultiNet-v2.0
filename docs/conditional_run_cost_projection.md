# Conditional-Eval Cost & Time Projection (Kimi K2.6 / Sonnet 4.6 / Opus 4.6 / Qwen 3.6)

Projection for the **6-set conditional sweep** = **10 unique prompt-variant configs × 15 mazes × 1 seed**
(`manifest.conditional_eval.json`), built from the Qwen3.5-27B validation-10 run data in
`docs/qwen35_hf_local_pipeline_notes.md` and `artifacts/qwen35_27b_hf_validation10/partial_results_summary/`.

> "A single run of the full 15" = **one variant run across the 15 mazes**.
> "All 10 experiments" = **the 10 deduplicated unique configs** (baseline + 9 distinct variants;
> the baseline is shared across all six condition sets and is paid once — per
> `docs/validation10_condition_sweep_rollout.md`).

---

## 1. Method & data sources

| Quantity | Source |
|---|---|
| Per-turn **text** input/output tokens (by obs format) | Qwen3.5 fresh rows, `fresh_rows.csv` (19 episodes, 1.75M tokens) |
| Per-call **latency** (by obs format) | Qwen3.5 `llm_latency_s` (666 min total recorded) |
| **Optimal path length** per maze | canonical BFS in `tasks/*/canonical_paths.json` (v01–v10 exact; 5 held-out estimated) |
| **Maze pixel size** | MiniGrid `TILE_PIXELS = 32` × maze (W,H); images sent full-res (no resize in `claude.py`/`kimi_k26.py`) |
| **Image → tokens** | Claude `(w·h)/750`; Kimi MoonViT ≈ `(w·h)/784` (14px patch + 2×2 merge) |
| **Pricing** | Opus 4.6 $5/$25; Sonnet 4.6 $3/$15; Kimi K2.6 $0.95(miss)/$0.16(hit) in, $4.00 out — all per 1M tok |

**Per-turn text-token profile measured from Qwen3.5 (images excluded from these counts):**

| Obs format | text in / call | out / call | latency / call |
|---|---:|---:|---:|
| `image_only` (baseline) | 386 | 173 | 13.3 s |
| `image_text` | 987 | 467 | 30.7 s |
| `text_only` | 942 | 500 | 33.0 s |

**Turns-per-episode (capable API model):** `step_by_step` ⇒ `turns ≈ 1.5 × optimal` (step caps don't bind).
Querying-strategy variants override this (`subgoal` ≈ 0.4×optimal calls; `full_trajectory` ≈ 3 calls total).

---

## 2. The image cost that was NOT in the token metrics  ⚠️

The Qwen metrics recorded **text tokens only** — proof: `image_only` logged *fewer* input tokens
(386/call) than `text_only` (942/call), because the rendered frame contributed **0** to the count.
Every turn ships one PNG; on the API models that PNG **is** billed as input tokens.

| Maze | W×H | Pixels | Img tok (Claude) | Img tok (Kimi) |
|---|---|---|---:|---:|
| v01 empty_room | 8×8 | 256×256 | 87 | 84 |
| v02 winding_corridor | 20×8 | 640×256 | 218 | 209 |
| v03 multi_path | 12×12 | 384×384 | 197 | 188 |
| v04–v08 (14×12) | 14×12 | 448×384 | 229 | 219 |
| v09–v10 (16×12) | 16×12 | 512×384 | 262 | 251 |
| S 14×14 | 14×14 | 448×448 | 268 | 256 |
| M / D1 / D3 10×10 | 10×10 | 320×320 | 137 | 131 |
| B blind 15×9 | 15×9 | 480×288 | 184 | 176 |
| **Average / maze** | | | **≈202** | **≈194** |

**At the baseline (`image_only`), image tokens are ≈35% of all input tokens.** Omitting them
understates input by roughly a third — and for the cheap-output Kimi model, input is the majority of spend.

---

## 3. Cost per experiment (one variant × 15 mazes × 1 seed)

Input = text + image tokens. Output assumes the terse one-line `FINAL_OUTPUT` contract
(Qwen ran `enable_thinking=False`). All USD.

| Experiment (unique config) | LLM calls | Input (M) | Output (M) | **Opus 4.6** | **Sonnet 4.6** | **Kimi K2.6** | Kimi (cache-hit) |
|---|---:|---:|---:|---:|---:|---:|---:|
| standard (baseline) | 624 | 0.373 | 0.108 | **4.57** | **2.74** | **0.78** | 0.49 |
| minimal (Prompt) | 624 | 0.325 | 0.108 | 4.32 | 2.59 | 0.74 | 0.48 |
| verbose (Prompt) | 624 | 0.458 | 0.119 | 5.26 | 3.15 | 0.90 | 0.55 |
| image_text (Obs format) | 624 | 0.748 | 0.291 | 11.03 | 6.62 | 1.87 | 1.28 |
| last3 (Context window) | 624 | 0.349 | 0.108 | 4.44 | 2.67 | 0.76 | 0.49 |
| text_summary (Context) | 624 | 0.349 | 0.108 | 4.44 | 2.67 | 0.76 | 0.49 |
| cardinal (Action space) | 624 | 0.373 | 0.108 | 4.57 | 2.74 | 0.78 | 0.49 |
| subgoal (Querying) | 166 | 0.099 | 0.086 | 2.65 | 1.59 | 0.44 | 0.36 |
| full_trajectory (Querying) | 45 | 0.026 | 0.019 | 0.61 | 0.36 | 0.10 | 0.08 |
| one_shot (In-context learning) | 624 | 0.904 | 0.108 | 7.22 | 4.33 | 1.28 | 0.58 |

Cost drivers worth noting: **`image_text` ≈ 2.4× baseline** (full text grid *plus* image every turn);
**`one_shot` ≈ 1.6×** (worked example in every prompt); **`full_trajectory` ≈ 0.13×** and
**`subgoal` ≈ 0.58×** (far fewer LLM calls).

---

## 4. TOTALS — all 10 experiments (15 mazes, 1 seed each)

| Model | 10-experiment total | Per the rollout (Kimi + one Claude tier) |
|---|---:|---|
| **Opus 4.6** | **$49.10** | Opus + Kimi = **$57.51** ($54.39 w/ Kimi cache) |
| **Sonnet 4.6** | **$29.46** | Sonnet + Kimi = **$37.87** ($34.75 w/ Kimi cache) |
| **Kimi K2.6** | **$8.41** (miss) / **$5.29** (cache-hit) | — |
| **Qwen 3.6 (local)** | **$0** (compute/time only) | — |

The doc's rollout runs **Claude + Kimi + local Qwen**. So a realistic full sweep is one of the
"Kimi + one Claude tier" lines above, **plus the Qwen wall-clock** below. Per seed.

---

## 5. Qwen 3.6 wall-clock (local, 6 parallel runners)

Qwen runs on local hardware (RTX 4090, 4-bit) — **no token cost**, only time. Latency basis = Qwen3.5
per-call latencies; **assumes 3.6 generates at a similar speed** (scale up if 3.6 is materially larger).

| Scenario | Sequential model-time | **6 runners (ideal)** | + ~15% orchestration |
|---|---:|---:|---:|
| 3.6 navigates near-optimally (1.5×opt calls) | ~24 h | **~4.0 h** | ~4.6 h |
| 3.6 truncates/loops like 3.5 (~35 min/episode measured) | ~88 h | **~14.6 h** | ~17 h |

**Plan for ½–1 day** of local Qwen time on 6 runners for the full 10×15 suite (1 seed).
The driver is truncation: Qwen3.5 hit step caps (140–220 calls/episode) on hard mazes; if 3.6
solves more it lands near the 4 h end.

---

## 6. Sensitivities (what could move these numbers)

- **Extended/adaptive thinking (biggest risk for Claude).** Output is ~55–60% of Opus/Sonnet spend.
  The estimate assumes terse one-line outputs. If adaptive thinking is left **on**, output tokens can
  rise 2–4× → Opus 10-exp total could move from ~$49 toward **$90–130**. Keep effort low / thinking
  constrained to hold the projection.
- **Prompt caching (biggest savings).** `step_by_step` turns share a growing prefix.
  Claude cache-read ≈0.1× input; Kimi cache-hit $0.16 vs $0.95 (~6×). Input is 35–77% of cost, so
  caching can cut totals materially (Kimi 10-exp already shown: $8.41 → $5.29). Not assumed on by
  default — the simple HTTP agents don't set `cache_control`/context caching yet.
- **Seeds.** All figures are **1 seed**. Costs and time scale linearly with seed count.
- **API-model truncation (upside risk).** If Claude/Kimi loop instead of solving (the Qwen failure
  mode), calls/episode rise toward the step cap and costs scale ~3–4×. Capable models shouldn't, but
  watch parse-failure / no-`FINAL_OUTPUT` loops.
- **Held-out maze optimals** (S/M/B/D, 5 of 15) are estimated; they're ~⅓ of mazes and small, so the
  effect on totals is minor.

---

## 7. Thinking-enabled projection

Assumption: **a 70%-thinking pipeline** — reasoning is ~70% of generated tokens, the answer ~30% —
so **output tokens × 3.33** (input unchanged). Applied uniformly to all variants.

### 7a. API cost with thinking (one variant × 15 mazes × 1 seed, USD)

| Experiment | out (M) | Opus 4.6 | Sonnet 4.6 | Kimi K2.6 | Kimi (cache) |
|---|---:|---:|---:|---:|---:|
| standard (baseline) | 0.360 | 10.86 | 6.52 | 1.79 | 1.50 |
| minimal (Prompt) | 0.360 | 10.62 | 6.37 | 1.74 | 1.49 |
| verbose (Prompt) | 0.396 | 12.18 | 7.31 | 2.01 | 1.66 |
| image_text (Obs) | 0.971 | 28.03 | 16.82 | 4.59 | 4.00 |
| last3 (Context) | 0.360 | 10.74 | 6.45 | 1.77 | 1.49 |
| text_summary (Context) | 0.360 | 10.74 | 6.45 | 1.77 | 1.49 |
| cardinal (Action) | 0.360 | 10.86 | 6.52 | 1.79 | 1.50 |
| subgoal (Querying) | 0.287 | 7.68 | 4.61 | 1.24 | 1.16 |
| full_trajectory (Querying) | 0.063 | 1.71 | 1.03 | 0.28 | 0.26 |
| one_shot (ICL) | 0.360 | 13.51 | 8.11 | 2.29 | 1.58 |

### 7b. Totals — all 10 experiments, thinking enabled (1 seed)

| Model | No-thinking | **Thinking (70%)** | Uplift |
|---|---:|---:|---:|
| **Opus 4.6** | $49.10 | **$116.94** | 2.38× |
| **Sonnet 4.6** | $29.46 | **$70.17** | 2.38× |
| **Kimi K2.6** | $8.41 | **$19.27** | 2.29× |
| **Kimi (cache-hit input)** | $5.29 | **$16.14** | 3.05× |

Uplift is below 3.33× because thinking scales **output only** (input is unchanged). The cheaper the
model's output relative to input — and the more input is cached — the closer the uplift gets to 3.33×.
Realistic dual sweep with thinking (Kimi + one Claude tier): **Sonnet+Kimi ≈ $89.4** ($86.3 w/ Kimi
cache); **Opus+Kimi ≈ $136.2** ($133.1 w/ Kimi cache). Per seed.

**Does the 4096-output cap change these?** No. Under the 70%-thinking model the **per-call** output is
577–2,293 tokens (max = `full_trajectory` on the largest maze) — all below 4096, so the cap never
binds and the totals above stand. The cap is a per-call *ceiling*: the absolute max if every one of the
~5,200 calls hit 4096 output would be ~$553 Opus / ~$332 Sonnet / ~$89 Kimi (≈9× the modeled output —
not a reachable regime here). The only way the cap raises cost is *indirectly*: if a call's reasoning
alone exceeds ~3,900 tokens it truncates before `FINAL_OUTPUT` → parse-failure retry → extra calls. If
that shows up on the hardest mazes, raise the API `max_tokens` to 8192 (currently 4096 for the paid
models per the M6 budget-parity invariant; only local Qwen is at 8192).

### 7c. Qwen 3.6 wall-clock with thinking (local, 6 runners)

Effective rate from Qwen3.5 = **14.6 output tok/s/runner** (583,219 output tok ÷ 666 min). Total
suite output (no-think) × 3.33 ÷ rate ÷ 6 runners:

| Turn regime | Suite output (no-think → thinking) | No-thinking | **Thinking** |
|---|---|---:|---:|
| Near-optimal (1.5×opt) | 1.16M → 3.88M tok | ~3.7 h | **~12.3 h** |
| Truncates/loops like Qwen3.5 | 4.60M → 15.35M tok | ~14.6 h | **~48.7 h** |

So with thinking on, **budget ~½ day if Qwen 3.6 navigates well, up to ~2 days if it truncates like
3.5.** Thinking roughly triples local generation time. (The 14.6 tok/s is *effective* — it bundles
prefill — so the thinking figure is mildly conservative for the added pure-decode reasoning tokens.)

---

## 8. Implementation changes shipped with this projection

- **Prompt caching wired for Claude** (`interface/agents/claude.py`): `cache_control` breakpoints on
  the system prompt and the last block of the latest turn (`enable_prompt_cache=True` by default), so
  each step reads the prior step's prefix. **Telemetry fixed** (`interface/telemetry.py`) to fold
  `cache_read_input_tokens` + `cache_creation_input_tokens` back into `input_tokens` — otherwise
  caching would silently undercount prompt tokens (Anthropic reports `input_tokens` as the *uncached
  remainder* only). Moonshot/Kimi caches identical prefixes automatically (no per-message field).
- **Qwen thinking enabled for the full conditional sweep**: the six
  `run_config.conditional_*_claude_kimi_qwen.json` set `enable_thinking: true` + `max_tokens: 8192`
  for the local Qwen model (mirrors the `ogbench_50` thinking precedent; the larger budget leaves room
  for chain-of-thought *plus* the trailing `FINAL_OUTPUT` line). API models stay at 4096; the M6
  budget-parity test was refined to enforce 4096 for the paid models and 8192 for local Qwen.
