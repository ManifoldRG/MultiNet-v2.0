"""Assemble the self-contained analysis Artifact: inlines every figure in analysis/figures/
as a base64 data URI (the Artifact CSP blocks external assets) into a single HTML report.
Reproducible: re-run after analysis/plots.py to refresh. Writes analysis/figures/report.html.
"""
from __future__ import annotations
import base64
from pathlib import Path

FIG = Path("analysis/figures")
OUT = FIG / "report.html"


def _img(name: str) -> str:
    data = base64.b64encode((FIG / name).read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def figure(name, alt, caption):
    return (f'<figure class="plate">\n<img loading="lazy" src="{_img(name)}" alt="{alt}">\n'
            f'<figcaption>{caption}</figcaption>\n</figure>')


HEAD = """<title>Where language models get lost — MultiNet v2.0 conditional eval</title>
<style>
:root{
  --bg:#f4f6f8; --surface:#ffffff; --ink:#0f1419; --ink2:#47535f; --muted:#7d8894;
  --line:#e3e7eb; --line-strong:#cfd6dd; --accent:#2a78d6; --accent-soft:#eaf2fc;
  --claude:#2a78d6; --kimi:#d1571f; --qwen:#4a3aa7; --good:#0a8a0a; --crit:#c62f2f;
  --plate:#fcfcfb; --plate-line:rgba(15,20,25,.12);
  --sans:system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:ui-monospace,"SF Mono","Cascadia Code",Menlo,monospace;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0b0e11; --surface:#14191e; --ink:#eef2f6; --ink2:#a9b4bf; --muted:#6a7681;
  --line:#222a31; --line-strong:#2e3841; --accent:#4a91e8; --accent-soft:#16273b;
  --claude:#4a91e8; --kimi:#e2703a; --qwen:#9085e9; --good:#2fb02f; --crit:#e05b5b;
}}
:root[data-theme="dark"]{
  --bg:#0b0e11; --surface:#14191e; --ink:#eef2f6; --ink2:#a9b4bf; --muted:#6a7681;
  --line:#222a31; --line-strong:#2e3841; --accent:#4a91e8; --accent-soft:#16273b;
  --claude:#4a91e8; --kimi:#e2703a; --qwen:#9085e9; --good:#2fb02f; --crit:#e05b5b;
}
:root[data-theme="light"]{
  --bg:#f4f6f8; --surface:#ffffff; --ink:#0f1419; --ink2:#47535f; --muted:#7d8894;
  --line:#e3e7eb; --line-strong:#cfd6dd; --accent:#2a78d6; --accent-soft:#eaf2fc;
  --claude:#2a78d6; --kimi:#d1571f; --qwen:#4a3aa7; --good:#0a8a0a; --crit:#c62f2f;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  line-height:1.62;-webkit-font-smoothing:antialiased}
.wrap{max-width:1000px;margin:0 auto;padding:0 24px}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--muted)}
a{color:var(--accent)}
/* header */
header{border-bottom:1px solid var(--line);padding:64px 0 40px;
  background:
   linear-gradient(90deg,var(--line) 1px,transparent 1px) 0 0/28px 28px,
   linear-gradient(180deg,var(--line) 1px,transparent 1px) 0 0/28px 28px;
  background-blend-mode:normal;position:relative}
header .wrap{background:linear-gradient(var(--bg),transparent 60%)}
h1{font-size:clamp(30px,5vw,52px);line-height:1.05;letter-spacing:-.02em;font-weight:800;
  text-wrap:balance;margin:14px 0 12px}
h1 .em{color:var(--accent)}
.lede{font-size:19px;color:var(--ink2);max-width:64ch;margin:0 0 22px}
.meta{font-family:var(--mono);font-size:12.5px;color:var(--muted);display:flex;flex-wrap:wrap;
  gap:8px 18px;border-top:1px solid var(--line);padding-top:16px}
.meta b{color:var(--ink2);font-weight:600}
/* stat readout */
.readout{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;
  background:var(--line);border:1px solid var(--line);border-radius:12px;overflow:hidden;
  margin:34px 0 8px}
.stat{background:var(--surface);padding:18px 18px 16px}
.stat .n{font-size:30px;font-weight:800;letter-spacing:-.01em;font-variant-numeric:tabular-nums}
.stat .l{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--muted);margin-top:4px}
.stat.claude .n{color:var(--claude)} .stat.kimi .n{color:var(--kimi)} .stat.qwen .n{color:var(--qwen)}
/* sections */
section{padding:52px 0;border-bottom:1px solid var(--line)}
.snum{font-family:var(--mono);font-size:12px;color:var(--accent);letter-spacing:.1em}
h2{font-size:clamp(22px,3vw,30px);font-weight:800;letter-spacing:-.015em;margin:6px 0 10px;
  text-wrap:balance}
h3{font-size:16px;font-weight:700;margin:26px 0 6px;color:var(--ink)}
p{max-width:66ch;color:var(--ink2);margin:12px 0}
p.wide{max-width:none}
strong{color:var(--ink);font-weight:650}
.k{font-family:var(--mono);font-size:.9em;background:var(--accent-soft);color:var(--accent);
  padding:1px 6px;border-radius:5px}
.chip{font-family:var(--mono);font-size:.86em;padding:1px 7px;border-radius:20px;font-weight:600;
  border:1px solid currentColor}
.chip.claude{color:var(--claude)} .chip.kimi{color:var(--kimi)} .chip.qwen{color:var(--qwen)}
/* figure plates */
.plate{margin:26px 0 6px;background:var(--plate);border:1px solid var(--plate-line);
  border-radius:12px;padding:14px;box-shadow:0 1px 2px rgba(15,20,25,.04)}
.plate img{width:100%;height:auto;display:block;border-radius:6px}
figcaption{font-size:13.5px;color:var(--muted);margin-top:12px;padding-top:12px;
  border-top:1px solid var(--plate-line);max-width:none;font-family:var(--sans)}
figcaption b{color:var(--ink2)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:760px){.grid2{grid-template-columns:1fr}}
/* callout */
.callout{border-left:3px solid var(--accent);background:var(--accent-soft);border-radius:0 10px 10px 0;
  padding:16px 18px;margin:22px 0;font-size:15px}
.callout.warn{border-color:var(--crit);background:color-mix(in srgb,var(--crit) 8%,var(--surface))}
.callout .h{font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--muted);margin-bottom:4px}
/* table */
.tbl{width:100%;border-collapse:collapse;font-size:14px;margin:18px 0;font-variant-numeric:tabular-nums}
.tbl th,.tbl td{text-align:right;padding:8px 12px;border-bottom:1px solid var(--line)}
.tbl th:first-child,.tbl td:first-child{text-align:left;font-family:var(--mono);font-size:12.5px}
.tbl thead th{font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--muted);font-weight:600}
.tbl tr:hover td{background:var(--accent-soft)}
footer{padding:44px 0 80px;color:var(--muted);font-size:13.5px}
footer .wrap{max-width:74ch}
ul{color:var(--ink2);max-width:66ch} li{margin:5px 0}
</style>"""


def build():
    S = []
    S.append(HEAD)

    # ---------------- header ----------------
    S.append("""<header><div class="wrap">
<div class="eyebrow">MultiNet v2.0 · conditional-eval field report</div>
<h1>Where language models <span class="em">get lost</span></h1>
<p class="lede">Three frontier models solve gridworld mazes under ten prompting conditions.
The same picture keeps returning: <strong>path length decides everything</strong>, failures collapse
into a handful of spatial signatures, and the winning conditions are the ones that let a model
commit a plan instead of second-guessing every step.</p>
<div class="meta">
<span><b>540</b> clean episodes</span><span><b>3</b> models</span>
<span><b>15</b> mazes · <b>1</b> seed</span><span><b>10</b> conditions</span>
<span><b>41,318</b> queries mined</span><span><b>98.5M</b> tokens</span>
<span style="color:var(--crit)">directional — not powered for significance</span>
</div>
<div class="readout">
<div class="stat claude"><div class="n">58.9%</div><div class="l">Claude · solve</div></div>
<div class="stat kimi"><div class="n">37.2%</div><div class="l">Kimi · solve</div></div>
<div class="stat qwen"><div class="n">30.6%</div><div class="l">Qwen · solve</div></div>
<div class="stat"><div class="n">−0.84</div><div class="l">solve × path-len r</div></div>
<div class="stat"><div class="n">1.98M</div><div class="l">tok/solve · image-only</div></div>
</div>
</div></header>""")

    # ---------------- 1. the result ----------------
    S.append('<section><div class="wrap">')
    S.append('<div class="snum">01 / RESULT</div><h2>Claude leads; Kimi and Qwen are a statistical tie</h2>')
    S.append("""<p>Pooled across all ten conditions, <span class="chip claude">Claude 58.9%</span>
clears the field. The headline table below is a heatmap: the single darkest cell is Claude under
<span class="k">qry_subgoal</span> (93%), and the whole bottom row — <span class="k">obs_image_only</span> —
is the universal floor.</p>""")
    S.append(figure("fig01_solve_heatmap.png", "Solve rate heatmap, condition by model",
        "Solve rate (%) for every condition × model, sorted by pooled rate. The pooled column is set "
        "off at right. <b>qry_subgoal is a Claude-only win</b> (93 vs 20/20); image-only is the floor for all."))
    S.append("""<p>But at <span class="k">n = 15</span> mazes the model gap needs error bars.
Bootstrapping the per-episode solve indicator (n = 180 each): Claude beats both others significantly
(<strong>+21.6 pp</strong> over Kimi, +28.3 over Qwen), while <strong>Kimi vs Qwen is not significant</strong>
(+6.7 pp, 95% CI −16 to +3). The popular "Kimi &gt; Qwen" ordering is inside the noise.</p>""")
    S.append(figure("fig02_model_ci.png", "Model solve rate with bootstrap confidence intervals",
        "Overall solve rate with 95% bootstrap CIs. Claude separates cleanly; the Kimi–Qwen interval "
        "straddles zero."))
    S.append('</div></section>')

    # ---------------- 2. difficulty ----------------
    S.append('<section><div class="wrap">')
    S.append('<div class="snum">02 / DIFFICULTY</div><h2>It is the length of the path, not the machinery in it</h2>')
    S.append("""<p>Solve rate tracks the BFS-optimal path length almost perfectly
(<strong>r = −0.84</strong>) and is flat against mechanism count (r ≈ 0). A logistic model of
<span class="k">P(solve | model, config, difficulty)</span> lands a Brier score of
<strong>0.148</strong> with all three models folded in. Claude has the <em>shallowest</em>
difficulty slope and the highest floor — it degrades most gracefully as mazes get longer.</p>""")
    S.append('<div class="grid2">')
    S.append(figure("fig03_difficulty_curve.png", "P(solve) versus optimal path length per model",
        "Per-model logistic fits. Claude (blue) degrades slowest; Kimi falls fastest."))
    S.append(figure("fig04_mechanism_confound.png", "Maze solve rate vs path length colored by mechanism count",
        "Each dot is a maze. The two hardest (<b>winding</b>, <b>s5-corridor</b>, ~3% solved) are the "
        "<b>lightest</b> — zero mechanisms. Keys/switches cluster at moderate difficulty."))
    S.append('</div>')
    S.append("""<div class="callout"><div class="h">Caveat carried throughout</div>
Mechanism difficulty and path length are <strong>confounded by selection</strong>: the longest mazes
happen to be bare corridors. At n = 15 they cannot be separated — that is what the planned 200-maze
run is for. We do not claim mechanisms are free, only that this sample cannot price them.</div>""")
    S.append('</div></section>')

    # ---------------- 3. efficiency ----------------
    S.append('<section><div class="wrap">')
    S.append('<div class="snum">03 / COST</div><h2>What a single solved maze costs</h2>')
    S.append("""<p>Token spend is driven by <strong>decision count</strong>: step-by-step querying makes
one LLM call per environment step, so a messy 84-step wander costs double a clean 36-step solve of the
same maze. Priced per success, the conditions spread across <strong>two orders of magnitude</strong>.</p>""")
    S.append(figure("fig05_token_frontier.png", "Efficiency frontier: solve rate versus tokens per success",
        "Up-and-left is better. <b>qry_full_trajectory</b> is the cost champion (25k tok/solve) but "
        "brittle; <b>obs_image_only</b> is the money pit — <b>1.98M tokens per success at 11% solve</b>, "
        "78× the champion."))
    S.append("""<p>Per model, the same ranking as accuracy: Claude <strong>300k</strong> tokens/solve,
Kimi 478k, Qwen 631k. Efficiency and capability are the same axis here — the strong navigator is also
the cheap one, because it does not grind.</p>""")
    S.append('</div></section>')

    # ---------------- 4. how they fail ----------------
    S.append('<section><div class="wrap">')
    S.append('<div class="snum">04 / FAILURE</div><h2>Four ways to not reach the goal</h2>')
    S.append("""<p>Mining the 41,318 individual queries for position, blocked moves, and repeated actions,
every failed episode falls into one of four behavioral signatures. Clustering on
blocked-rate / revisit-rate / toggle-rate recovers them cleanly:</p>""")
    S.append("""<table class="tbl"><thead><tr><th>Signature</th><th>n</th><th>Tell</th>
<th>Where</th></tr></thead><tbody>
<tr><td>Wanderer</td><td>116</td><td style="text-align:left">circles, revisits 85% of cells</td>
<td style="text-align:left">cond_prompt·*, ctx_* · Kimi</td></tr>
<tr><td>Wall-banger</td><td>68</td><td style="text-align:left">82% of moves blocked, directional lock</td>
<td style="text-align:left">hist_multiturn, ctx_* · Qwen</td></tr>
<tr><td>Early parse-fail</td><td>38</td><td style="text-align:left">emits invalid actions, never starts</td>
<td style="text-align:left">act_cardinal · Qwen</td></tr>
<tr><td>Toggle-jammer</td><td>8</td><td style="text-align:left">fires TOGGLE at a gate 10–35×</td>
<td style="text-align:left">baseline_thinking, ctx_current</td></tr>
</tbody></table>""")
    S.append(figure("fig06_failure_clusters.png", "Failure episodes by blocked-rate and revisit-rate",
        "Every failed episode. The right-hand band is the wall-banger zone (Qwen-heavy); the high top-left "
        "is the wanderer zone (Kimi-heavy). Claude fails least and mostly by wall-banging."))
    S.append("""<h3>The failures have an address</h3>
<p>Because we recover the agent's coordinates at every step, failure has a <em>location</em>. On the
long corridors the entire episode-mass piles onto one or two cells far from the goal.</p>""")
    S.append(figure("fig07_spatial_heatmaps.png", "Dwell-density heatmaps for the three hardest mazes",
        "Visit density (log) per cell. On <b>s5-corridor</b> the mass never leaves the start (12,1) — "
        "16/33 episodes end there; on <b>winding</b> they ping-pong at the (18,3) dead-end, 3 cells short "
        "of the goal (19/33 die there). The goal star sits in empty space."))
    S.append(figure("fig08_blocked_by_config.png", "Median blocked-move rate by condition and model",
        "Wall-banging is condition-specific: worst under <b>hist_multiturn</b> and <b>ctx_current</b> for "
        "Kimi/Qwen; thinking mode drives Claude/Kimi to ~zero blocked moves."))
    S.append("""<div class="callout"><div class="h">Extends the prior read</div>
Toggle-perseveration was a single anecdote (one Claude episode firing TOGGLE 30× at a gate). It is
<strong>systematic</strong>: 13 episodes across all three models jam a gate ≥10×, concentrated in
<span class="k">obs_image_only</span> and <span class="k">baseline_thinking</span>. In image-only the
"gate won't toggle" feedback is rendered <em>in the image</em>, so the model never reads it.</div>""")
    S.append("""<h3>Distractors don't tempt — they hide the real key</h3>
<p>The natural story for the distractor mazes (<span class="k">v09</span>/<span class="k">v10</span>) is that
models get lured into grabbing the wrong-colour key. They don't: models almost never reach or dwell at a decoy
key. What actually happens is quieter — on these mazes, <strong>failed episodes simply never reach the
<em>correct</em> key (38–43% vs ~100% for solves)</strong>. The extra branches don't seduce; they make the
right key harder to find.</p>""")
    S.append("""<h3>Why they wall-bang: perception is fine, rotation is not</h3>
<p>Kimi and Qwen narrate their reasoning as inline prose, so their stated beliefs can be checked against
ground truth across <strong>12,419 traces</strong> (Claude's reasoning is suppressed/redacted — only 49
narrated queries). Three of four spatial faculties are intact — they read their own state almost perfectly
(≈100%), recall the wall map (95–98%), and take a single step without clipping a wall (92–94%). The one
that collapses is <strong>mental rotation: the facing after a planned turn is wrong ~46% of the time</strong>,
barely above chance. The deficit is spatial <em>transformation</em>, not <em>perception</em>.</p>""")
    S.append(figure("fig11_world_model.png", "Belief accuracy across four spatial faculties",
        "Belief accuracy vs ground truth, Kimi/Qwen. A clean staircase: perception, wall recall, and "
        "single-step translation are all 92–100%; <b>only mental rotation falls to 53–55%</b> — a coin flip. "
        "The rotation convention was validated at 100% against 11,260 real env transitions."))
    S.append("""<p>The error is <strong>structured</strong>, not random: a third of wrong rotations apply the
opposite turn (a literal LEFT/RIGHT swap), and the rate scales with how much a condition makes the model
plan turns in its head — from 18% under <span class="k">qry_full_trajectory</span> to <strong>61% under
<span class="k">baseline_thinking</span></strong>. More reasoning means more compounding rotation error.</p>""")
    S.append(figure("fig12_rotation_detail.png", "Rotation error rate by condition, and how the rotation is wrong",
        "Left: error rate climbs with in-head planning. Right: a third are LEFT/RIGHT inversions; most of the "
        "rest are the model claiming a turn didn't change its facing."))
    S.append("""<div class="callout"><div class="h">Honest caveat — correlation, not (yet) causation</div>
Tempting as it is, this does <strong>not</strong> cleanly explain the wall-banger cluster on its own. At the
episode level, rotation-error rate barely correlates with blocked-move rate (r ≈ +0.10) and doesn't move
solve rate (36% vs 34%). The likely reason: <strong>step-by-step querying re-grounds the true facing every
step</strong> (perception is ~100%), so a mis-planned rotation is usually corrected at the next observation.
The rotation gap is a real reasoning-fidelity weakness that bites hardest when a model must <em>commit</em> a
multi-step route — not a direct, per-step cause of ramming.</div>""")
    S.append("""<p>And you can't reason your way out of it: Qwen's per-query rotation-error rate <em>rises</em>
with reasoning length — <strong>32% on its shortest replies to 56% on its longest</strong> — because a longer
plan is a longer turn-chain with more places to compound. "Add more chain-of-thought" is not the fix for this
particular skill.</p>""")
    S.append('</div></section>')

    # ---------------- 5. method & integrity ----------------
    S.append('<section><div class="wrap">')
    S.append('<div class="snum">05 / METHOD &amp; INTEGRITY</div><h2>What the data is, and what it hides</h2>')
    S.append("""<p>The canonical set reconciles to exactly <strong>540 clean episodes</strong> (15 mazes ×
3 models × 10 conditions). Rather than discard the buggy runs, they were relabeled into a
<span class="k">bugged_*</span> family — kept for mining, excluded from every statistic.</p>""")
    S.append("""<h3>The worked example is an attractor for Claude — a dose-response</h3>
<p>The <span class="k">bugged_1</span> run duplicates the one-shot example every turn (3 copies in
context). Reading it alongside the ablation condition gives a clean monotonic curve: the more the example
is present, the worse Claude does — it drifts toward re-executing the example's action string instead of
the actual maze, grinding to the step cap (45→93 steps, 192k→537k tokens).</p>""")
    S.append("""<table class="tbl"><thead><tr><th>Example copies in context</th><th>Claude solve</th>
</tr></thead><tbody>
<tr><td>0 &nbsp;(icl_zero_shot, ablated)</td><td style="color:var(--good);font-weight:700">80%</td></tr>
<tr><td>1 &nbsp;(hist_multiturn, clean)</td><td>53%</td></tr>
<tr><td>3 &nbsp;(bugged_1, ICL-dup)</td><td style="color:var(--crit);font-weight:700">7%</td></tr>
</tbody></table>""")
    S.append("<h3>Two behaviors worth not misreading</h3>")
    S.append('<div class="grid2">')
    S.append(figure("fig09_parse_decay.png", "Parse-OK rate by step bucket within an episode",
        "Output-format validity does <b>not</b> decay with episode length — it is <b>lowest at steps 1–10</b> "
        "and stable after. Failures are format-fumbles at the start, not fatigue."))
    S.append(figure("fig10_subgoal_delegation.png", "Actions emitted per query under qry_subgoal, by model",
        "Under subgoal mode the <b>model emits the full primitive route itself</b> (Claude up to 90 actions/"
        "query); the harness does not pathfind. So Claude's 93% is a fair result — its own planning."))
    S.append('</div>')
    S.append("""<div class="callout warn"><div class="h">Data-integrity finding</div>
The BFS difficulty solver <strong>over-counts one maze</strong> (<span class="k">d3_dense_deadend</span>):
it reports optimal = 31, but two independent models reached the goal in <strong>23 legal steps</strong>,
so the true optimum is ≤23. It inflated that maze's difficulty and handed it a 93-step cap instead of ~69.
Every other maze is consistent (a separate benign +1/+2 solver drift aside).</div>""")
    S.append("""<div class="callout warn"><div class="h">Data-integrity finding · scoring artifact</div>
The <span class="k">qry_full_trajectory</span> parser matches the first <span class="k">Actions:</span>
anywhere in the text, so a model that writes "Actions: MOVE_FORWARD x 5" <em>in its reasoning</em> has that
line captured instead of its real plan — and <strong>28% of full-trajectory queries fail this way</strong>
(Qwen worst, the most verbose reasoner). Qwen's "failure" on the trivial <span class="k">empty_room</span>
is one of these: its plan was correct, the parser rejected it. So full-trajectory's low solve is partly a
scoring artifact, not model brittleness.</div>""")
    S.append("""<h3>One condition that is a real, model-specific failure</h3>
<p><span class="k">act_cardinal</span> replaces the ego-centric verbs (<span class="k">MOVE_FORWARD</span> +
turns) with absolute directions (<span class="k">MOVE_NORTH/SOUTH/EAST/WEST</span>). Claude and Kimi adapt
(parse-OK 0.95 / 0.85, 40% solve); <strong>Qwen does not</strong> — it keeps emitting
<span class="k">MOVE_FORWARD</span> from the default vocabulary, so half its outputs fail to parse
(0.46) and it solves <strong>0%</strong>. A clean instruction-following gap, not a navigation one.</p>""")
    S.append('</div></section>')

    # ---------------- 6. cost ----------------
    S.append('<section><div class="wrap">')
    S.append('<div class="snum">06 / COST</div><h2>Where the compute actually goes — and what is recoverable</h2>')
    S.append("""<p>Two levers dwarf everything else. The first is <strong>condition choice</strong>: at
matched accuracy, cost per success spans two orders of magnitude (§03) — running <span class="k">obs_image_only</span>
costs 1.98M tokens per solve, <span class="k">qry_full_trajectory</span> 25k. The second is <strong>wasted
grinding</strong>: in failed episodes, <strong>~46% of output tokens are spent after the agent visits its last
new cell</strong> — pure thrashing in place. Successful episodes waste under 1%, so the signal is clean.</p>""")
    S.append(figure("fig13_cost_waste.png", "Recoverable waste by model, and the early-abort tradeoff",
        "Left: share of failed-episode tokens spent thrashing after the last new cell — Qwen 54%, Claude 43%, "
        "Kimi 31% (~7.7M tokens). Right: aborting after a model sits frozen in one cell for K steps. "
        "<b>K=20 recovers 23% of all output tokens for a 2% success loss</b>; recovery from a deep freeze is rare."))
    S.append("""<p>A frozen-in-place watchdog is a safer trigger than a no-progress one (backtracking to fetch
a key looks like no-progress but is productive). At <span class="k">K=20</span> it reclaims about a quarter of
output tokens while sacrificing 5 of 223 successes — and it degrades gracefully, so the patience can be tuned
to how much a lost solve is worth. But the bigger win remains not launching the money-pit conditions at all.</p>""")
    S.append('</div></section>')

    # ---------------- footer ----------------
    S.append("""<footer><div class="wrap">
<div class="eyebrow" style="margin-bottom:10px">Standing caveats</div>
<ul>
<li><strong>n = 15 mazes, 1 seed</strong> — everything is directional; CIs are wide.</li>
<li>Path-length ⟂ mechanism confound is unresolved and needs the 200-maze run.</li>
<li>Qwen wall-clock is local-A100 inference and is not comparable to API latency.</li>
<li><span class="k">baseline_thinking</span> ran with <strong>unequal per-model token caps</strong>
(Qwen 4,096 / Claude 8,192 / Kimi 16,384 — a 4× spread). Claude and Kimi failures are blank thinking-mode
truncations; Qwen's are genuine grind-outs. Each cap is a fair within-model budget, but the cross-model
ranking is an artifact of budget, not reasoning — re-run with an equal ≥16,384 cap for a clean comparison.</li>
</ul>
<p style="color:var(--muted)">Generated from <span class="k">analysis/plots.py</span> over the cached
episode/query parquets · figures reproducible · palette validated for CVD.</p>
</div></footer>""")

    OUT.write_text("\n".join(S))
    print("wrote", OUT, f"({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    build()
