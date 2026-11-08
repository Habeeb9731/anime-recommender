"""Stage 4: build results.md, the claims file, result charts and the README results block.

Every number here is read from reports/generated/*.json; nothing is typed by hand.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from animerec.charts import INK, _save, apply_style
from animerec.metrics import bootstrap_lift_ci
from animerec.utils import ROOT, get_logger, load_config, read_json, write_json

log = get_logger("report")

MODELS = [
    ("popularity", "Popularity (all-time)"),
    ("recent_popularity", "Recent popularity"),
    ("ease", "EASE"),
    ("als", "ALS"),
    ("content", "Content"),
    ("hybrid", "Hybrid"),
]
SEGMENTS = [
    ("all", "All users"),
    ("little_history", "Users with little history"),
    ("warm_titles", "Warm titles"),
    ("cold_titles", "Cold titles (full-catalog ranking)"),
    ("cold_titles_within_cold", "Cold titles (ranked among cold only)"),
]
SERIES = {"popularity": "#2a78d6", "recent_popularity": "#eb6834", "hybrid": "#1baf7a"}
LABEL = dict(MODELS)


def f4(x: Any) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def pct(x: Any, d: int = 1) -> str:
    return "n/a" if x is None else f"{x:.{d}f}%"


def signed(x: Any) -> str:
    return "n/a" if x is None else f"{x:+.1f}%"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def ci_text(b: dict[str, Any]) -> str:
    if not b.get("n_users") or b.get("lift_pct") is None:
        return "n/a"
    lo, hi = b["lift_pct_ci95"]
    return f"{signed(b['lift_pct'])} [{signed(lo)}, {signed(hi)}]"


def diff_text(b: dict[str, Any]) -> str:
    if not b.get("n_users"):
        return "n/a"
    lo, hi = b["diff_ci95"]
    return f"{b['diff_mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def charts(test: dict[str, Any], fig_dir: Path) -> None:
    apply_style()
    m = test["models"]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    names = [n for n, _ in MODELS][::-1]
    vals = [m[n]["all"]["ndcg@10"] for n in names]
    colors = ["#1baf7a" if n == "hybrid" else "#2a78d6" for n in names]
    bars = ax.barh([LABEL[n] for n in names], vals, color=colors, height=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f"  {v:.4f}", va="center", color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(vals) * 1.15)
    ax.set(title="Test NDCG@10 by model, all users", xlabel="NDCG@10")
    _save(fig, fig_dir / "results_models.png")

    segs = [s for s in SEGMENTS if s[0] in m["hybrid"]]
    y = np.arange(len(segs))[::-1]
    h = 0.24
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    for j, base in enumerate(["popularity", "recent_popularity", "hybrid"]):
        v = [m[base][s]["ndcg@10"] for s, _ in segs]
        ax.barh(y + (1 - j) * h, v, height=h * 0.9, color=SERIES[base], label=LABEL[base])
    ax.set_yticks(y, [n for _, n in segs])
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")
    ax.set(title="Test NDCG@10 by segment", xlabel="NDCG@10")
    _save(fig, fig_dir / "results_segments.png")

    rows = test["failure_analysis"]["by_history_length"]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    x = [r["history"] for r in rows]
    for base, marker in (("popularity", "o"), ("recent_popularity", "s"), ("hybrid", "D")):
        ax.plot(x, [r[f"{base}_ndcg@10"] for r in rows], color=SERIES[base], lw=2, marker=marker, ms=8, label=LABEL[base])
    ax.legend(frameon=False)
    ax.set(title="Test NDCG@10 by length of a user's history", xlabel="Titles in history", ylabel="NDCG@10")
    _save(fig, fig_dir / "results_history.png")


def build(cfg: dict[str, Any]) -> None:
    gen = cfg["paths"]["reports_dir"] / "generated"
    data, val, test, trace = (read_json(gen / n) for n in ("data_report.json", "validation_results.json", "test_results.json", "tuning_trace.json"))
    charts(test, cfg["paths"]["figures_dir"])
    m = test["models"]
    boot = test["bootstrap"]
    fr = test["frozen"]
    hyb = m["hybrid"]["all"]["ndcg@10"]
    pop = m["popularity"]["all"]["ndcg@10"]
    rec = m["recent_popularity"]["all"]["ndcg@10"]
    x = (hyb - pop) / pop * 100
    x_rec = (hyb - rec) / rec * 100
    bp, br = boot["all"]["popularity"], boot["all"]["recent_popularity"]
    beats_pop = bp["diff_ci95"][0] > 0
    beats_rec = br["diff_ci95"][0] > 0
    fa = test["failure_analysis"]
    ej = data["catalog_join"]
    fold = test["fold"]
    audit = data["rejected_source_timestamp_audit"]
    audit_pct = audit["share_equal_to_air_date_or_placeholder"] * 100
    decay = {str(r["value"]): r["val_ndcg@10"] for r in trace if r["stage"] == "decay_days"}
    decay_best = max(v for k, v in decay.items() if k != "None")
    decay_txt = f"On validation, recency-weighted EASE reached {decay_best:.4f} NDCG@10 versus {decay['None']:.4f} without recency weighting."

    per = ["ndcg@10", "recall@10", "precision@10"]
    all_rows = [
        [label, *[f4(m[n]["all"][k]) for k in per],
         pct(m[n]["beyond_accuracy"]["catalog_coverage"] * 100), f4(m[n]["beyond_accuracy"]["diversity"]),
         f"{m[n]['beyond_accuracy']['novelty_bits']:.2f}", pct(m[n]["beyond_accuracy"]["share_of_rec_slots_cold"] * 100, 2)]
        for n, label in MODELS
    ]
    all_table = table(["Model", "NDCG@10", "Recall@10", "Precision@10", "Coverage", "Diversity", "Novelty (bits)", "Cold share of slots"], all_rows)

    seg_blocks = []
    for s, label in SEGMENTS:
        if s not in m["hybrid"]:
            continue
        rows = [[lab, m[n][s]["n_users"], *[f4(m[n][s][k]) for k in per]] for n, lab in MODELS]
        seg_blocks.append(f"### {label}\n\n" + table(["Model", "Users", "NDCG@10", "Recall@10", "Precision@10"], rows))

    ci_rows = []
    for s, label in SEGMENTS:
        if s in boot:
            ci_rows.append([label, boot[s]["popularity"]["n_users"], diff_text(boot[s]["popularity"]), ci_text(boot[s]["popularity"]), diff_text(boot[s]["recent_popularity"]), ci_text(boot[s]["recent_popularity"])])
    ci_table = table(["Segment", "Users", "vs popularity: NDCG diff [95% CI]", "vs popularity: lift [95% CI]", "vs recent popularity: NDCG diff [95% CI]", "vs recent popularity: lift [95% CI]"], ci_rows)

    val_rows = [[lab, f4(val["models"][n]["all"]["ndcg@10"]), f4(val["models"][n].get("cold_titles", {}).get("ndcg@10"))] for n, lab in MODELS]
    filt_rows = [[r["filter"], f"{r['users_removed']:,}", f"{r['titles_removed']:,}", f"{r['interactions_removed']:,}", f"{r['users']:,}", f"{r['interactions']:,}"] for r in data["filters"]]
    sens_rows = [[str(r["label_definition"]), r["eval_users"], f4(r["popularity"]), f4(r["hybrid"]), signed(r["lift_pct"])] for r in val["label_threshold_sensitivity"]]
    hist_rows = [[r["history"], r["n_users"], f4(r["popularity_ndcg@10"]), f4(r["recent_popularity_ndcg@10"]), f4(r["hybrid_ndcg@10"])] for r in fa["by_history_length"]]
    miss_rows = [[r["title"], r["relevant_users"], r["recommended_to_those_users"], "yes" if r["cold"] else "no"] for r in fa["most_missed_titles"]]
    o1, o2 = fa["hybrid_vs_popularity_user_outcomes"], fa["hybrid_vs_recent_popularity_user_outcomes"]
    zh = test["zero_history_users"]

    verdict = (
        f"**The hybrid beats all-time popularity by {x:.1f}% on NDCG@10** "
        f"({f4(hyb)} vs {f4(pop)}; paired bootstrap 95% CI on the lift {signed(bp['lift_pct_ci95'][0])} to {signed(bp['lift_pct_ci95'][1])})."
        if beats_pop
        else f"**The hybrid did not reliably beat all-time popularity** (lift {x:.1f}%; 95% CI {signed(bp['lift_pct_ci95'][0])} to {signed(bp['lift_pct_ci95'][1])})."
    )
    verdict_rec = (
        f"Against the harder recent-popularity baseline ({f4(rec)}), the lift is {x_rec:.1f}% "
        f"(95% CI {signed(br['lift_pct_ci95'][0])} to {signed(br['lift_pct_ci95'][1])})"
        + (", which is also statistically clear." if beats_rec else ", which is not statistically distinguishable from zero.")
    )

    # Saved per-user scores give the EASE comparison without scoring the test set again.
    per_user = np.load(cfg["paths"]["artifacts_dir"] / "test_per_user_ndcg.npz")
    be = bootstrap_lift_ci(per_user["hybrid"], per_user["ease"], cfg["evaluation"]["bootstrap_samples"], cfg["seed"])
    write_json(gen / "hybrid_vs_ease.json", be)
    ease_clear = be["diff_ci95"][0] > 0 or be["diff_ci95"][1] < 0
    verdict_ease = (
        f"Against EASE alone ({f4(be['baseline_mean'])}), the hybrid scores {f4(be['model_mean'])}: a difference of {be['diff_mean']:+.4f} "
        f"(95% CI {be['diff_ci95'][0]:+.4f} to {be['diff_ci95'][1]:+.4f}), "
        + ("so EASE is statistically ahead overall. " if be["diff_mean"] < 0 and ease_clear else "so the two are effectively level overall. " if not ease_clear else "so the hybrid is statistically ahead overall. ")
        + "The hybrid's purpose is reaching titles EASE cannot score at all, not raising overall accuracy."
    )
    cold_all = m["hybrid"].get("cold_titles", {}).get("ndcg@10")
    cold_in = m["hybrid"].get("cold_titles_within_cold", {}).get("ndcg@10")
    cold_in_pop = m["popularity"].get("cold_titles_within_cold", {}).get("ndcg@10")
    cold_slots = m["hybrid"]["beyond_accuracy"]["share_of_rec_slots_cold"] * 100
    verdict_cold = (
        f"Cold titles are handled only partly. Ranked among cold titles alone, the hybrid reaches NDCG@10 {f4(cold_in)} versus {f4(cold_in_pop)} for popularity. "
        f"In full-catalog ranking they take just {cold_slots:.2f}% of recommendation slots and NDCG@10 on them is {f4(cold_all)}, so they rarely reach a user's top 10."
    )

    parity = val.get("demo_truncation_parity")
    parity_txt = (
        f"Truncating EASE to {parity['neighbors']} neighbors for the demo changes validation NDCG@10 from {f4(parity['full_ndcg@10'])} to {f4(parity['truncated_ndcg@10'])}."
        if parity else "The demo serves the same ALS factors as the evaluated model."
    )

    results = f"""# Results

Generated by `scripts/report.py` from `reports/generated/*.json`. Do not edit by hand.

## Headline

{verdict}

{verdict_rec}

{verdict_ease}

{verdict_cold}

X = (hybrid NDCG@10 - popularity NDCG@10) / popularity NDCG@10 x 100 = **{x:.1f}%**, using the all-time popularity baseline from the project brief. Recent popularity is the stronger baseline on this data, so read both numbers together.

- Test window: {cfg['split']['test_start']} to {cfg['split']['test_end']}; models were fit on all activity before {cfg['split']['test_start']}.
- Evaluated users: {fold['eval_users']:,}. Full eligible catalog: {fold['eligible_titles']:,} titles, including {fold['cold_eligible_titles']:,} cold titles. No negative sampling.
- Frozen configuration hash: `{test['config_hash']}`.

## Model comparison (test, all users)

{all_table}

Coverage is the share of the eligible catalog recommended to at least one user. Diversity is intra-list (1 minus mean pairwise content cosine) and novelty is mean self-information; both use the first 5,000 users.

![Model comparison](figures/results_models.png)

## Bootstrap confidence intervals over users

Paired resampling of users, {cfg['evaluation']['bootstrap_samples']:,} replicates.

{ci_table}

## Segment results (test)

Warm and cold rows score only that kind of title as relevant. The "full-catalog" cold row uses the same full ranking as every other row. The "ranked among cold only" row restricts candidates to cold titles, which isolates how well a model orders them.

{chr(10).join(seg_blocks)}

![Segment results](figures/results_segments.png)

Users with no history before the test window ({zh['n_users']:,}) are excluded from "All users" because no model can personalize for them. Their results: {('hybrid ' + f4(zh['hybrid']['ndcg@10']) + ', popularity ' + f4(zh['popularity']['ndcg@10']) + ', recent popularity ' + f4(zh['recent_popularity']['ndcg@10'])) if zh['n_users'] else 'none in this window'}.

## Validation results (used for all tuning)

{table(['Model', 'Val NDCG@10', 'Val cold-title NDCG@10'], val_rows)}

### Positive-threshold sensitivity (validation)

The models stay fixed; only the definition of a relevant title changes.

{table(['Label definition', 'Users', 'Popularity NDCG@10', 'Hybrid NDCG@10', 'Lift'], sens_rows)}

Selected configuration: {fr}.

{parity_txt}

## Failure analysis

- Compared with popularity, the hybrid is better for {pct(o1['better'] * 100)} of users, tied for {pct(o1['tie'] * 100)}, and worse for {pct(o1['worse'] * 100)}.
- Compared with recent popularity: better for {pct(o2['better'] * 100)}, tied for {pct(o2['tie'] * 100)}, worse for {pct(o2['worse'] * 100)}.
- Cold titles: {fa['cold_relevant_pairs']:,} relevant user-title pairs. Hit rate in the top 10 is {pct(None if fa['cold_hit_rate_with_synopsis'] is None else fa['cold_hit_rate_with_synopsis'] * 100, 2)} with a synopsis and {pct(None if fa['cold_hit_rate_without_synopsis'] is None else fa['cold_hit_rate_without_synopsis'] * 100, 2)} without one ({fa['cold_pairs_without_synopsis']:,} pairs).

### By length of user history

{table(['Titles in history', 'Users', 'Popularity', 'Recent popularity', 'Hybrid'], hist_rows)}

![By history length](figures/results_history.png)

### Most-missed relevant titles

Titles with the most relevant users who did not get them in their top 10.

{table(['Title', 'Relevant users', 'Recommended to those users', 'Cold'], miss_rows)}

## Data quality

- Rejected source: in the newer crawl, {pct(audit_pct)} of {audit['rows_sampled']:,} sampled `last_interaction_date` values equal a title's air date or the placeholder {audit['most_common_timestamp']}, so that column was not used for the split.
- Catalog titles: {ej['catalog_rows_raw']:,} raw, {ej['catalog_rows_final']:,} after removing adult titles and music or unknown types.
- Synopsis join by anime ID: {pct(ej['synopsis_id_match_rate'] * 100)} of titles matched; {pct(ej['synopsis_coverage_final_catalog'] * 100)} of the final catalog has a usable synopsis.
- Sparsity: {data['eda']['density_pct']:.2f}% of the sampled user-title matrix is filled. The top 10% of titles receive {pct(data['eda']['share_top_10pct_titles'] * 100)} of interactions.
- Sampled users after cleaning: {data['users_sampled']:,} of {data['users_after_cleaning']:,}; interactions kept: {data['interactions_sampled']:,}.

{table(['Filter', 'Users removed', 'Titles removed', 'Interactions removed', 'Users left', 'Interactions left'], filt_rows)}

![Long tail](figures/eda_long_tail.png)
![Status](figures/eda_status.png)
![Scores](figures/eda_scores.png)
![Over time](figures/eda_over_time.png)

Tuning trace: {len(trace)} validation runs, stored in `reports/generated/tuning_trace.json`.
"""
    (cfg["paths"]["reports_dir"] / "results.md").write_text(results)

    cold = m["hybrid"].get("cold_titles", {})
    cold_within = boot.get("cold_titles_within_cold", {}).get("popularity", {})
    bullet_recent = (
        f"beating a harder recent-popularity baseline by {x_rec:.1f}% (95% CI {signed(br['lift_pct_ci95'][0])} to {signed(br['lift_pct_ci95'][1])})"
        if beats_rec else f"with no reliable gain over a recent-popularity baseline ({signed(x_rec)}, 95% CI {signed(br['lift_pct_ci95'][0])} to {signed(br['lift_pct_ci95'][1])})"
    )
    claims = f"""# Resume bullets and interview answer

Generated by `scripts/report.py`. Every number comes from the files in `reports/generated/`.

## Resume bullets

- Built a hybrid anime recommender on {data['interactions_sampled']/1e6:.1f}M MyAnimeList interactions from {data['users_sampled']:,} users, using a global time-based split and full-catalog ranking; {'improved' if beats_pop else 'measured'} NDCG@10 by {x:.1f}% over a popularity baseline ({f4(pop)} to {f4(hyb)}; 95% CI on the lift {signed(bp['lift_pct_ci95'][0])} to {signed(bp['lift_pct_ci95'][1])}), {bullet_recent}.
- Designed a score blend that shifts from collaborative filtering to synopsis and metadata signals as a title's history falls to zero, matching EASE overall ({f4(be['model_mean'])} vs {f4(be['baseline_mean'])}) while ranking titles with no training history at NDCG@10 {f4(cold_in)} among themselves; shipped it as a Streamlit demo that folds in new users instantly with per-title explanations.

## 60-second interview explanation

I built a recommender for MyAnimeList that has to work for new visitors and for new shows. The first dataset I tried looked fine, but an audit showed that {audit_pct:.1f}% of its "interaction timestamps" in a {audit['rows_sampled']:,}-row sample were really air dates or placeholders, so a temporal split on it would have been fake. I switched to a dataset with real per-user update times and joined synopses by anime ID.

I evaluated with a global time split and ranked against the full catalog of {fold['eligible_titles']:,} titles, with all tuning on validation and the test set scored once. The hybrid blends collaborative and content scores per title, moving toward content as interaction history goes to zero, so a title with no history can still be ranked.

On the test set, NDCG@10 was {f4(hyb)} versus {f4(pop)} for all-time popularity, a {x:.1f}% lift. The more meaningful comparison is recent popularity at {f4(rec)}, where the lift is {x_rec:.1f}%, because anime viewing follows the airing season. {decay_txt} The hybrid did not beat plain EASE overall: {f4(be['model_mean'])} versus {f4(be['baseline_mean'])}. What it adds is the ability to score new titles. Among titles with no history it reached {f4(cold_in)}, across {cold_within.get('n_users', 0):,} users, but those titles took only {cold_slots:.2f}% of recommendation slots, so getting more of them into the top 10 without hurting accuracy is the main thing I would improve next.
"""
    (cfg["paths"]["reports_dir"] / "claims.md").write_text(claims)

    summary = f"""{verdict}

{verdict_rec}

{verdict_ease}

{verdict_cold}

{table(['Model', 'NDCG@10', 'Recall@10', 'Precision@10'], [[lab, *[f4(m[n]['all'][k]) for k in per]] for n, lab in MODELS])}

{table(['Segment', 'Users', 'Popularity', 'Recent popularity', 'Hybrid'], [[lab, m['hybrid'][s]['n_users'], f4(m['popularity'][s]['ndcg@10']), f4(m['recent_popularity'][s]['ndcg@10']), f4(m['hybrid'][s]['ndcg@10'])] for s, lab in SEGMENTS if s in m['hybrid']])}

Test window {cfg['split']['test_start']} to {cfg['split']['test_end']}, {fold['eval_users']:,} users, {fold['eligible_titles']:,} eligible titles ({fold['cold_eligible_titles']:,} cold). Full details are in [reports/results.md](reports/results.md)."""
    readme = ROOT / "README.md"
    if readme.exists():
        text = readme.read_text()
        text = re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->", lambda _: f"<!-- RESULTS:START -->\n{summary}\n<!-- RESULTS:END -->", text, flags=re.S)
        readme.write_text(text)
    log.info("Report written. X = %.2f%% (vs recent popularity %.2f%%)", x, x_rec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    build(load_config(parser.parse_args().config))
