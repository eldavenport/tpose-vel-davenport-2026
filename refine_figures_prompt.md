# Autonomous Figure Refinement Prompt

Paste the block below into Claude Code to start an autonomous refinement loop on a notebook.
Replace `<NOTEBOOK_PATH>` and `<OUTPUT_PNG_PATHS>` with the actual paths before running.

---

```
I want you to autonomously refine the figures in <NOTEBOOK_PATH> against the style specification
in figure_spec.md. Work in a loop without asking me for confirmation between steps.
Only stop if you hit an unrecoverable error, or after 10 iterations.

## Loop instructions

**Step 1 — Read the notebook before touching it.**
Read every code cell that generates a figure. List any non-default parameter values already set
(framealpha, rect, vmin/vmax, fontsize overrides, suptitle y-position, hspace/wspace values).
You must preserve these exactly — do not change them unless they directly violate figure_spec.md.

**Step 2 — Execute the notebook.**
Use:
  conda run -n tpose --no-capture-output jupyter nbconvert \
    --to notebook --execute <NOTEBOOK_PATH> \
    --output <NOTEBOOK_PATH>_executed.ipynb \
    --ExecutePreprocessor.timeout=600
Do NOT use --inplace. If execution fails, report the exact traceback and stop.

**Step 3 — Visual inspection.**
Use the Read tool to open each output PNG at these paths:
  <OUTPUT_PNG_PATHS>

For each figure, check it against figure_spec.md line by line:
- [ ] Font size ≥ 14 for axis labels; ≥ 11 for minor text
- [ ] Legend fully visible, not clipped
- [ ] Colorbar label includes units
- [ ] No panel title overlaps suptitle or neighboring panel
- [ ] All time series lines distinguishable (color + linestyle match the color/linestyle convention table)
- [ ] Boxplot mean markers (white diamond) visible
- [ ] dpi=150 and bbox_inches='tight' in savefig call
- [ ] hspace/wspace match the layout spec for this figure type
- [ ] Suptitle y-position does not collide with column titles

**Step 4 — Report deviations.**
List every failing check as: `[FIGURE] [CHECK] observed vs. expected`.
If no deviations: write "All specs satisfied — stopping." and stop.

**Step 5 — Fix one deviation at a time.**
Edit only the notebook cell responsible for the failing check.
Do NOT change: vmin/vmax, normalization factors, cost multipliers, depth clip values,
or any parameter not mentioned in the failing check.
After each cell edit, state what changed in one sentence.

**Step 6 — Go back to Step 2.**

## Scope constraint
Make ONLY changes that fix a spec violation. If you think an additional change would improve the
figure but it isn't in figure_spec.md, write it as a suggestion at the end — do not make it.
```

---

## Tips for filling in the paths

**Single notebook, figures saved to a known directory:**
```
<NOTEBOOK_PATH>  →  forecasts/UV_forecast.ipynb
<OUTPUT_PNG_PATHS>  →
  forecasts/results_figs/skill/vel_rmse_vs_tao_jan012013.png
  forecasts/results_figs/skill/vel_corr_vs_tao_jan012013.png
```

**If you don't know all output PNG paths ahead of time**, replace Step 3 with:
```
Use the Read tool to list all PNGs in the output directory modified in the last 5 minutes:
  find <OUTPUT_DIR> -name "*.png" -newer <NOTEBOOK_PATH> | sort
Then Read each one.
```

---

## Why this works

- `--output <nb>_executed.ipynb` (not `--inplace`) means your cell edits are never overwritten by nbconvert.
- Reading cell contents in Step 1 before executing prevents losing user-set style values.
- Checking one PNG at a time with the Read tool gives Claude genuine visual feedback, not just code inspection.
- The scope constraint prevents scope creep (the most common failure mode in iterative refinement).
