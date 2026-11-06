# How Bitter Graphs

The tunable graph parameters live in `parameters.toml`. Change the scaling-law
numbers there, then regenerate the main outputs with:

```bash
cd experimental_analysis/how-bitter-graphs
python make_all_graphs.py
```

The default command rewrites:

- `figures/kaplan_to_chinchilla.png`
- `figures/algorithmic_progress_types.png`
- `figures/scale_dependent_stackplot.png`
- `figures/combined_scale_dependence_graphs.png`
- `figures/CEG_reference_combined.png`
- `figures/sensitivity_scale_dependent_share.png`
- `figures/sensitivity_scale_dependent_share.pdf`
- `numbers.txt`

For a parameter sweep or an alternate version, copy `parameters.toml`, edit the
copy, and run:

```bash
python make_all_graphs.py --config my-parameters.toml --output-dir figures/my-run
```

When `--output-dir` is set, `numbers.txt` is written into that directory too.

To regenerate only one or two outputs:

```bash
python make_all_graphs.py --only numbers kaplan
```
