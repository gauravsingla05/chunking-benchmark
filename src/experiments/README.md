# Experiments

## List available methods

```bash
python3 research-paper/src/experiments/run_experiment.py --list-methods
```

## Pilot run (first 10 docs)

Put `.pdf`, `.txt`, or `.md` files in `research-paper/data/raw/`, then run:

```bash
python3 research-paper/src/experiments/run_experiment.py \
  --method truncation \
  --max-words 2000 \
  --limit-docs 10 \
  --save-outputs
```

Results are saved under `research-paper/results/runs/<timestamp>/`.

Output filenames in `outputs/` are now `<method>__<doc_id>.txt` for quick scanning.
