# Reproducibility

## Primary campaign

The supplied launchers run five independently seeded experiments (`0,1,2,3,4`) with the
configuration used for the saved campaign results.

Windows:

```powershell
.\run_primary_campaign.ps1
```

Linux/macOS:

```bash
chmod +x run_primary_campaign.sh
./run_primary_campaign.sh
```

The Python program generates Lorenz–96 data internally; no external dataset is required.

## Reproduce the paper figures

The repository includes the saved campaign CSVs used by the MATLAB plotting script.
The largest tables (`spectral_modes_campaign.csv`) are stored as gzip files so that the
repository avoids individual ~60 MB text files.

Restore them with:

```bash
python prepare_results.py
```

Then open MATLAB in the repository root and run:

```matlab
generate_all_figures
```

The MATLAB script expects `results/seed_*/` and writes PDFs to `figures/`. It requires the
Statistics and Machine Learning Toolbox for Spearman correlation and percentile operations.

## Recorded environment

See `ENVIRONMENT.md`. The reported environment is Python 3.13.14 and PyTorch 2.14.0+cpu.

## Important archival note

The repository preserves the supplied experiment and figure-generation code rather than
silently rewriting the scientific workflow. If the manuscript figures are changed later,
tag the exact code/data state used for the public preprint (for example `arxiv-v1`).
