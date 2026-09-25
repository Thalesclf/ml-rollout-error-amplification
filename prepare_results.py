from pathlib import Path
import gzip
import shutil

root = Path(__file__).resolve().parent / "results"
archives = sorted(root.glob("seed_*/spectral_modes_campaign.csv.gz"))
if not archives:
    raise SystemExit("No compressed spectral campaign files were found.")

for src in archives:
    dst = src.with_suffix("")  # remove .gz
    if dst.exists():
        print(f"exists: {dst.relative_to(root.parent)}")
        continue
    print(f"extracting: {src.relative_to(root.parent)}")
    with gzip.open(src, "rb") as fi, open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo)

print("Results are ready for generate_all_figures.m")
