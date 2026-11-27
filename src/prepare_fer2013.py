"""Audit the local folder-format dataset and persist a deterministic split manifest."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"datasets/FER2013"
RESULTS = ROOT/"results"
FIGURES = ROOT/"figures"
CLASSES = ("angry","disgust","fear","happy","neutral","sad","surprise")


def partition(frame, seed=42):
    frame = frame.copy()
    train = frame[frame.split == "train"]
    # Keep identical decoded pixels together and keep test content out of development.
    test_hashes = set(frame.loc[frame.split == "test","digest"])
    overlap = train.digest.isin(test_hashes)
    frame.loc[train.index[overlap],"split"] = "excluded_test_duplicate"
    groups = train[~overlap].drop_duplicates("digest")
    _, val = train_test_split(groups, test_size=.15, random_state=seed, stratify=groups.label)
    frame.loc[(frame.split=="train") & frame.digest.isin(val.digest),"split"] = "val"
    return frame


def prepare():
    rows, errors = [], []
    for split in ("train","test"):
        for folder in sorted((DATA/split).iterdir()):
            if folder.name.lower() not in CLASSES:
                raise ValueError(f"Unexpected class {folder}")
            for path in sorted(folder.iterdir()):
                try:
                    with Image.open(path) as im:
                        pixels = np.asarray(im.convert("L"))
                        digest = hashlib.sha256(pixels.tobytes()).hexdigest()
                    rows.append((path.relative_to(DATA).as_posix(), folder.name.lower(),
                                 split,digest,*pixels.shape))
                except (ValueError,OSError) as exc:
                    errors.append((path.relative_to(DATA),str(exc)))
    frame = pd.DataFrame(rows,columns=["path","label","split","digest","height","width"])
    frame = partition(frame)
    frame.to_csv(RESULTS/"fer_manifest.csv",index=False)
    pd.DataFrame(errors,columns=["path","error"]).to_csv(RESULTS/"fer_errors.csv",index=False)
    table = pd.crosstab(frame.label,frame.split)
    table.to_csv(RESULTS/"fer_summary.csv")
    ax = table.plot.bar(figsize=(9,5),rot=0)
    ax.set(title="FER2013 decoded image counts",xlabel="Dataset label",ylabel="Images")
    plt.tight_layout()
    plt.savefig(FIGURES/"fer_distribution.png",dpi=220)
    plt.close()
    print(table, f"\nUnreadable files: {len(errors)}. Subject IDs unavailable.")


if __name__ == "__main__":
    prepare()
