"""Index frame-level AU intensity annotations in a legitimately supplied DISFA release."""
import re
import pandas as pd
from prepare_fer2013 import ROOT,RESULTS


def main():
    rows = []
    for path in sorted((ROOT/"datasets/DISFA").rglob("*.txt")):
        match = re.fullmatch(r"(SN\d+)_au(\d+)",path.stem,re.I)
        if match:
            for line in path.read_text().splitlines():
                frame,intensity = map(int,line.split(","))
                if not 0 <= intensity <=5:
                    raise ValueError(f"Invalid intensity in {path}")
                rows.append((*match.groups(),frame,intensity))
    if rows:
        pd.DataFrame(rows,columns=["participant","au","frame","intensity"]).to_csv(RESULTS/"disfa_au.csv",index=False)
    else:
        print("DISFA annotations unavailable. Expected SNxxx_auN.txt frame,intensity files.")


if __name__=="__main__":
    main()
