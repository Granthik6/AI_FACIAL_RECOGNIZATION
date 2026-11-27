"""Read peak-frame AU codes and preserve source intensity coding."""
import pandas as pd
from prepare_fer2013 import ROOT,RESULTS


def main():
    rows = []
    for path in sorted((ROOT/"datasets/CK+").rglob("*_facs.txt")):
        subject,sequence,frame = path.stem.split("_")[:3]
        for line in path.read_text().splitlines():
            if line.strip():
                au,intensity = map(float,line.split())
                rows.append((subject,sequence,frame,au,intensity))
    if rows:
        pd.DataFrame(rows,columns=["participant","sequence","peak_frame","au","source_intensity"]).to_csv(RESULTS/"ckplus_au.csv",index=False)
    else:
        print("CK+ FACS annotations unavailable. Expected extracted *_facs.txt files.")


if __name__=="__main__":
    main()
