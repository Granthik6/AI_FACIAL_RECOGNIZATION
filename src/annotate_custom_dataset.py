"""Review a session against independent observation, optionally showing opted-in frames."""
import argparse
import pandas as pd
from collect_custom_dataset import labels
from vision import ROOT, cv2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("session",help="Anonymous participant/session path relative to datasets/custom")
    a = p.parse_args()
    base = (ROOT/"datasets/custom").resolve()
    path = (base/a.session/"features.csv").resolve()
    if not path.is_relative_to(base):
        p.error("Session must be inside datasets/custom.")
    data = pd.read_csv(path)
    try:
        for idx,row in data.iterrows():
            photo = path.parent/f"{row.ms}.jpg"
            if photo.exists():
                cv2.imshow("Annotation",cv2.imread(str(photo)))
                cv2.waitKey(1)
            answer = input(f"{row.ms} ms, prompt {row.labels}. Enter independently observed labels, skip or quit: ").strip()
            if answer=="quit":
                break
            if answer=="skip":
                continue
            data.loc[idx,["labels","verified"]] = [labels(answer),True]
            tmp = path.with_suffix(".tmp")
            data.to_csv(tmp,index=False)
            tmp.replace(path)
    finally:
        cv2.destroyAllWindows()


if __name__=="__main__":
    main()
