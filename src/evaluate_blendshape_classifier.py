import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prepare_fer2013 import FIGURES
from sklearn.metrics import classification_report, accuracy_score, multilabel_confusion_matrix
from train_blendshape_classifier import MODEL, dataset, targets, RESULTS
from movements import GROUPS, UNCERTAIN, Smoother, strengths, classify


def main():
    if not MODEL.exists():
        print("Pending trained movement classifier and held-out participants.")
        return
    saved = joblib.load(MODEL)
    data = dataset()
    data = data[data.participant.map(saved["splits"])=="test"].sort_values(["participant","session","ms"])
    y = targets(data)
    methods = {"supervised":saved["model"].predict(data[saved["features"]])}
    uncertain = {"supervised":0}
    stability = {}
    for method in ("fixed","smoothed"):
        preds,abstain,transitions,duration = [],0,0,0.
        for _,rows in data.groupby(["participant","session"],sort=False):
            smoother = Smoother()
            prev = None
            duration += (rows.ms.max()-rows.ms.min())/1000
            for _,row in rows.iterrows():
                scores = strengths(row)
                state = classify(scores) if method=="fixed" else smoother.update(scores,row.ms/1000)
                abstain += state==UNCERTAIN
                transitions += prev is not None and state!=prev
                prev = state
                preds.append([k in state for k in GROUPS])
        methods[method] = np.array(preds)
        uncertain[method] = abstain/len(data)
        stability[method] = transitions / duration if duration else None
    report = {}
    for name,pred in methods.items():
        report[name] = dict(report=classification_report(y,pred,target_names=list(GROUPS),output_dict=True,zero_division=0),
                            subset_accuracy=accuracy_score(y,pred),uncertain_rate=uncertain[name],
                            changes_per_second=stability.get(name),
                            confusion=multilabel_confusion_matrix(y,pred).tolist())
    (RESULTS/"movement_evaluation.json").write_text(json.dumps(report,indent=2))
    table = pd.DataFrame({name:{key:res["report"][key]["f1-score"] for key in GROUPS}
                          for name,res in report.items()})
    table.to_csv(RESULTS/"movement_f1.csv")
    ax = table.plot.bar(rot=15)
    ax.set(title="Verified movements on held-out participants",ylabel="F1 score",ylim=(0,1))
    plt.tight_layout()
    plt.savefig(FIGURES/"movement_comparison.png",dpi=220)
    plt.close()


if __name__=="__main__":
    main()
