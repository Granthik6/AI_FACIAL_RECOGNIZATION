"""Select a multilabel estimator using participant-disjoint validation data."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from movements import GROUPS
from prepare_fer2013 import ROOT, RESULTS

MODEL = ROOT/"models/blendshape_classifier.joblib"


def dataset():
    files = sorted((ROOT/"datasets/custom").glob("*/*/features.csv"))
    frames = [pd.read_csv(f) for f in files if f.stat().st_size]
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames,ignore_index=True)
    return frame[frame.verified & frame.usable & (frame.labels!="uncertain")].copy()


def targets(rows):
    return np.array([[key in s.split(",") for key in GROUPS] for s in rows.labels],dtype=int)


def main():
    data = dataset()
    if data.empty or data.participant.nunique()<5:
        print("Pending verified movement labels from at least five distinct participants.")
        return
    names = sorted(data.participant.unique())
    dev,test = train_test_split(names,test_size=.2,random_state=42)
    train,val = train_test_split(dev,test_size=.25,random_state=42)
    splits = {p:k for k,ids in (("train",train),("val",val),("test",test)) for p in ids}
    columns = sorted(set(data)-{"participant","session","ms","verified","usable","labels"})
    a,b = data[data.participant.isin(train)],data[data.participant.isin(val)]
    if any(len(np.unique(col))<2 for col in targets(a).T):
        print("Pending positive and negative training examples for every movement.")
        return
    scores, best = {},-1.
    candidates = dict(logistic=LogisticRegression(max_iter=1000),
                      forest=RandomForestClassifier(n_estimators=100,random_state=42),
                      svm=SVC(),mlp=MLPClassifier(hidden_layer_sizes=(32,),max_iter=500,random_state=42))
    for name,base in candidates.items():
        model = OneVsRestClassifier(make_pipeline(StandardScaler(),base))
        model.fit(a[columns],targets(a))
        score = f1_score(targets(b),model.predict(b[columns]),average="macro",zero_division=0)
        scores[name] = score
        if score>best:
            best = score
            joblib.dump(dict(model=model,features=columns,splits=splits,name=name),MODEL)
    (RESULTS/"movement_validation.json").write_text(json.dumps(scores,indent=2))


if __name__=="__main__":
    main()
