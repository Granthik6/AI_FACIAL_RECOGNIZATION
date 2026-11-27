"""Deterministic engineering stress test with synthetic coefficients, separate from participant data."""
import pandas as pd
import matplotlib.pyplot as plt
from movements import DEFAULT, Smoother, classify, UNCERTAIN
from prepare_fer2013 import RESULTS, FIGURES


def main():
    rows = []
    smoother = Smoother()
    for frame in range(120):
        t = frame/30
        active = 1 <= t < 3
        strength = .66 if active else .1
        if frame in (17,24,47,58,78,102,109):
            strength = .18 if active else .72
        scores = dict.fromkeys(DEFAULT,0.)
        scores["smile"] = strength
        raw,stable = classify(scores),smoother.update(scores,t)
        rows.append(dict(seconds=t,programmed=int(active),coefficient=strength,
                         fixed=int("smile" in raw),smoothed=int("smile" in stable),
                         abstained=stable==UNCERTAIN))
    data = pd.DataFrame(rows)
    data.to_csv(RESULTS/"synthetic_temporal_trace.csv",index=False)
    summary = []
    for method in ("fixed","smoothed"):
        summary.append(dict(method=method,changes=int(data[method].diff().abs().sum()),
                            persistent_onset_delay_s=data.loc[(data.seconds>=1)&(data[method]==1),"seconds"].min()-1))
    pd.DataFrame(summary).to_csv(RESULTS/"synthetic_temporal_summary.csv",index=False)
    ax = data.plot(x="seconds",y=["programmed","fixed","smoothed"],drawstyle="steps-post",
                   subplots=True,figsize=(8,5),sharex=True)
    ax[0].set_title("Synthetic coefficient stress test at 30 Hz")
    ax[-1].set_xlabel("Time (s)")
    for panel in ax:
        panel.set_ylabel("Active")
    plt.tight_layout()
    plt.savefig(FIGURES/"synthetic_temporal.png",dpi=220)
    plt.close()
    print(summary)


if __name__=="__main__":
    main()
