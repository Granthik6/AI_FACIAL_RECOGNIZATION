import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from prepare_fer2013 import RESULTS, FIGURES, CLASSES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    suffix = "_smoke" if p.parse_args().smoke else ""
    tables = [pd.read_csv(path) for runtime in ("torch","onnx","onnx_int8")
              if (path:=RESULTS/f"benchmark_{runtime}{suffix}.csv").exists()]
    if tables:
        data = pd.concat(tables)
        data.to_csv(RESULTS/f"fer_runtime_comparison{suffix}.csv",index=False)
        for col,unit in (("total_ms","End-to-end latency (ms)"),("fps","Images per second"),
                         ("file_bytes","Model size (bytes)"),("process_rss_bytes","Process RSS (bytes)")):
            ax = data.pivot(index="model",columns="runtime",values=col).plot.bar(rot=0)
            ax.set(title="FER CPU benchmark"+suffix,ylabel=unit,xlabel="Architecture")
            plt.tight_layout()
            plt.savefig(FIGURES/f"{col}{suffix}.png",dpi=220)
            plt.close()
    scores = {}
    for kind in ("cnn","mobilenet"):
        path = RESULTS/f"{kind}{suffix}_evaluation.json"
        if path.exists():
            report = json.loads(path.read_text())
            scores[kind] = {k:report["classes"][k]["f1-score"] for k in CLASSES}
    if scores:
        ax = pd.DataFrame(scores).plot.bar(rot=0)
        ax.set(title="FER category evaluation"+suffix,ylabel="F1 score",ylim=(0,1))
        plt.tight_layout()
        plt.savefig(FIGURES/f"fer_f1{suffix}.png",dpi=220)
        plt.close()


if __name__=="__main__":
    main()
