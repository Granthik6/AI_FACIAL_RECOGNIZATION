"""Batch-one CPU benchmark using the same ordered validation paths in every runtime."""
import argparse
import json
import time
import numpy as np
import pandas as pd
import psutil
import torch
import onnxruntime as ort
from fer_models import ROOT, RESULTS, network, Faces


def main(optimized=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--samples",type=int,default=100)
    parser.add_argument("--warmup",type=int,default=10)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--int8",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    suffix = "_smoke" if args.smoke else ""
    frame = pd.read_csv(RESULTS/"fer_manifest.csv")
    val = frame[frame.split=="val"].sample(frac=1,random_state=42).head(args.samples)
    results = []
    for kind in ("cnn","mobilenet"):
        data = Faces(val,kind)
        path = ROOT/f"models/fer2013_{kind}{suffix}_best.pth"
        if not path.exists():
            print("Pending",path.name)
            continue
        model = network(kind).eval()
        model.load_state_dict(torch.load(path,map_location="cpu"))
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        session = None
        if optimized:
            path = path.with_suffix(".int8.onnx" if args.int8 else ".onnx")
            if not path.exists():
                print("Pending",path.name)
                continue
            session = ort.InferenceSession(str(path),options,providers=["CPUExecutionProvider"])

        def infer(x):
            if session:
                return session.run(None,{"image":x.numpy()})[0]
            return model(x).numpy()

        with torch.inference_mode():
            x = data[0][0].unsqueeze(0)
            for _ in range(args.warmup):
                infer(x)
            timings, differences, disagree = [],[],0
            proc = psutil.Process()
            proc.cpu_percent()
            for i in range(len(data)):
                start = time.perf_counter_ns()
                x = data[i][0].unsqueeze(0)
                ready = time.perf_counter_ns()
                logits = infer(x)
                finish = time.perf_counter_ns()
                pred = logits.argmax()
                end = time.perf_counter_ns()
                timings.append([(ready-start)/1e6,(finish-ready)/1e6,(end-finish)/1e6,(end-start)/1e6])
            cpu, rss = proc.cpu_percent(),proc.memory_info().rss
            if optimized:
                for i in range(len(data)):
                    x = data[i][0].unsqueeze(0)
                    ref,actual = model(x).numpy(),infer(x)
                    differences.append(float(abs(ref-actual).max()))
                    disagree += ref.argmax()!=actual.argmax()
        values = np.array(timings)
        runtime = ("onnx_int8" if args.int8 else "onnx") if optimized else "torch"
        row = dict(model=kind,runtime=runtime,samples=len(data),
                   threads=args.threads,warmup=args.warmup,file_bytes=path.stat().st_size,
                   parameters=sum(p.numel() for p in model.parameters()),process_rss_bytes=rss,
                   process_cpu_percent=cpu,power_watts=None,gpu_usage=None,
                   fps=1000/values[:,-1].mean(),p95_ms=np.percentile(values[:,-1],95),
                   median_ms=np.median(values[:,-1]),max_logit_error=max(differences,default=None),
                   differing_predictions=int(disagree) if optimized else None)
        for col,label in enumerate(("preprocess_ms","inference_ms","postprocess_ms","total_ms")):
            row[label] = values[:,col].mean()
        results.append(row)
    if results:
        dest = RESULTS/("benchmark_"+results[0]["runtime"]+suffix)
        pd.DataFrame(results).to_csv(dest.with_suffix(".csv"),index=False)
        dest.with_suffix(".json").write_text(json.dumps(results,indent=2))
        print(pd.DataFrame(results)[["model","runtime","total_ms","fps"]])


if __name__=="__main__":
    main()
