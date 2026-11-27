import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import distributions
import psutil
from vision import MODEL, ROOT, cv2, detector, image, np


def main():
    report = {"python":sys.version,"executable":sys.executable,"os":platform.platform(),
              "cpu":platform.processor(),"logical_cores":psutil.cpu_count(),
              "ram_bytes":psutil.virtual_memory().total,"model_bytes":MODEL.stat().st_size,
              "model_sha256":hashlib.sha256(MODEL.read_bytes()).hexdigest(),
              "packages":{p.metadata["Name"]:p.version for p in distributions()}}
    try:
        report["nvidia"] = subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv"],text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        report["nvidia"] = str(exc)
    try:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        report["torch_cuda"] = torch.version.cuda
    except ImportError as exc:
        report["torch"] = str(exc)
    with detector(video=False) as model:
        blank = model.detect(image(np.zeros((128,128,3),dtype=np.uint8)))
        report["blank_faces"] = len(blank.face_landmarks)
        import csv
        for path in sorted((ROOT/"datasets/FER2013/train/happy").iterdir())[:100]:
            sample = cv2.imread(str(path))
            if sample is None:
                continue
            sample = cv2.resize(sample,(192,192))
            start = time.perf_counter()
            result = model.detect(image(sample))
            elapsed = (time.perf_counter()-start)*1000
            if result.face_landmarks:
                features = {c.category_name:c.score for c in result.face_blendshapes[0]}
                report.update(sample=str(path.relative_to(ROOT)),landmarks=len(result.face_landmarks[0]),
                              blendshapes=len(features),categories=sorted(features))
                dest = ROOT/"results/blendshape_samples.csv"
                if not dest.exists():
                    with dest.open("w",newline="") as stream:
                        writer = csv.DictWriter(stream,fieldnames=["timestamp_ms","processing_ms",*sorted(features)])
                        writer.writeheader()
                        writer.writerow(dict(timestamp_ms=int(start*1000),processing_ms=elapsed,**features))
                break
    (ROOT/"results/environment_report.txt").write_text(json.dumps(report,indent=2),encoding="utf8")
    print({k:v for k,v in report.items() if k not in ("packages","categories")})


if __name__ == "__main__":
    main()
