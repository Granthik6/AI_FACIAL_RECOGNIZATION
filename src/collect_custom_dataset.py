"""Capture one consented participant per session. Features are the default."""
import argparse
import csv
import json
import re
import time
import uuid
from movements import GROUPS
from vision import ROOT, detector, cv2, image, visible, select


def labels(text):
    items = text.split(",")
    if any(i not in (*GROUPS,"neutral","uncertain") for i in items):
        raise ValueError("Use comma-separated movement keys, neutral or uncertain.")
    if len(items)>1 and ("neutral" in items or "uncertain" in items):
        raise ValueError("Neutral and uncertain must be used alone.")
    return ",".join(sorted(set(items)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--participant",required=True,help="Reuse the same anonymous code for every session.")
    p.add_argument("--labels",type=labels,required=True)
    p.add_argument("--lighting",required=True)
    p.add_argument("--angle",required=True)
    p.add_argument("--consent",action="store_true")
    p.add_argument("--save-images",action="store_true",help="Requires specific raw-image consent.")
    p.add_argument("--seconds",type=float,default=5)
    p.add_argument("--hz",type=float,default=5)
    p.add_argument("--camera",type=int,default=0)
    a = p.parse_args()
    if not a.consent or not re.fullmatch(r"P[0-9]{3,8}",a.participant) or min(a.seconds,a.hz)<=0:
        p.error("Confirm consent, supply an anonymous code such as P001 and positive timing.")
    session = uuid.uuid4().hex
    folder = ROOT/"datasets/custom"/a.participant/session
    with detector(faces=2) as model:
        cam = cv2.VideoCapture(a.camera)
        try:
            if not cam.isOpened():
                raise SystemExit("Camera unavailable.")
            folder.mkdir(parents=True)
            (folder/"session.json").write_text(json.dumps(vars(a),indent=2))
            with (folder/"features.csv").open("x",newline="") as stream:
                writer = None
                start = time.perf_counter()
                while time.perf_counter()-start < a.seconds:
                    ok,frame = cam.read()
                    if not ok:
                        raise RuntimeError("Camera read failed.")
                    t = time.perf_counter()
                    ms = int((t-start)*1000)
                    result = model.detect_for_video(image(frame),ms)
                    if len(result.face_landmarks)==1:
                        _,idx,_,box = select(result,None)
                        row = dict(participant=a.participant,session=session,ms=ms,
                                   labels=a.labels,verified=False,usable=visible(frame,box,result.face_landmarks[idx]))
                        row.update({c.category_name:c.score for c in result.face_blendshapes[idx]})
                        if writer is None:
                            writer = csv.DictWriter(stream,fieldnames=list(row))
                            writer.writeheader()
                        if a.save_images and not cv2.imwrite(str(folder/f"{ms}.jpg"),frame):
                            raise OSError("Could not save opted-in frame.")
                        writer.writerow(row)
                    time.sleep(max(0,1/a.hz-(time.perf_counter()-t)))
        finally:
            cam.release()
    print(folder)


if __name__=="__main__":
    main()
