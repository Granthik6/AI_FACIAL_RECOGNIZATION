"""Run the wearable prototype or one of its staged demonstrations."""
import argparse
import csv
from contextlib import ExitStack
from math import dist
from time import perf_counter as monotonic

from movements import DEFAULT, NO_FACE, UNCERTAIN, Smoother, classify, describe, strengths
from speech import Speech
from vision import ROOT, cv2, detector, image, select, visible


def run(stage=8):
    p = argparse.ArgumentParser(description=__doc__)
    for name, default in {"camera":0,"width":640,"height":480,"faces":3,"frames":0,
                          "rate":175}.items():
        p.add_argument("--"+name, type=int, default=default)
    for name, default in {"tau":.12,"persistence":.28,"hysteresis":.08,"volume":1.,
                          "cooldown":2.,"size-weight":1.,"center-weight":.22,
                          "continuity":.35}.items():
        p.add_argument("--"+name, type=float, default=default)
    for key, value in DEFAULT.items():
        p.add_argument("--"+key, type=float, default=value)
    for flag in ("mute","on-demand","headless","log","features"):
        p.add_argument("--"+flag, action="store_true")
    args = p.parse_args()
    limits = {k:getattr(args,k) for k in DEFAULT}
    if args.tau <= 0 or args.persistence < 0 or not all(0 < t <= 1 for t in limits.values()):
        p.error("Choose positive thresholds up to 1, positive tau and nonnegative persistence.")
    smooth = Smoother(limits,args.tau,args.persistence,args.hysteresis)
    with ExitStack() as stack:
        model = stack.enter_context(detector(faces=args.faces)) if stage >= 2 else None
        camera = cv2.VideoCapture(args.camera)
        stack.callback(camera.release)
        if not camera.isOpened():
            raise SystemExit(f"Camera {args.camera} unavailable. Check Windows camera access and --camera.")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH,args.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT,args.height)
        if not args.headless:
            stack.callback(cv2.destroyAllWindows)
        voice = None
        muted, auto = args.mute, not args.on_demand
        dots, bars = stage == 3, stage == 4
        last_center, seen, stamp, count = None, 0., -1, 0
        last_frame = monotonic()
        offered = None
        feature_writer = None
        perf = None
        if args.log:
            perf = csv.writer(stack.enter_context((ROOT/"results/performance.csv").open("a",newline="")))
        while True:
            ok, frame = camera.read()
            if not ok:
                raise SystemExit("Camera opened but returned no frame.")
            now = monotonic()
            fps = 1/max(now-last_frame,1e-6)
            last_frame = now
            stamp = max(stamp+1,int(now*1000))
            state, features, scores, faces = NO_FACE, {}, {}, 0
            if model:
                result = model.detect_for_video(image(frame),stamp)
                faces = len(result.face_landmarks)
                if now-seen > .6:
                    last_center = None
                target = select(result,last_center,args.size_weight,args.center_weight,args.continuity)
                if target:
                    _, idx, center, box = target
                    if last_center is None or dist(last_center,center) > .2:
                        smooth.reset()
                    last_center, seen = center, now
                    points = result.face_landmarks[idx]
                    features = {c.category_name:c.score for c in result.face_blendshapes[idx]}
                    scores = strengths(features) if visible(frame,box,points) else {}
                    state = smooth.update(scores,now) if stage >= 7 else classify(scores,limits)
                    h,w = frame.shape[:2]
                    x,y,r,b = box
                    cv2.rectangle(frame,(int(x*w),int(y*h)),(int(r*w),int(b*h)),(0,225,180),2)
                    if dots:
                        for point in points:
                            cv2.circle(frame,(int(point.x*w),int(point.y*h)),1,(255,180,40),-1)
                elif now-seen > .6:
                    smooth.reset()
            latency = (monotonic()-now)*1000
            if features and (stage == 4 or args.features):
                if feature_writer is None:
                    path = ROOT/"results/blendshape_samples.csv"
                    fresh = not path.exists() or not path.stat().st_size
                    stream = stack.enter_context(path.open("a",newline=""))
                    feature_writer = csv.DictWriter(stream,fieldnames=["timestamp_ms","processing_ms",*sorted(features)])
                    if fresh:
                        feature_writer.writeheader()
                feature_writer.writerow(dict(timestamp_ms=stamp,processing_ms=latency,**features))
            sentence = describe(state)
            if perf:
                perf.writerow((stamp,faces,fps,latency,sentence))
            if stage >= 6 and not muted:
                if voice is None:
                    voice = Speech(args.rate,args.volume,args.cooldown)
                    stack.callback(voice.close)
                if voice.error:
                    print("Speech unavailable:",voice.error)
                    muted = True
                elif auto and sentence != offered:
                    offered = sentence
                    voice.offer(sentence if state not in (NO_FACE,UNCERTAIN) and state else None)
            lines = [f"{fps:.1f} FPS   {latency:.1f} ms   Faces {faces}"]
            if stage >= 5:
                lines.append(sentence)
                if scores:
                    key_score = max(scores,key=scores.get)
                    lines.append(f"Heuristic score: {key_score} {scores[key_score]:.2f}")
            if bars:
                lines.extend(f"{k} {v:.2f}" for k,v in sorted(features.items(),key=lambda item:-item[1])[:8])
            key = -1
            if not args.headless:
                for row,line in enumerate(lines):
                    cv2.putText(frame,line,(8,24+23*row),cv2.FONT_HERSHEY_SIMPLEX,.52,(80,255,180),1,cv2.LINE_AA)
                cv2.imshow("Companion glasses",frame)
                key = cv2.waitKey(1) & 255
            count += 1
            if key == ord("q") or (args.frames and count >= args.frames):
                print(f"Processed {count} camera frames. Last frame: {faces} faces, {latency:.1f} ms.")
                break
            if key == ord("l"):
                dots = not dots
            if key == ord("b"):
                bars = not bars
            if key == ord("m"):
                muted = not muted
            if key == ord("a"):
                auto = not auto
            if voice and key == ord("m") and muted:
                voice.clear()
                offered = None
            if voice and key == ord("a"):
                voice.offer(None)
                offered = None
            if key == 32 and voice and not muted:
                voice.offer(sentence,manual=True)


def launch(stage=8):
    try:
        run(stage)
    except KeyboardInterrupt:
        print("Camera stopped.")


if __name__ == "__main__":
    launch()
