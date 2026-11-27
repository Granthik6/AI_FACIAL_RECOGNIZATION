from pathlib import Path
from math import hypot
import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/face_landmarker.task"


def detector(video=True, faces=3):
    vision = mp.tasks.vision
    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
        num_faces=faces, output_face_blendshapes=True,
        min_face_detection_confidence=.5, min_face_presence_confidence=.5,
        min_tracking_confidence=.5)
    return vision.FaceLandmarker.create_from_options(options)


def image(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


def select(result, previous, size_weight=1., center_weight=.22, continuity=.35):
    choices = []
    for index, points in enumerate(result.face_landmarks):
        xs, ys = [p.x for p in points], [p.y for p in points]
        box = min(xs), min(ys), max(xs), max(ys)
        x, y, right, bottom = box
        center = (x+right)/2, (y+bottom)/2
        score = size_weight*((right-x)*(bottom-y))**.5
        score -= center_weight*hypot(center[0]-.5, center[1]-.5)
        if previous:
            score += continuity * max(0., 1 - 5*hypot(center[0]-previous[0], center[1]-previous[1]))
        choices.append((score, index, center, box))
    return max(choices) if choices else None


def visible(frame, box, points):
    h, w = frame.shape[:2]
    x, y, r, b = box
    if min(x, y) < -.015 or max(r, b) > 1.015 or min((r-x)*w,(b-y)*h) < 38:
        return False
    crop = frame[max(0,int(y*h)):int(b*h), max(0,int(x*w)):int(r*w)]
    if not crop.size or not 24 < crop.mean() < 240:
        return False
    # Reject pronounced yaw using the nose position between outer eye corners.
    if len(points) < 264:
        return False
    left, right = sorted((points[33].x, points[263].x))
    ratio = (points[1].x-left)/max(right-left, 1e-6)
    return .12 < ratio < .88
