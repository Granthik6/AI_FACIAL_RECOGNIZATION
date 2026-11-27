"""Named blendshape coefficients are heuristic strengths, not probabilities."""
import math

GROUPS = {
    "smile": ("mouthSmileLeft", "mouthSmileRight"),
    "corners_down": ("mouthFrownLeft", "mouthFrownRight"),
    "brows_up": ("browInnerUp", "browOuterUpLeft", "browOuterUpRight"),
    "brows_down": ("browDownLeft", "browDownRight"),
    "mouth_open": ("jawOpen",),
}
DEFAULT = dict(zip(GROUPS, (.45, .35, .40, .40, .45)))
UNCERTAIN = ("uncertain",)
NO_FACE = ("no_face",)
PHRASES = dict(zip(GROUPS, ("smiling", "turning their mouth corners down",
                            "raising their eyebrows", "furrowing their eyebrows",
                            "opening their mouth")))


def describe(state):
    if state == NO_FACE:
        return "No face detected."
    if state == UNCERTAIN:
        return "Facial expression could not be determined confidently."
    if not state:
        return "No distinct facial expression detected."
    return "The person is " + " and ".join(PHRASES[key] for key in state) + "."


def strengths(features):
    scores = {}
    for label, names in GROUPS.items():
        values = [features.get(name, math.nan) for name in names]
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
            return {}
        # A strong unilateral movement can pass on its own.
        scores[label] = max(sum(values) / len(values), .8 * max(values))
    return scores


def classify(scores, thresholds=DEFAULT, previous=(), hysteresis=0):
    if not scores:
        return UNCERTAIN
    active = tuple(k for k in GROUPS if scores[k] >= thresholds[k] -
                   (hysteresis if k in previous else 0))
    if not active and any(scores[k] > .7 * thresholds[k] for k in scores):
        return UNCERTAIN
    return active


class Smoother:
    """Time-based EMA followed by release hysteresis and persistence."""
    def __init__(self, thresholds=DEFAULT, tau=.12, persistence=.28, hysteresis=.08):
        self.thresholds = thresholds
        self.tau = tau
        self.persistence = persistence
        self.hysteresis = hysteresis
        self.reset()

    def reset(self):
        self.ema = {}
        self.last = None
        self.stable = UNCERTAIN
        self.candidate = None
        self.since = 0

    def update(self, scores, now):
        if self.last is not None and now - self.last > .6:
            self.reset()
        if not scores:
            self.reset()
            return UNCERTAIN
        alpha = 1 if self.last is None else -math.expm1(-(now-self.last)/self.tau)
        self.ema = {k: self.ema.get(k, v) + alpha*(v-self.ema.get(k, v))
                    for k, v in scores.items()}
        self.last = now
        state = classify(self.ema, self.thresholds, self.stable, self.hysteresis)
        if state != self.candidate:
            self.candidate, self.since = state, now
        if now - self.since >= self.persistence:
            self.stable = state
        return self.stable
