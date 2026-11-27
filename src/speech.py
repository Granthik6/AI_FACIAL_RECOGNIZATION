"""Queue each key press and keep only the newest automatic announcement."""
from collections import deque
from contextlib import contextmanager
import math
import sys
import threading
import time


@contextmanager
def speaker(rate, volume, factory):
    if sys.platform == "win32" and factory is None:
        import pythoncom
        from win32com.client import Dispatch
        pythoncom.CoInitialize()
        try:
            voice = Dispatch("SAPI.SpVoice")
            voice.Volume = max(0, min(100, round(volume * 100)))
            voice.Rate = max(-10, min(10, round(math.log(max(1, rate) / 175, 1.2))))
            # Synchronous playback blocks only the speech worker.
            yield lambda message: voice.Speak(message, 0)
        finally:
            voice = None
            pythoncom.CoUninitialize()
        return

    import pyttsx3
    engine = (factory or pyttsx3.init)()
    try:
        engine.setProperty("rate", rate)
        engine.setProperty("volume", volume)

        def speak(message):
            engine.say(message)
            engine.runAndWait()

        yield speak
    finally:
        engine.stop()


class Speech:
    def __init__(self, rate=175, volume=1., cooldown=2., factory=None):
        self.condition = threading.Condition()
        self.pending = None
        self.requests = deque()
        self.previous_offer = None
        self.closed = False
        self.error = None
        self.rate, self.volume, self.cooldown = rate, volume, cooldown
        self.factory = factory
        self.worker = threading.Thread(target=self.run, daemon=True)
        self.worker.start()

    def offer(self, message, manual=False):
        with self.condition:
            if manual and message:
                self.requests.append(message)
                self.pending = None
                self.condition.notify()
                return
            if message and message == self.previous_offer and not manual:
                return
            self.previous_offer = message
            self.pending = (message, time.monotonic()) if message else None
            self.condition.notify()

    def clear(self):
        with self.condition:
            self.requests.clear()
            self.pending = self.previous_offer = None
            self.condition.notify()

    def run(self):
        try:
            available = 0
            with speaker(self.rate, self.volume, self.factory) as speak:
                while True:
                    with self.condition:
                        self.condition.wait_for(lambda: self.closed or self.requests or self.pending)
                        if self.closed:
                            return
                        if self.requests:
                            message = self.requests.popleft()
                        else:
                            message, created = self.pending
                            now = time.monotonic()
                            if now < available:
                                self.condition.wait(available-now)
                                continue
                            self.pending = None
                            if now-created > max(2., self.cooldown + .5):
                                continue
                    speak(message)
                    available = time.monotonic()+self.cooldown
        except Exception as exc:
            self.error = str(exc)

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify()
        self.worker.join(8)
        if self.worker.is_alive():
            raise RuntimeError("Speech driver did not shut down in eight seconds.")
