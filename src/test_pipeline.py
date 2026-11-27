import threading
from queue import Queue
import pandas as pd
from movements import GROUPS, DEFAULT, classify, strengths, Smoother, UNCERTAIN, NO_FACE, describe
from prepare_fer2013 import partition
from speech import Speech


def test_multilabel_and_missing():
    x = {name:0. for group in GROUPS.values() for name in group}
    assert classify(strengths(x)) == ()
    x.update(mouthSmileLeft=.8,jawOpen=.75)
    assert classify(strengths(x)) == ("smile","mouth_open")
    del x["jawOpen"]
    assert classify(strengths(x)) == UNCERTAIN
    assert describe(NO_FACE)=="No face detected."


def test_persistence_and_gap():
    s = Smoother()
    x = dict.fromkeys(DEFAULT,0.)
    x["smile"] = .8
    assert s.update(x,0)==UNCERTAIN
    assert s.update(x,.3)==("smile",)
    assert s.update(x,2)==UNCERTAIN


def test_speech_replaces_stale_requests():
    gate,spoken = threading.Event(),[]
    class Engine:
        def setProperty(self,*args): pass
        def stop(self): pass
        def runAndWait(self): pass
        def say(self,message):
            spoken.append(message)
            done.set()
    done = threading.Event()
    def factory():
        assert gate.wait(3)
        return Engine()
    s = Speech(factory=factory,cooldown=0)
    try:
        s.offer("old")
        s.offer("new")
        gate.set()
        assert done.wait(3)
        s.offer("new")
    finally:
        s.close()
    assert spoken==["new"]


def test_partition_preserves_test_and_separates_duplicates():
    rows = [(str(i),"happy","train") for i in range(30)]
    rows += [("0","happy","test"),("1","happy","train")]
    frame = partition(pd.DataFrame(rows,columns=["digest","label","split"]))
    assert list(frame[frame.digest=="0"].split)==["excluded_test_duplicate","test"]
    assert frame[frame.digest=="1"].split.nunique()==1
    assert set(frame.split)=={"train","val","test","excluded_test_duplicate"}


def test_each_space_request_plays_while_first_is_busy():
    started = Queue()
    release = threading.Event()

    class Engine:
        def setProperty(self, *args): pass
        def stop(self): pass
        def say(self, message): started.put(message)
        def runAndWait(self): assert release.wait(5)

    voice = Speech(factory=Engine, cooldown=60)
    try:
        voice.offer("Smile", manual=True)
        assert started.get(timeout=5) == "Smile"
        for text in ("Smile", "Smile", "Open mouth"):
            voice.offer(text, manual=True)
        voice.offer(None)
        release.set()
        assert [started.get(timeout=5) for _ in range(3)] == ["Smile", "Smile", "Open mouth"]
    finally:
        release.set()
        voice.close()
