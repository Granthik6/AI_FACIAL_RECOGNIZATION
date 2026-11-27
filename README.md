# AI Companion Glasses

A Windows webcam prototype for people who want spoken descriptions of a conversation partner’s visible facial movements. Press **Space** to hear a short description such as “The person is smiling and raising their eyebrows.” After installation, the camera, recognition, and speech run offline. By default, webcam frames stay in memory.

## Run the webcam demo

Use PowerShell in the folder containing this README. For a fresh checkout, install 64-bit Python 3.11 and run the setup once. The requirements reproduce the Windows research environment, including the CUDA 12.6 build of PyTorch. The webcam demo itself uses the CPU and the included MediaPipe model.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If the project environment is already set up, start here:

```powershell
.\.venv\Scripts\python.exe .\src\main.py --on-demand
```

Click the **camera preview** to give it keyboard focus. Press and release **Space** each time you want a description. Each press captures the current description, even when the expression stays the same. Requests made during playback are spoken in order. Windows playback uses the offline SAPI voice.

| Key | Action |
| --- | --- |
| **Space** | Speak the current description |
| **A** | Toggle automatic announcements of stable expression changes |
| **M** | Mute or unmute speech |
| **L** | Show or hide facial landmarks |
| **B** | Show or hide the strongest blendshape scores |
| **Q** | Close the preview and release the camera |

Use `--camera 1` to try a second webcam. Omit `--on-demand` to start in automatic mode. Run `.\.venv\Scripts\python.exe .\src\main.py --help` for camera resolution, speech rate, thresholds, and smoothing options.

## How a description is chosen

MediaPipe Face Landmarker processes each frame. When several faces are visible, the app favors a large, central face and uses its previous position to reduce switching. Named blendshape scores drive five movement labels: smiling, downturned mouth corners, raised eyebrows, furrowed eyebrows, and an open mouth. Several labels can be active together. The app also reports a missing face, an uncertain reading, or no distinct expression.

Temporal smoothing and a minimum persistence time reduce brief label changes. In a synthetic 30 Hz test, smoothing reduced transitions from 16 to 2 and delayed detection of the sustained movement by 0.4 seconds. The thresholds are heuristic. Recognition accuracy during real conversations, false announcements, and time to audible feedback still need validation with consenting participants.

## FER2013 research results

Two separately trained classifiers predict the seven conventional FER2013 categories. Both were evaluated on the same 7,178 held-out images. These results measure category classification. The live spoken descriptions use the blendshape pipeline above.

| Classifier | Test accuracy | Macro F1 |
| --- | ---: | ---: |
| Compact CNN | 50.85% | 0.409 |
| MobileNetV3-Small | 59.25% | 0.547 |

The repository includes trained checkpoints, float and INT8 ONNX exports, and laptop CPU benchmarks. To reproduce training, supply your own FER2013 copy under `datasets/FER2013/{train,test}/<class>/` and follow the run guide. The webcam demo works with the files included in the repository. Physical glasses, Raspberry Pi, and Android deployment remain untested.

- [Run guide, training commands, and deployment notes](research_papers/Run%20guide.md)
- [CNN evaluation](results/cnn_evaluation.json) and [MobileNet evaluation](results/mobilenet_evaluation.json)
- [Runtime comparison](results/fer_runtime_comparison.csv) and [research figures](figures/)
- [Research references](research_papers/References.md)

## Run the tests

The test suite covers movement classification, smoothing, speech queue behavior, repeated on-demand descriptions, and dataset splitting. It runs without a webcam.

```powershell
.\.venv\Scripts\python.exe -m pytest .\src\test_pipeline.py -q
```
