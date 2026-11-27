# Running the companion glasses

Use PowerShell from the existing AI_Companion_Glasses directory. Assign the project interpreter once per terminal.

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py src\check_environment.py
& $py v1_camera.py
& $py src\v2_face_detection.py
& $py src\v3_facial_landmarks.py
& $py src\v4_blendshape_extraction.py
& $py src\v5_expression_recognition.py
& $py src\v6_audio_feedback.py
& $py src\v7_temporal_smoothing.py
& $py src\main.py
```

Use Q to quit, M to mute, Space to describe, A to toggle automatic speech, L for landmarks and B for coefficients. Start with `--on-demand` for quieter conversations. Use `--camera 1` for a second webcam. Each entry accepts `--help`. Stage 4 appends all named coefficients to results/blendshape_samples.csv. The first environment probe can seed this CSV with its documented FER training sample. The real-time pipeline keeps images in memory. The heuristics can abstain on missing features, low light, extreme head turn, or marginal activation. Speech, occlusion and blinking can still confuse the system. Tune thresholds only on development participants.

## FER2013 experiments

The manifest holds original relative paths, decoded image hashes, category and assigned partition. Exact decoded duplicates of test content are excluded from development. Duplicate development images share a split. Subject identifiers are unavailable. The seven category labels address a different task from multilabel facial movements. The test set is reserved for final evaluation after choosing a configuration with validation loss.

```powershell
& $py src\prepare_fer2013.py
& $py src\train_cnn.py --smoke
& $py src\train_mobilenet.py --smoke
& $py src\train_cnn.py --epochs 24 --batch-size 64
& $py src\train_mobilenet.py --epochs 24 --batch-size 64
& $py src\evaluate_cnn.py
& $py src\evaluate_mobilenet.py
& $py src\export_models.py
& $py src\benchmark_models.py --threads 2
& $py src\benchmark_optimized.py --threads 2
& $py src\compare_models.py
```

Resume an interrupted full job with the same arguments plus `--resume`. The defaults use random initialization, AdamW, validation-loss scheduling and early stopping. For a transfer-learning experiment, the MobileNet trainer accepts `--pretrained` and torchvision may download ImageNet weights. Archive an earlier full checkpoint and its matching history before starting another experiment. Both smoke runs use 12 training and 12 validation images per class, one epoch, separate filenames, and no held-out test images. Evaluation, export, both benchmark commands and comparison also accept `--smoke`. CPU benchmark preprocessing includes decoding, resizing and normalization. RSS covers the Python process and imported libraries. In ONNX verification it also includes the reference PyTorch model. Throughput is sequential batch-one throughput.

## Verification and movement studies

```powershell
& $py -m pytest src\test_pipeline.py -q
& $py src\design_figures.py
& $py src\probe_temporal.py
& $py src\prepare_disfa.py
& $py src\prepare_ckplus.py
& $py src\train_blendshape_classifier.py
& $py src\evaluate_blendshape_classifier.py
```

DISFA expects extracted SNxxx_auN.txt frame/intensity files under datasets/DISFA. CK+ expects peak-frame FACS files under datasets/CK+. Preserve original intensity codes. AU12, AU15, AU1/2, AU4 and AU25/26 are approximate anatomical correspondences for the five project movements. Index files alone cannot train the classifier. Pair them with legitimate media and verified visible movement annotations before combining sources. Never propagate CK+ peak labels to every frame in a sequence.

For an approved collection with a consenting participant, use the same anonymous participant code on every visit. Keep other people outside the camera view. Obtain separate consent before selecting `--save-images`. The default saves features only. An independent observer can record frame-time annotations during the session for later verification. Exclude ambiguous or unverifiable observations.

```powershell
& $py src\collect_custom_dataset.py --participant P001 --labels smile,brows_up --lighting indoor --angle frontal --seconds 5 --hz 5 --consent
& $py src\annotate_custom_dataset.py P001/SESSION_ID
```

The supervised trainer requires verified usable examples across at least five participants and positive and negative training examples per movement. This threshold only permits a split. Reliable research estimates require substantially broader participant and condition coverage. Review errors by viewing angle, illumination, occlusion and speech. Report false announcements per observed minute and onset-to-audible-output latency against time-aligned independent annotations and a recorded audio reference. These measurements require approved observations.

## Edge deployment preparation

The Windows prototype uses MediaPipe Tasks VIDEO on CPU, OpenCV camera capture and pyttsx3. The FER classifiers are research tools. The wearable path requires the downloaded task model only.

For a Raspberry Pi 5 trial use 64-bit Raspberry Pi OS, a supported Python version, a UVC camera and a wired audio output or USB headset. A 4 GB board is a starting estimate. Measure RSS after warm-up before selecting a smaller system. From the copied project directory, prepare a target-specific environment.

```bash
sudo apt install python3-venv espeak-ng libespeak1
python3 -m venv .venv
.venv/bin/python -m pip install mediapipe opencv-contrib-python numpy pyttsx3
.venv/bin/python src/main.py --camera 0 --width 640 --height 480 --on-demand
```

Use a local desktop display for the OpenCV control window. A headless automatic trial uses `--headless` and stops with Ctrl+C. A CSI camera may need a Picamera2 capture adapter. For glasses exposing USB video, select the UVC index on the external compute unit. A proprietary glasses camera requires its vendor's acquisition interface. Select an audio route that keeps ambient speech audible.

For Android, port capture to CameraX ImageAnalysis, bundle the task model in app assets, use MediaPipe Tasks Vision with monotonically increasing frame timestamps, and port the movement thresholds and time-based smoothing as a unit. Request camera permission and choose a locally installed Android TextToSpeech voice that works in airplane mode. Build that native port before testing USB or built-in cameras. The desktop Python command does not install an Android app.

Measure sustained end-to-end latency, dropped frames, peak memory, enclosure temperature and USB input power. Estimate runtime from usable battery watt-hours divided by measured average system watts. Test again under thermal steady state. Battery runtime, Android deployment, Pi deployment, smart-glasses hardware and enclosure thermals await hardware trials. [MediaPipe Python setup](https://developers.google.com/edge/mediapipe/solutions/setup_python) · [Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)

## Optional language model

For a separate experiment using an already available local SmolVLM-256M-Instruct checkpoint and compatible Transformers 4.x install, run `python src/compare_vlm.py --model LOCAL_MODEL_DIR --image CONSENTED_IMAGE`. Review correctness and unsupported details manually. The script uses local files only and checks available RAM before loading. [Model card](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)
