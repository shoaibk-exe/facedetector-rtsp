# Face Recognition System

Staff vs **Unknown** recognition using InsightFace Buffalo_L.

Pipeline: quality filter → track → collect 20 frames → top 5 → average embeddings → multi-gallery match → temporal vote.

Works for **live RTSP** and **MP4** — switch with `--source` only. Cameras/videos are configured in `cameras.yaml`.

---

## 1. Setup (once)

### Option A — Conda (recommended if you use `dw`)

```powershell
cd D:\dw\face_recognition_system
conda activate dw
pip install -r requirements.txt
copy .env.example .env
```

### Option B — Python venv

```powershell
cd D:\dw\face_recognition_system
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` if needed (`THRESHOLD`, `ONNX_PROVIDERS=cpu` if no GPU, etc.).

Edit `cameras.yaml` for RTSP URLs and MP4 paths.

---

## 2. Enroll staff (required before recognize)

Put many images per person (prefer CCTV frames):

```text
staff/
  ali/
    01.jpg
    02.jpg
    ...
  ritesh/
    ...
```

```powershell
python enroll_staff.py
```

Calibrate threshold on your gallery (optional but recommended):

```powershell
python calibrate_threshold.py
```

Then set `THRESHOLD` in `.env` to the suggested value.

---

## 3. List configured sources

```powershell
python recognize.py --list-cameras
```

Shows both `rtsp:` and `mp4:` sections from `cameras.yaml`.

---

## 4. Run — RTSP (live cameras)

### All enabled cameras + live display

```powershell 
python recognize.py --source rtsp --display
```

Press **q** in a preview window to quit, or **Ctrl+C** in the terminal.

### All enabled cameras — detection only (no windows)

```powershell
python recognize.py --source rtsp --no-display
```

### One camera

```powershell
python recognize.py --source rtsp --cameras cam1 --display
```

### Several cameras

```powershell
python recognize.py --source rtsp --cameras cam1,cam2,cam3 --display
```

### Display + save staff clips + detailed log

```powershell
python recognize.py --source rtsp --display --save-staff --log-detections
```

### Full recording of each stream + staff clips + log

```powershell
python recognize.py --source rtsp --display --save --save-staff --log-detections
```

### Override threshold for this run

```powershell
python recognize.py --source rtsp --display --threshold 0.40
```

### CPU only (if CUDA fails)

```powershell
$env:ONNX_PROVIDERS="cpu"
python recognize.py --source rtsp --display
```

---

## 5. Run — MP4

### Which file controls the video?

```
  README.md  ──X──>  NOT used by the app (docs only)

  cameras.yaml  ──✓──>  THIS is what recognize.py reads
```

**Do not edit README to change videos.** Open **`cameras.yaml`** in the project root.

Example — edit the `mp4:` section in **`cameras.yaml`**:

```yaml
mp4:
  - id: vid1
    name: Camera 3 customer enter
    enabled: true
    path: "input/New folder/Camera_3_customer_enter_clean_20260713_225257.mp4"
```

Use forward slashes `/` in the path (works on Windows too).

Run:

```powershell
python recognize.py --source mp4 --cameras vid1 --display
```

**Option B — drop files in `input/` (no yaml edit)**

If nothing is enabled in `mp4:`, the app auto-picks every `input/*.mp4`:

```powershell
# 1) Copy your video to input/
# 2) Run:
python recognize.py --source mp4 --display
```

**Force one yaml entry (even if disabled)**

```powershell
python recognize.py --source mp4 --cameras vid1 --display
```

# Specific videos
python recognize.py --source mp4 --cameras vid1,vid2

# With staff clips + detailed log
python recognize.py --source mp4 --save-staff --log-detections

# With custom threshold
python recognize.py --source mp4 --cameras vid1 --threshold 0.40
```

Annotated outputs go to `output/`.

Example `cameras.yaml` MP4 entry:

```yaml
mp4:
  - id: vid1
    name: Store front
    enabled: true
    path: "input\WhatsApp Video 2025-12-19 at 1.09.29 PM (1).mp4"
```

---

## 6. All flags

| Flag | Meaning |
|------|---------|
| `--source rtsp` | Use `rtsp:` section in `cameras.yaml` |
| `--source mp4` | Use `mp4:` section in `cameras.yaml` |
| `--cameras cam1,cam3` | Run only these ids |
| `--display` | Show live preview windows (RTSP) |
| `--no-display` | No preview; detection only (RTSP) |
| `--save-staff` | Save clip when staff identity is **confirmed** |
| `--log-detections` | Write per-frame log with confidence % |
| `--save` | Record full annotated RTSP stream per camera |
| `--threshold 0.40` | Override `.env` match threshold |
| `--list-cameras` | Print RTSP + MP4 entries and exit |

---

## 7. Outputs

| Output | Location |
|--------|----------|
| Annotated videos | `output/` |
| Staff event clips | `output/staff_clips/<id>/` |
| Detection logs | `logs/detections_YYYYMMDD_HHMMSS.log` |
| Staff gallery | `database/embeddings.pkl` |

---

## 8. `cameras.yaml` layout

```yaml
rtsp:
  - id: cam1
    name: Channel 1
    enabled: true
    url: "rtsp://..."

  - id: cam2
    name: Channel 4
    enabled: true
    url: "rtsp://..."

mp4:
  - id: vid1
    name: Temp video 1
    enabled: false
    path: "input/temp_cam1.mp4"
```

- Change mode with `--source rtsp` or `--source mp4`
- Add/remove/disable cams by editing this file only
- Force a disabled item with `--cameras <id>`

---

## 9. Quick command cheat sheet

```powershell
cd D:\dw\face_recognition_system
conda activate dw

# Setup data
python enroll_staff.py
python calibrate_threshold.py
python recognize.py --list-cameras

# Live
python recognize.py --source rtsp --display
python recognize.py --source rtsp --cameras cam1 --display
python recognize.py --source rtsp --display --save-staff --log-detections
python recognize.py --source rtsp --no-display

# Files
python recognize.py --source mp4
python recognize.py --source mp4 --cameras vid1 --save-staff --log-detections
```

---

## 10. Troubleshooting

| Problem | Fix |
|---------|-----|
| `Staff database not found` | Run `python enroll_staff.py` |
| Unknown camera / video id | `python recognize.py --list-cameras` |
| MP4 skipped / file not found | Fix `path:` and set `enabled: true` in `cameras.yaml` |
| RTSP connect fails | Check URL / password (`@` → `%40`), network, `subtype` 0 vs 1 |
| CUDA provider warning | `$env:ONNX_PROVIDERS="cpu"` or set in `.env` |
| Stuck on `tracking N/20` | Wait for quality frames; lower `MIN_FACE_WIDTH` / `MIN_BLUR_SCORE` in `.env` if needed |
| Many false staff matches | Raise `THRESHOLD`; run `calibrate_threshold.py` |
