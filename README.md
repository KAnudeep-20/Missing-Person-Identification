# Missing Person Identification

A local Python computer-vision project demonstrating missing-person
identification from images and CCTV footage — no frontend, no backend, no
SQL database. The local `database/` folder is the database substitute.

Full build specification: see `PROJECT_SPEC.md`.

## Architecture

```text
database/ reference photos          input image / CCTV video
        |                                  |
   SCRFD detection                    SCRFD detection
        |                                  |
   5-point landmark alignment         5-point landmark alignment
        |                                  |
   ArcFace ONNX (512-D embedding)     ArcFace ONNX (512-D embedding)
        └──────────── cosine similarity + threshold ───────────┘
                              |
                  MATCH FOUND  /  NO MATCH FOUND
```

## Project layout

```text
database/      person_001/, person_002/, ... (reference photos; folder = ID)
input/         images/ and videos/ (query images / CCTV clips)
models/        arcface.onnx, detector/ (SCRFD weights auto-downloaded)
evaluation/    lfw/, widerface/ (public datasets for evaluation only)
src/           detector.py aligner.py arcface.py database.py matcher.py
               image_pipeline.py video_pipeline.py main.py
results/       images/, videos/ (annotated outputs)
```

## Setup

Requires Python 3.10 or 3.11.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Model download

```powershell
hf download onnx-community/arcface-onnx arcface.onnx --local-dir models
```

The SCRFD detector weights (`buffalo_l` detection module via the InsightFace
model zoo) download automatically on first run into `models/detector/`.

## Usage

```powershell
python src/main.py --mode image --input input/images/test_01.jpg
python src/main.py --mode video --input input/videos/cctv_test.mp4
python src/main.py --mode image --input input/images/test_01.jpg --threshold 0.30
```

## Threshold

The default similarity threshold (0.30) is **not** a universal number — it is
calibrated on 1,991 LFW verification pairs for this exact model + alignment +
preprocessing pipeline (see `results/lfw_threshold_report.txt`):

- same-person similarity: mean 0.55, std 0.17
- different-person similarity: mean 0.04, std 0.09
- AUC 0.981 | max accuracy 0.965 @ t=0.28 | FAR 0.3% / FRR 7.6% @ t=0.30

Recalibrate after changing the model or preprocessing:
`python src/evaluate_lfw.py --limit 2000`

## Adding reference people

Create one folder per person inside `database/`; the folder name is the
person ID:

```text
database/person_001/front.jpg
database/person_001/left.jpg
```

Use 2-5 photos per person with pose/lighting variation. Photos must be
supplied with the consent of (and authorization for) the people in them.

## Responsible use

Scores are *potential* matches requiring human verification — a similarity
score does not prove identity. Only use photographs and footage you are
authorized to use (see `PROJECT_SPEC.md` sections 32-33).

## Evaluation

LFW is used for face-verification evaluation and threshold calibration;
WIDER FACE (optional subset) for detector evaluation. See
`evaluation/` and `PROJECT_SPEC.md` sections 14-16 and 29.
