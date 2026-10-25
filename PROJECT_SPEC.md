# Missing Person Identification — OpenCode Build Specification

## 1. Project goal

Build a **local Python computer-vision project** that demonstrates missing-person identification without a frontend, backend server, SQL database, or web application.

The program must support:

1. A single input image.
2. A CCTV/video file.
3. Face detection in the input.
4. Face alignment/cropping.
5. Face embedding generation using a pretrained ArcFace model.
6. Searching a local folder of missing-person reference photographs.
7. Similarity scoring.
8. A configurable match threshold.
9. Visual output showing detected faces and candidate matches.
10. A clear `MATCH FOUND` / `NO MATCH FOUND` result.

The local reference-image folder is the project's **database substitute**.

---

# 2. Final technology stack

| Component | Selected technology | Purpose |
|---|---|---|
| Language | Python 3.10 or 3.11 | Main implementation |
| Image/video processing | OpenCV | Read images/videos, draw boxes, display/save results |
| Face detector | SCRFD | Detect faces and provide facial keypoints |
| Face recognition | ArcFace ONNX | Convert aligned faces into 512-D embeddings |
| ONNX inference | ONNX Runtime | Run ArcFace locally |
| Numerical operations | NumPy | Embeddings, normalization, cosine similarity |
| Dataset 1 | LFW | Face-verification evaluation |
| Dataset 2 | WIDER FACE | Face-detection evaluation |
| Optional Dataset 3 | CelebA | Additional identity/face experiments |
| Reference data | Local `database/` folder | Missing-person reference photographs |
| Interface | Command line + OpenCV windows | No frontend required |
| Database server | None | Intentionally not required |
| Backend | None | Intentionally not required |

## Important design decision

Do **not** train ArcFace from scratch.

The pretrained ArcFace network is used as a **feature/embedding extractor**:

```text
face image
    ↓
ArcFace
    ↓
512-dimensional embedding
```

Two embeddings can then be compared using cosine similarity.

The Hugging Face ArcFace ONNX model documents a 112×112 RGB input and 512-dimensional output and provides an ONNX Runtime example. The current model file is about 137 MB.  
Source: https://huggingface.co/onnx-community/arcface-onnx

---

# 3. Overall architecture

```text
                    ┌───────────────────────────┐
                    │   Local Reference Folder  │
                    │        database/          │
                    │                           │
                    │ person_001/               │
                    │ person_002/               │
                    │ person_003/               │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                         Face Detection
                                  │
                                  ▼
                           Face Alignment
                                  │
                                  ▼
                           ArcFace ONNX
                                  │
                                  ▼
                       Reference Embeddings
                                  │
                                  │
       ┌──────────────────────────┴─────────────────────────┐
       │                                                    │
       ▼                                                    ▼
  Input Image                                          CCTV Video
       │                                                    │
       └──────────────────────────┬─────────────────────────┘
                                  ▼
                           Face Detection
                                  │
                                  ▼
                           Face Alignment
                                  │
                                  ▼
                       ArcFace Query Embedding
                                  │
                                  ▼
                         Cosine Similarity
                                  │
                                  ▼
                          Best Candidate
                                  │
                                  ▼
                        Threshold Decision
                           /           \
                          /             \
                    MATCH FOUND      NO MATCH
```

---

# 4. Recommended project directory

Create this structure:

```text
missing-person-identification/
│
├── database/
│   ├── person_001/
│   │   ├── front.jpg
│   │   ├── left.jpg
│   │   └── right.jpg
│   │
│   ├── person_002/
│   │   ├── front.jpg
│   │   └── side.jpg
│   │
│   └── person_003/
│       ├── front.jpg
│       └── outdoor.jpg
│
├── input/
│   ├── images/
│   │   ├── test_01.jpg
│   │   └── test_02.jpg
│   │
│   └── videos/
│       └── cctv_test.mp4
│
├── models/
│   ├── arcface.onnx
│   └── detector/
│
├── evaluation/
│   ├── lfw/
│   └── widerface/
│
├── src/
│   ├── detector.py
│   ├── aligner.py
│   ├── arcface.py
│   ├── database.py
│   ├── matcher.py
│   ├── image_pipeline.py
│   ├── video_pipeline.py
│   └── main.py
│
├── results/
│   ├── images/
│   └── videos/
│
├── requirements.txt
├── README.md
└── PROJECT_SPEC.md
```

Keep `database/` separate from `evaluation/`. The reference folder represents the missing-person records used by the demo; public datasets are for evaluation/testing.

---

# 5. Python environment

Use Python 3.10 or 3.11.

Create a virtual environment.

## Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the main dependencies:

```bash
pip install numpy opencv-python onnxruntime scipy tqdm pandas
```

If the detector is implemented through InsightFace:

```bash
pip install insightface
```

The project should not require a web server, Flask, FastAPI, React, Node.js, PostgreSQL, MongoDB, or a vector database.

---

# 6. Face detector — SCRFD

## Why SCRFD?

SCRFD is a face detector designed for efficient and accurate face detection. It is especially useful for this project because the detector can provide facial keypoints, which can be used for alignment before ArcFace.

The official InsightFace model zoo reports multiple SCRFD variants. The larger SCRFD models provide stronger detection performance but require more compute.

Official model zoo:

https://github.com/deepinsight/insightface/tree/master/model_zoo

InsightFace documents:

- SCRFD_500M
- SCRFD_1G
- SCRFD_2.5G
- SCRFD_10G
- SCRFD_34G
- keypoint-enabled variants

For a student laptop, start with **SCRFD_500M_KPS** or **SCRFD_2.5G_KPS**.

`KPS` means the detector provides facial keypoints.

## Important

Do not train SCRFD.

Use pretrained weights.

A convenient alternative is to use the InsightFace Python package/model package for the detector and use its detected five-point facial landmarks for alignment.

The official InsightFace model zoo also provides `buffalo_l`, which packages SCRFD-10GF with a ResNet50-based recognition model. However, for this project the recognition stage should remain explicitly ArcFace ONNX so that the architecture is easy to explain and reproduce.

Official InsightFace model information:

https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md

---

# 7. Face alignment

ArcFace should receive a properly aligned face.

The detector provides five facial landmarks:

```text
left eye
right eye
nose
left mouth corner
right mouth corner
```

Use these points to perform a similarity/affine transformation into a 112×112 canonical face.

Conceptually:

```text
Detected face
     +
5 facial landmarks
     ↓
Geometric alignment
     ↓
112 × 112 RGB face
     ↓
ArcFace
```

Do not simply resize the entire detected bounding box if landmark-based alignment is available.

---

# 8. ArcFace model

## Selected model

Hugging Face:

`onnx-community/arcface-onnx`

Model page:

https://huggingface.co/onnx-community/arcface-onnx

Model file:

`arcface.onnx`

Current file size is approximately 137 MB.

Model information:

- Input: cropped RGB face
- Input size: 112 × 112
- Output: 512-dimensional embedding
- Runtime: ONNX Runtime
- Purpose: face verification / recognition

Model documentation:

https://huggingface.co/onnx-community/arcface-onnx/blob/main/README.md

## Download

Install the Hugging Face CLI if desired:

```bash
pip install -U huggingface_hub
```

Then:

```bash
hf download onnx-community/arcface-onnx arcface.onnx --local-dir models
```

If that command is not available in the installed CLI version, download the model from the Hugging Face model page and save it as:

```text
models/arcface.onnx
```

Direct model file:

https://huggingface.co/onnx-community/arcface-onnx/blob/main/arcface.onnx

The Hugging Face model card also provides an ONNX Runtime example.

---

# 9. ArcFace preprocessing

For each aligned face:

```text
BGR image from OpenCV
        ↓
BGR → RGB
        ↓
Resize to 112 × 112
        ↓
float32
        ↓
normalization
        ↓
batch dimension
        ↓
ONNX Runtime
```

The model card uses normalization approximately equivalent to:

```python
img = (img.astype(np.float32) - 127.5) / 128.0
```

Do not invent a different preprocessing pipeline without validating it against the model.

Check the model card if the model revision changes.

---

# 10. Embedding generation

The ArcFace output is a 512-dimensional vector.

Example conceptual result:

```text
[0.013, -0.087, 0.221, ... , 0.031]
```

Normalize it:

```python
embedding = embedding / np.linalg.norm(embedding)
```

Store the normalized embedding in memory.

For each reference photograph:

```text
reference image
      ↓
detect face
      ↓
landmarks
      ↓
align
      ↓
ArcFace
      ↓
512-D embedding
```

The program should precompute reference embeddings once when the application starts.

---

# 11. Reference "database"

Do not create SQL tables.

Use folders.

Recommended structure:

```text
database/
│
├── person_001/
│   ├── front.jpg
│   ├── left.jpg
│   └── outdoor.jpg
│
├── person_002/
│   ├── front.jpg
│   ├── right.jpg
│   └── glasses.jpg
│
└── person_003/
    ├── front.jpg
    └── side.jpg
```

The directory name becomes the identity label.

For example:

```text
database/person_001/front.jpg
```

means:

```text
Person ID = person_001
```

Do not put multiple unrelated people into one person's directory.

---

# 12. Multiple reference images per person

Use 2–5 reference images per person if possible.

Good reference variation:

- frontal photograph
- slight left pose
- slight right pose
- indoor photograph
- outdoor photograph
- different lighting
- glasses/no glasses when relevant

This is much better than using one photograph.

The system can generate an embedding for each image and either:

1. Compare the query with every reference image and take the highest similarity, or
2. Build a representative identity embedding.

For the first version, use option 1 because it is simpler to explain.

```text
person_001
   ├── embedding_1
   ├── embedding_2
   └── embedding_3

query
   ↓
compare against all three
   ↓
best similarity = identity score
```

---

# 13. Matching algorithm

Use cosine similarity.

For normalized embeddings:

```python
similarity = np.dot(query_embedding, reference_embedding)
```

Conceptually:

```text
1.0     extremely similar
0.9     very similar
0.7     less similar
0.4     probably unrelated
```

Do NOT assume these values are universal identity thresholds.

The actual threshold must be experimentally calibrated for the selected model and preprocessing pipeline.

---

# 14. Threshold calibration

This is important for the project report.

Create two groups:

### Positive pairs

Two images of the same person:

```text
image A → person_001
image B → person_001
```

Calculate:

```text
similarity(A, B)
```

### Negative pairs

Images of different people:

```text
image A → person_001
image B → person_002
```

Calculate:

```text
similarity(A, B)
```

Plot the two similarity distributions.

Choose a threshold that gives a reasonable trade-off between:

- False Accept Rate (FAR)
- False Reject Rate (FRR)

For the classroom demonstration, the system should report a **potential match**, not claim legally confirmed identity.

---

# 15. LFW dataset

## Purpose

Use LFW primarily for **face verification evaluation**.

LFW contains approximately:

- 13,233 face images
- 5,749 identities
- 6,000 standard verification pairs
- same-person and different-person pairs

A Hugging Face version is available here:

https://huggingface.co/datasets/marcelohaps/lfw

The dataset includes metadata and the standard verification pairs.

The official LFW project is:

https://vis-www.cs.umass.edu/lfw/

## Download

Recommended first option:

```bash
pip install -U huggingface_hub
```

Then download the dataset repository:

```bash
hf download marcelohaps/lfw --repo-type dataset --local-dir evaluation/lfw
```

Alternatively, download it manually from the Hugging Face dataset page.

## How to use it

Do NOT use all 13,233 images as your project "missing person database".

Instead use it to test:

```text
same person
vs
different person
```

Example:

```text
LFW image A
     ↓
ArcFace
     ↓
embedding A

LFW image B
     ↓
ArcFace
     ↓
embedding B

cosine similarity
```

Then compare the score with your threshold.

---

# 16. WIDER FACE dataset

## Purpose

Use WIDER FACE to evaluate the **face detector**, not the identity-recognition component.

Official information:

https://shuoyang1213.me/WIDERFACE/

The dataset contains:

- 32,203 images
- 393,703 annotated faces
- substantial variation in scale, pose and occlusion

Source paper:

https://shuoyang1213.me/WIDERFACE/support/paper.pdf

## Download

Use the official WIDER FACE project page:

https://shuoyang1213.me/WIDERFACE/

Download the image archive and annotations from the official links.

Organize it as:

```text
evaluation/
└── widerface/
    ├── WIDER_train/
    ├── WIDER_val/
    └── wider_face_split/
```

Do not automatically download the entire dataset if disk space is limited.

For this project, a representative subset is sufficient for demonstrating detector behavior.

---

# 17. Optional CelebA dataset

CelebA is optional.

Official page:

https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

It contains:

- 202,599 images
- 10,177 identities
- 5 facial landmark locations
- 40 facial attributes

It can support additional face-recognition/identity experiments.

However, **do not make CelebA a mandatory dependency** for the first working version.

Its official terms state that it is available for non-commercial research purposes only and restrict redistribution/commercial use.

For a subject project, LFW + WIDER FACE is enough.

---

# 18. Actual missing-person reference data

Public face datasets such as LFW and CelebA contain public-figure/celebrity imagery and should not simply be presented as a real missing-person database.

For the actual demonstration:

```text
database/
```

should contain photographs that you have permission to use for the project.

Good options:

- photographs supplied by the project team with consent
- synthetic/demo identities
- publicly licensed images where the license permits the intended educational use

Keep the dataset documentation clear about the source and permissions.

---

# 19. Image pipeline

Implement:

```text
src/image_pipeline.py
```

Expected flow:

```text
input image
    ↓
load with OpenCV
    ↓
detect all faces
    ↓
for each face:
    get bounding box
    get 5 landmarks
    align face
    generate ArcFace embedding
    compare with reference embeddings
    select best candidate
    threshold decision
    draw result
    ↓
save/display output
```

Output should show:

```text
┌─────────────────────────────────────┐
│                                     │
│       CCTV / Input Image            │
│                                     │
│       ┌──────────────┐              │
│       │    FACE      │              │
│       └──────────────┘              │
│       person_001                    │
│       similarity: 0.87              │
│       MATCH                         │
│                                     │
└─────────────────────────────────────┘
```

If no candidate passes the threshold:

```text
UNKNOWN
No match found
Best similarity: 0.42
```

---

# 20. Video/CCTV pipeline

Implement:

```text
src/video_pipeline.py
```

Read the video using:

```python
cv2.VideoCapture(...)
```

Do not necessarily run recognition on every frame.

For example:

```text
Frame 1
Frame 2
Frame 3
...
```

can be sampled:

```text
Frame 1
Frame 6
Frame 11
Frame 16
...
```

Start with a configurable parameter:

```python
FRAME_SKIP = 5
```

For each sampled frame:

```text
video frame
     ↓
face detection
     ↓
each detected face
     ↓
alignment
     ↓
ArcFace embedding
     ↓
similarity search
     ↓
draw result
```

Save an annotated output video:

```text
results/videos/cctv_result.mp4
```

---

# 21. Avoid repeated match messages

CCTV contains the same person across many consecutive frames.

Do not print:

```text
MATCH person_001
MATCH person_001
MATCH person_001
MATCH person_001
...
```

for every frame.

At the first implementation stage, it is acceptable to display the label on each frame.

As an improvement, aggregate consecutive detections:

```text
Frames 100–150
→ person_001
→ average similarity = 0.89
→ one detection event
```

This makes the demonstration much cleaner.

---

# 22. Main program interface

Keep it simple.

Example:

```bash
python src/main.py --mode image --input input/images/test_01.jpg
```

Video:

```bash
python src/main.py --mode video --input input/videos/cctv_test.mp4
```

Optional threshold:

```bash
python src/main.py --mode image --input input/images/test_01.jpg --threshold 0.60
```

The exact threshold should eventually come from your validation experiment.

---

# 23. Expected startup behavior

When the program starts:

```text
========================================
 MISSING PERSON IDENTIFICATION SYSTEM
========================================

Loading face detector...
Detector loaded.

Loading ArcFace model...
ArcFace loaded.

Loading reference database...

person_001:
  3 reference images
  3 embeddings

person_002:
  2 reference images
  2 embeddings

person_003:
  2 reference images
  2 embeddings

Reference database ready.

Ready for input.
```

Then process the requested image/video.

---

# 24. Expected image output

Example:

```text
Input image: test_01.jpg

Faces detected: 2

Face #1
  Best candidate: person_002
  Similarity: 0.873
  Result: MATCH FOUND

Face #2
  Best candidate: person_007
  Similarity: 0.431
  Result: NO MATCH

Output saved:
results/images/test_01_result.jpg
```

---

# 25. Expected video output

Example:

```text
Video: cctv_test.mp4
FPS: 25
Frames: 1500

Frame 125:
  Face 1 → person_002 → 0.873 → MATCH

Frame 130:
  Face 1 → person_002 → 0.881 → MATCH

Frame 245:
  Face 1 → unknown → 0.421 → NO MATCH

...

Annotated video saved:
results/videos/cctv_test_result.mp4
```

---

# 26. Source code responsibilities

## `detector.py`

Responsibilities:

- load SCRFD
- detect faces
- return bounding boxes
- return confidence
- return five facial keypoints

API concept:

```python
detect_faces(image) -> list[FaceDetection]
```

---

## `aligner.py`

Responsibilities:

- receive face image + landmarks
- align to ArcFace canonical geometry
- return 112×112 RGB face

API:

```python
align_face(image, landmarks) -> aligned_face
```

---

## `arcface.py`

Responsibilities:

- load `models/arcface.onnx`
- preprocess image
- run ONNX inference
- L2-normalize embedding

API:

```python
get_embedding(aligned_face) -> np.ndarray
```

Expected shape:

```text
(512,)
```

---

## `database.py`

Responsibilities:

- scan `database/`
- load reference images
- generate embeddings
- associate embeddings with person IDs

Concept:

```python
reference_database = {
    "person_001": [embedding1, embedding2, embedding3],
    "person_002": [embedding1, embedding2],
}
```

No SQL database.

---

## `matcher.py`

Responsibilities:

- cosine similarity
- compare query against all reference embeddings
- return highest scoring candidate
- apply threshold

API:

```python
match_embedding(query_embedding, reference_database, threshold)
```

Return:

```python
{
    "person_id": "person_001",
    "similarity": 0.873,
    "matched": True
}
```

---

## `image_pipeline.py`

Responsibilities:

- load image
- detect faces
- align faces
- create embeddings
- search database
- annotate image
- save result

---

## `video_pipeline.py`

Responsibilities:

- open video
- sample frames
- detect faces
- recognize faces
- annotate frames
- write output video

---

## `main.py`

Responsibilities:

- parse command-line arguments
- load models
- load reference database
- choose image/video mode
- run pipeline

---

# 27. Requirements file

Create:

```text
requirements.txt
```

Initial contents:

```text
numpy
opencv-python
onnxruntime
scipy
pandas
tqdm
huggingface_hub
insightface
```

Pin exact versions after the first successful installation so that OpenCode can reproduce the environment.

Do not blindly pin versions before testing on the target machine.

---

# 28. Development order

OpenCode should build the project in this order.

## Phase 1 — Environment

1. Create virtual environment.
2. Install dependencies.
3. Verify OpenCV.
4. Verify ONNX Runtime.

Test:

```python
import cv2
import numpy
import onnxruntime

print(cv2.__version__)
print(numpy.__version__)
print(onnxruntime.__version__)
```

---

## Phase 2 — ArcFace

1. Download `arcface.onnx`.
2. Load it with ONNX Runtime.
3. Feed a known aligned face.
4. Verify output shape is `(1, 512)` or equivalent.
5. Normalize embedding.

Do not proceed until this works.

---

## Phase 3 — Face detector

1. Install/use SCRFD.
2. Load pretrained weights.
3. Detect faces in a photograph.
4. Verify bounding boxes.
5. Verify five landmarks.

Do not proceed until landmarks are correct.

---

## Phase 4 — Alignment

1. Implement five-point alignment.
2. Save aligned faces for debugging.
3. Visually inspect them.

Expected:

```text
eyes approximately horizontal
face centered
face size ≈ 112×112
```

---

## Phase 5 — Single-image recognition

Build:

```text
reference image
        ↓
embedding
```

and:

```text
query image
        ↓
embedding
```

Then:

```text
cosine similarity
```

Verify that same-person images score higher than different-person images.

---

## Phase 6 — Reference-folder search

Implement:

```text
database/
    person_001/
    person_002/
    person_003/
```

Generate all reference embeddings at startup.

---

## Phase 7 — Threshold calibration

Use LFW pairs.

Measure:

```text
positive similarities
negative similarities
```

Choose a threshold.

Document the threshold and why it was selected.

---

## Phase 8 — Image demonstration

Input:

```text
test.jpg
```

Output:

```text
MATCH FOUND
person_002
similarity = 0.87
```

or:

```text
NO MATCH FOUND
best similarity = 0.43
```

---

## Phase 9 — CCTV demonstration

Input:

```text
cctv_test.mp4
```

Process sampled frames.

Draw:

```text
bounding box
person ID
similarity
match/no-match
```

Save:

```text
cctv_test_result.mp4
```

---

# 29. Evaluation plan

The project should report at least:

### Face detection

Use WIDER FACE or a representative subset.

Metrics:

- Precision
- Recall
- optionally IoU/mAP if implementing formal detector evaluation

### Face verification

Use LFW.

Metrics:

- Accuracy
- False Accept Rate
- False Reject Rate
- optionally ROC curve
- optionally AUC

### System demonstration

Report:

- number of reference persons
- number of reference photographs
- number of query images
- number of CCTV videos
- selected similarity threshold
- correct matches
- false matches
- missed matches
- average processing time

---

# 30. What NOT to build

Do not add these unless the subject specifically requires them:

```text
❌ React frontend
❌ HTML/CSS frontend
❌ Flask/FastAPI backend
❌ PostgreSQL
❌ MongoDB
❌ Firebase
❌ Cloud deployment
❌ REST API
❌ Authentication
❌ User accounts
❌ Vector database
❌ Model training from scratch
```

This is intentionally a local ML/CV demonstration.

---

# 31. Important distinction: dataset vs project database

The project has two different types of data.

## Evaluation datasets

```text
evaluation/
├── lfw/
└── widerface/
```

Purpose:

```text
measure model/system behavior
```

## Missing-person reference repository

```text
database/
├── person_001/
├── person_002/
└── person_003/
```

Purpose:

```text
perform the actual demonstration search
```

Never mix these directories.

---

# 32. Data privacy and responsible use

This is a face-identification system.

Only use reference photographs and CCTV footage that the project team is authorized to use.

For the academic demonstration:

- obtain consent where appropriate;
- do not publish identifiable personal data unnecessarily;
- do not claim that a similarity score proves identity;
- present results as potential matches requiring human verification;
- document the source and permitted use of datasets;
- do not use the system for real-world surveillance without appropriate authorization.

---

# 33. Model/data licensing notes

### ArcFace / Hugging Face

Check the exact model repository license and model card before redistribution.

Hugging Face model page:

https://huggingface.co/onnx-community/arcface-onnx

### InsightFace

The official model zoo states that its pretrained models are for **non-commercial research purposes**.

Source:

https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md

### CelebA

The official CelebA page states that the dataset is available for **non-commercial research purposes** and includes restrictions on redistribution.

Source:

https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

### LFW

Check the original LFW terms before redistribution:

https://vis-www.cs.umass.edu/lfw/

### WIDER FACE

Use the official project download and usage information:

https://shuoyang1213.me/WIDERFACE/

---

# 34. Recommended minimum download set

Do NOT download everything initially.

Start with:

```text
1. ArcFace ONNX
2. SCRFD detector/model package
3. LFW
4. Small WIDER FACE subset
5. Your own authorized reference photographs
6. One short CCTV test video
```

You can add CelebA later if the evaluation requires it.

---

# 35. OpenCode implementation instructions

OpenCode should implement the project incrementally.

### First instruction

Build the Python project structure and environment setup.

### Second instruction

Implement SCRFD face detection and five-point landmark extraction.

### Third instruction

Implement ArcFace ONNX loading and 512-D embedding generation.

### Fourth instruction

Implement five-point face alignment.

### Fifth instruction

Implement local reference-folder indexing.

### Sixth instruction

Implement cosine-similarity matching.

### Seventh instruction

Implement image inference.

### Eighth instruction

Implement CCTV/video inference.

### Ninth instruction

Implement LFW evaluation and threshold calibration.

### Tenth instruction

Add WIDER FACE detector evaluation if time permits.

After each phase, run a small test before continuing.

Do not rewrite working components unnecessarily.

---

# 36. Final target

The finished project should allow:

```bash
python src/main.py --mode image --input input/images/test_01.jpg
```

and:

```text
MISSING PERSON IDENTIFICATION

Loading models... OK
Loading reference images... OK

Faces detected: 1

Best candidate:
    Person: person_002
    Similarity: 0.873

Result:
    MATCH FOUND

Saved:
    results/images/test_01_result.jpg
```

And:

```bash
python src/main.py --mode video --input input/videos/cctv_test.mp4
```

with:

```text
Processing CCTV video...

Frame 125:
    Face detected
    Candidate: person_002
    Similarity: 0.873
    MATCH

Frame 250:
    Face detected
    Best similarity: 0.421
    NO MATCH

Processing complete.

Output:
results/videos/cctv_test_result.mp4
```

---

# 37. Final architecture summary

```text
                 REFERENCE IMAGES
                  database/
                       │
                       ▼
                 SCRFD Detector
                       │
                       ▼
                Face Landmarks
                       │
                       ▼
                 Face Alignment
                       │
                       ▼
                  ArcFace ONNX
                       │
                       ▼
               512-D Embeddings
                       │
                       │
                       │
IMAGE ───────► SCRFD ──► Alignment ──► ArcFace
                                      │
                                      ▼
                              Query Embedding
                                      │
                                      ▼
                              Cosine Similarity
                                      │
                                      ▼
                              Candidate Ranking
                                      │
                                      ▼
                             Threshold Decision
                                /           \
                               /             \
                        MATCH FOUND       NO MATCH
```

The core academic contribution is therefore the integration of:

**face detection + facial landmark alignment + pretrained deep face embeddings + similarity-based identification + video-frame processing + quantitative evaluation.**

No custom training is required for the initial system.
