# innovX VisualNav

**GPS-Denied Drone Visual Localization** · AI-Powered Visual Position Recovery

A working prototype that recovers a drone's position by matching a single
downward-facing camera frame against a stored reference map — no GPS, no IMU,
no SLAM. Upload a satellite/orthomosaic map and a drone capture; the system
searches the map, verifies the geometry, and either highlights exactly where
the frame came from or says it cannot find it.

```
Drone Image → Image Processing → Candidate Map Search → Feature Matching
            → Geometric Verification → Exact Map Region → Optional GPS Coordinate
```

---

## Table of contents

- [Why this exists](#why-this-exists)
- [Architecture](#architecture)
- [What it does not do](#what-it-does-not-do)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running](#running)
- [Generating test data](#generating-test-data)
- [Testing strategy](#testing-strategy)
- [How localization works](#how-localization-works)
- [Confidence and the four verdicts](#confidence-and-the-four-verdicts)
- [Georeferencing and GPS](#georeferencing-and-gps)
- [QGroundControl `.plan` files](#qgroundcontrol-plan-files)
- [API documentation](#api-documentation)
- [Configuration](#configuration)
- [Folder structure](#folder-structure)
- [Presentation flow](#presentation-flow)
- [Known limitations](#known-limitations)

---

## Why this exists

A drone that loses GPS still has a camera. If the ground below it appears
somewhere in a reference map that was loaded before the flight, its position is
recoverable from imagery alone. The hard part is not finding *a* match — it is
refusing to report a confident position when the match is wrong. Aerial scenes
are full of repeated structure: identical rooftops, regular field grids,
parallel roads. A system that always returns its best tile will confidently
place the drone in the wrong place.

VisualNav is built around that constraint. Retrieval only *proposes* candidate
regions; geometry *disposes*. Every candidate must produce a homography that is
numerically well conditioned, projects the drone frame to a plausible convex
quadrilateral, and is supported by inliers spread across the whole frame rather
than clustered on one building. Candidates that fail are rejected with a named
reason, and if none survive, the answer is `NO_MATCH`.

---

## Architecture

```
                          ┌────────────────────┐
       Reference map ───► │  Multi-scale tiling│  8–25% of map area, 25% overlap
                          └─────────┬──────────┘
                                    ▼
                          ┌────────────────────┐
                          │  Global embeddings │  DINOv2 (frozen) or classical
                          │   cached per map   │  gradient+colour descriptor
                          └─────────┬──────────┘
                                    │
       Drone frame ───┬─────────────┼──────────────────────────┐
                      │             │                          │
            MATCHING BRANCH         │                 VISUALIZATION BRANCH
                      │             │                          │
              normalize (CLAHE)     │                   enhance / grayscale
                      │             │                   edges / contours
              global embedding ─────┤                   Structural Terrain View
                      │             ▼                          │
                      │      cosine similarity                 │  (explainability
                      │             │                          │   renders only —
                      │             ▼                          │   never fed back
                      │      Top-K candidates                  │   into matching)
                      │             │                          │
              SuperPoint keypoints  │                          │
                      │             ▼                          │
                      └───► LightGlue matching ◄── candidate tile features
                                    │
                                    ▼
                          RANSAC + homography
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │   Geometric verification     │
                     │  convexity · scale · shear   │
                     │  reprojection · coverage     │
                     └──────────┬───────────────────┘
                                │
                   ┌────────────┴────────────┐
                   ▼                         ▼
             valid geometry            all rejected
                   │                         │
                   ▼                         ▼
      confidence engine (5 terms)        NO_MATCH
                   │
     ┌─────────────┼──────────────┬──────────────┐
     ▼             ▼              ▼              ▼
MATCH_FOUND  LOW_CONFIDENCE   AMBIGUOUS      NO_MATCH
     │
     ▼
frame corners + centre projected into map pixels → optional lat/lon
```

**Graceful degradation.** Every heavy dependency is optional. With PyTorch
installed the system uses DINOv2 for retrieval and SuperPoint + LightGlue for
matching. Without it, retrieval falls back to a classical gradient-orientation
+ colour descriptor and matching falls back to SIFT + FLANN + Lowe ratio. The
same geometric verification, confidence engine and UI run in both cases, and
the active backend is reported honestly in the UI and in `/api/system/info`.

---

## What it does not do

Deliberately out of scope for this stage:

- autonomous drone navigation or flight-controller integration
- SLAM, visual odometry or optical flow
- GPS/IMU sensor fusion
- training a custom neural network (only pretrained models are used)

---

## Requirements

| | |
|---|---|
| Python | 3.10 – 3.13 |
| Node.js | 18+ (20+ recommended) |
| OS | Windows, macOS or Linux |
| GPU | Optional. CUDA/MPS used automatically when available, else CPU |
| Disk | ~200 MB for the core stack, ~3 GB more if you install PyTorch |

---

## Installation

### Backend

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

That is enough to run everything. To enable the full AI stack (DINOv2,
SuperPoint, LightGlue):

```bash
# CPU-only build
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# or CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements-ai.txt
```

DINOv2 weights are downloaded on first use through `torch.hub` and cached in
your local torch hub directory. Nothing is trained.

### Frontend

```bash
cd frontend
npm install
```

### Environment

Copy `.env.example` to `.env` in the project root (or in `backend/`) and adjust
if needed. Every value has a sensible default, so this is optional.

---

## Running

Two terminals:

```bash
# Terminal 1 — API on http://localhost:8000
cd backend
uvicorn app.main:app --reload
```

```bash
# Terminal 2 — UI on http://localhost:5173
cd frontend
npm run dev
```

Open <http://localhost:5173>. The Vite dev server proxies `/api` and `/files`
to the backend, so no CORS configuration is needed in development.

Interactive API documentation is at <http://localhost:8000/docs>.

### Docker

```bash
docker compose up --build
```

Serves the API on port 8000 and the built frontend on port 5173.

---

## Generating test data

The repository ships a procedural generator so the prototype can be
demonstrated without sourcing real satellite imagery:

```bash
python test_data/generate_test_data.py --out test_data/generated
```

This writes a 3000×3000 synthetic aerial map (road grid, buildings, fields,
water), nine drone-capture variants covering the testing strategy below, a
valid QGroundControl `.plan`, and `ground_truth.json` with the true centre of
every crop.

---

## Testing strategy

Run the offline harness from the `backend` directory:

```bash
cd backend
python -m scripts.evaluate --data ../test_data/generated
```

| # | Test | Expected |
|---|------|----------|
| 1 | Direct 15% crop from the map | successful localization |
| 2 | Crop rotated 30° | successful localization |
| 3 | Crop rotated 90° | successful localization |
| 4 | Brightness / contrast change | successful localization |
| 5 | Mild Gaussian blur | successful localization |
| 6 | Significant downscale | successful localization |
| 7 | Perspective distortion | successful localization |
| 8 | Completely unrelated image | **`NO_MATCH`** |
| 9 | A different region of the map | successful localization |

Reference run on the synthetic dataset (CPU, classical retrieval + SIFT):

```
case                   status            conf   inl   err_px  result    time
test1_direct_crop.jpg  MATCH_FOUND       0.98   573      0.0    PASS    3.0s
test2_rotated_30.jpg   MATCH_FOUND       0.96   269      1.0    PASS    2.8s
test3_rotated_90.jpg   MATCH_FOUND       0.96   226      1.4    PASS    3.2s
test4_brightness.jpg   MATCH_FOUND       0.98   708      0.0    PASS    2.8s
test5_blur.jpg         MATCH_FOUND       0.96   370      0.0    PASS    2.6s
test6_resized.jpg      MATCH_FOUND       0.97   586      0.0    PASS    1.8s
test7_perspective.jpg  MATCH_FOUND       0.96   480     64.4    PASS    2.2s
test8_unrelated.jpg    NO_MATCH          0.35     8        -    PASS    3.4s
test9_other_area.jpg   MATCH_FOUND       0.95   574      0.0    PASS    1.6s

Top-1 accuracy      : 8/8
No-match detection  : 1/1
False localizations : 0
```

The same batch scoring is available interactively on the **Developer** page,
which reports Top-1 accuracy, Top-5 retrieval accuracy, no-match detection rate
and false-localization rate.

---

## How localization works

### 1. Multi-scale overlapping tiling

A drone frame typically covers 10–20% of the reference map. Rather than
squashing the whole map to the drone resolution, the map is cut into square
windows whose **area** is a fixed fraction of the map area — 8, 10, 12, 15, 18,
20 and 25% by default — with 25% overlap between neighbours. Each tile records
its map-space geometry so a homography solved in tile space can be lifted back
to absolute map pixels.

Tiles are embedded once when the map is uploaded and cached to
`backend/cache/{map_id}.npz`, so repeated localization requests against the
same map skip indexing entirely.

### 2. Candidate retrieval

The drone frame and every tile are reduced to a single global descriptor and
compared by cosine similarity. The top-K (default 5) tiles go forward.
**Retrieval never decides a position** — a modest similarity backed by 200
geometric inliers beats a high similarity with none.

### 3. Two branches, deliberately separate

The **matching branch** applies only illumination normalization (CLAHE on the
luminance channel). No edge maps, no morphology, no geometry changes — so
keypoint coordinates stay valid in the original frame.

The **visualization branch** produces the enhanced image, grayscale, Canny
edges, contour overlay and the Structural Terrain View. These exist purely to
explain the result to a human and are never substituted for the photographic
frame in the pipeline.

### 4. Local features and matching

SuperPoint keypoints and descriptors are matched with LightGlue against each
candidate tile. If LightGlue is unavailable, or returns too few
correspondences to be geometrically useful, the pipeline automatically retries
with SIFT + FLANN + Lowe's ratio test. `MATCHER=sift` forces the classical path
for debugging and side-by-side comparison.

### 5. Rotation handling

Nothing assumes the drone frame is north-up — feature matching is naturally
rotation tolerant. When the upright attempt *fails* verification, the pipeline
additionally evaluates the query at 90°, 180° and 270°. A strong upright match
is never disturbed by this search.

### 6. RANSAC, homography and verification

`cv2.findHomography` (MAGSAC, falling back to plain RANSAC) produces the
transform and inlier mask. The result is only accepted if it passes every
structural gate:

| Gate | Rejects |
|---|---|
| finite projection | numerically degenerate transforms |
| convex quadrilateral | frames folded inside out |
| scale ratio 0.04–25× | implausible area change |
| shear ≤ 0.7 | transforms no planar view could produce |
| opposite-edge ratio ≤ 4× | extreme perspective |
| ≥ 15 inliers, ≥ 18% inlier ratio | thin support |
| reprojection error ≤ 8 px | poor fits |
| ≥ 25% spatial coverage | inliers clustered on one building |

**Spatial coverage** divides the drone frame into a 4×4 grid and counts how
many cells contain an inlier. This is what stops a repetitive rooftop from
carrying an entire "match".

### 7. Position

The frame corners `(0,0) (w,0) (w,h) (0,h)` and the centre `(w/2, h/2)` are
projected through the homography, rebased from working resolution to full
resolution and translated by the tile origin, giving the polygon and centre in
absolute map pixels.

### 8. Whole-map fallback

A frame straddling four tiles can be missed by every window. When no candidate
passes verification, the pipeline makes one direct attempt against the whole
downscaled map — verified by exactly the same gates. Disable with
`GLOBAL_FALLBACK=false`.

---

## Confidence and the four verdicts

Confidence is computed from the metrics, never hard-coded. Five weighted
components sum to a genuine 0–1 score:

| Component | Weight | Source |
|---|---|---|
| Retrieval | 0.15 | global descriptor similarity |
| Inliers | 0.30 | absolute inlier count blended with inlier ratio |
| Geometry | 0.25 | reprojection error (exponential decay) + shear penalty |
| Coverage | 0.15 | spatial spread of inliers across the 4×4 grid |
| Ambiguity | 0.15 | margin over the runner-up candidate |

A candidate whose homography was rejected is capped at 0.35 and can never
present as a confident match.

| Status | Meaning |
|---|---|
| `MATCH_FOUND` | Verified geometry, confidence ≥ 0.60 |
| `LOW_CONFIDENCE` | Valid geometry but below the reporting threshold — *visual localization unreliable*, indicative only |
| `AMBIGUOUS` | Two independently valid candidates score within 6% **and** place the drone in different map regions |
| `NO_MATCH` | No candidate passed verification, or the best score is below 0.40 |

Because tiles overlap by design, the same physical location routinely appears
as two strong candidates. Ambiguity is only declared when the near-tied
candidates also *disagree about where the drone is*, by more than a third of
the projected frame diagonal.

---

## Georeferencing and GPS

GPS coordinates are never invented. `gps` is `null` until the operator supplies
a georeference for the reference map, either as a lat/lon bounding box
(north/south/west/east) or as four explicit corner coordinates, which are fitted
with a homography. Only then is the estimated map pixel converted to
latitude/longitude. The bounding-box form also reports an approximate ground
sample distance in metres per pixel.

---

## QGroundControl `.plan` files

Uploading a `.plan` is optional and never required for matching. The parser
extracts the planned home position, mission items (including nested
survey/corridor complex items), waypoint coordinates, altitudes, geofence and
rally-point counts, and displays them as mission metadata with a plotted
waypoint path.

> The `.plan` file contains mission coordinates. It does not contain satellite
> imagery.

The padded mission bounding box is offered as a *suggested* georeference, which
the operator can accept or ignore — the map image may cover a different area.

---

## API documentation

Swagger UI: <http://localhost:8000/docs> · OpenAPI schema: `/openapi.json`

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/map/upload` | Upload the reference map; tiling and embedding start in the background |
| `GET` | `/api/map/{map_id}` | Indexing status (`pending` → `indexing` → `ready`) |
| `POST` | `/api/drone/upload` | Upload the drone capture |
| `POST` | `/api/plan/upload` | Upload and parse a QGroundControl `.plan` |
| `POST` | `/api/localize` | Start a localization job (202 + `job_id`) |
| `GET` | `/api/process/{job_id}` | Live pipeline stage progress |
| `GET` | `/api/result/{job_id}` | Final structured result |
| `GET` | `/api/candidates/{job_id}` | Ranked candidate diagnostics |
| `POST` | `/api/georeference` | Attach a lat/lon extent to a map |
| `DELETE` | `/api/georeference/{map_id}` | Remove it |
| `GET` | `/api/system/info` | Device, model availability, active thresholds |
| `GET` | `/api/health` | Liveness probe |
| `POST` | `/api/dev/batch` | Developer Mode batch scoring |
| `GET` | `/files/processed/{job_id}/...` | Generated stage renders |
| `GET` | `/files/uploads/...` | Original uploads |

### Upload map response

```json
{
  "status": "success",
  "map_id": "map_b725721b05d3",
  "width": 3000,
  "height": 3000,
  "tiles_generated": 100,
  "embedding_status": "ready"
}
```

### Localization response (abridged)

```json
{
  "status": "MATCH_FOUND",
  "confidence": 0.9638,
  "best_candidate": { "tile_id": 78, "dino_similarity": 0.9427,
                      "rotation_applied_deg": 0 },
  "feature_metrics": {
    "raw_matches": 430,
    "ransac_inliers": 267,
    "inlier_ratio": 0.6209,
    "spatial_coverage": 0.6875,
    "reprojection_error": 0.449,
    "homography_valid": true
  },
  "map_pixel": { "x": 1860, "y": 1229 },
  "polygon": [[1709, 668], [2421, 1079], [2010, 1789], [1299, 1379]],
  "gps": { "latitude": 12.9686253, "longitude": 77.6003984 },
  "candidates": [ /* full per-candidate diagnostics */ ],
  "renders": { "structural_map": "job_.../structural_map.png", "...": "..." }
}
```

Each job also writes its stage renders to `backend/processed/{job_id}/`:
`original`, `corrected`, `enhanced`, `grayscale`, `edges`, `structural_map`,
`contours`, `keypoints`, `candidate_1..N`, `candidate_overview`, `matches_raw`,
`matches_inliers`, `result_map`, `localized_area`.

---

## Configuration

All settings are environment variables with defaults — see `.env.example`.

| Variable | Default | Effect |
|---|---|---|
| `BACKEND_PORT` | `8000` | API port |
| `FRONTEND_ORIGIN` | `http://localhost:5173` | Added to the CORS allow-list |
| `MODEL_DEVICE` | `auto` | `auto` / `cuda` / `mps` / `cpu` |
| `APP_MODE` | `real` | `demo` tags every result so its metrics can never be shown as a real fix |
| `MAX_MAP_SIZE` | `8000` | Reference maps are downscaled to this long edge |
| `TOP_K_CANDIDATES` | `5` | Candidates carried into verification |
| `MAX_KEYPOINTS` | `2048` | Per-image feature budget |
| `MATCHER` | `lightglue` | `lightglue` or `sift` |
| `TILE_SCALES` | `0.08…0.25` | Tile areas as a fraction of map area |
| `TILE_OVERLAP` | `0.25` | Overlap between neighbouring tiles |
| `MAX_TILES` | `600` | Tile budget; the grid is evenly subsampled beyond it |
| `WORK_SIZE` | `640` | Long edge used for feature extraction |
| `RANSAC_THRESHOLD` | `5.0` | Reprojection threshold in pixels |
| `MIN_INLIERS` | `15` | Hard acceptance floor |
| `MIN_INLIER_RATIO` | `0.18` | Hard acceptance floor |
| `MAX_REPROJECTION_ERROR` | `8.0` | Hard acceptance ceiling |
| `MIN_SPATIAL_COVERAGE` | `0.25` | Hard acceptance floor |
| `MATCH_CONFIDENCE` | `0.60` | `MATCH_FOUND` threshold |
| `LOW_CONFIDENCE` | `0.40` | Below this becomes `NO_MATCH` |
| `AMBIGUITY_GAP` | `0.06` | Margin below which two candidates tie |
| `ROTATION_SEARCH` | `true` | Try 90/180/270° when upright fails |
| `GLOBAL_FALLBACK` | `true` | Whole-map attempt when all tiles fail |

Frontend: `VITE_API_BASE` (leave unset to use the dev proxy).

---

## Folder structure

```
innovx-visualnav/
├── frontend/
│   ├── src/
│   │   ├── components/      Navbar, UploadCard, MapViewer, CandidateCard, ...
│   │   ├── pages/           Dashboard, Processing, MatchAnalysis, Developer, About
│   │   ├── services/        api.js
│   │   ├── hooks/           useAppState.jsx
│   │   └── utils/           format.js
│   ├── tailwind.config.js
│   └── package.json
│
├── backend/
│   ├── app/
│   │   ├── main.py          FastAPI app, CORS, static mounts, error handlers
│   │   ├── config.py        every tunable, no hard-coded paths
│   │   ├── store.py         in-memory registry + embedding cache
│   │   ├── services.py      uploads and background job execution
│   │   ├── schemas.py       typed request/response models
│   │   ├── api/             routes_upload / routes_localize / routes_system
│   │   ├── localization/
│   │   │   ├── imaging.py         I/O and resize helpers
│   │   │   ├── preprocessing.py   both branches + Structural Terrain View
│   │   │   ├── tiling.py          multi-scale overlapping windows
│   │   │   ├── dino.py            global descriptors + retrieval
│   │   │   ├── superpoint.py      local feature extraction
│   │   │   ├── lightglue.py       matching + matcher dispatch
│   │   │   ├── sift.py            SIFT + FLANN + Lowe ratio
│   │   │   ├── homography.py      RANSAC, verification, projection
│   │   │   ├── confidence.py      scoring and the four verdicts
│   │   │   ├── geolocation.py     optional pixel → lat/lon
│   │   │   ├── visualization.py   every explainability render
│   │   │   └── pipeline.py        orchestration
│   │   ├── plan/qgc_parser.py
│   │   └── models/loader.py       lazy model loading, device selection
│   ├── scripts/evaluate.py        offline accuracy harness
│   ├── cache/  uploads/  processed/
│   └── requirements.txt · requirements-ai.txt
│
├── test_data/generate_test_data.py
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Presentation flow

1. Upload the reference satellite map — watch tiling and embedding complete.
2. Upload the drone top-view image.
3. Click **Locate Drone** and follow the live pipeline stages.
4. Open **Processing** and step through all nine stages.
5. Show the **Enhanced** image and the before/after toggle.
6. Show the **Structural Terrain View** (with its disclaimer).
7. Show the **AI keypoints**.
8. Show the **Top candidate regions** and click through to their map locations.
9. Show the **correspondence lines**, then toggle to RANSAC inliers only.
10. Show the **geometric verification** metrics and the coverage grid.
11. Show the **final polygon** on the full map.
12. Show the **estimated drone centre** marker.
13. Show the **confidence breakdown** on Match Analysis — why #1 beat #2.
14. Add a georeference to reveal the **GPS coordinate**; then upload the
    unrelated test image to show the system answering **`NO_MATCH`**.

---

## Known limitations

- A homography assumes a locally planar scene. Tall buildings and steep terrain
  introduce parallax the model cannot represent, which shows up as elevated
  reprojection error.
- Large appearance gaps between the reference map and the live frame — season,
  time of day, construction, snow — reduce inlier counts and can push a true
  match into `LOW_CONFIDENCE`.
- Genuinely repetitive layouts can produce real ambiguity. The system reports
  `AMBIGUOUS` rather than guessing, which is the intended behaviour.
- The Structural Terrain View is derived from a single RGB frame. It is not an
  elevation or 3D terrain model.
- Position accuracy is bounded by reference map resolution and by the accuracy
  of the operator-supplied georeference.
- The registry is in-process: uploads and jobs are lost when the backend
  restarts. Embedding caches on disk survive, but a production deployment would
  need a real store.
- Runtime is a few seconds per frame on CPU. The prototype prioritises
  correctness over real-time performance.

---

innovX VisualNav reports an **estimated visual position** with a stated
**localization confidence**. It does not guarantee a drone position, and it
does not return a fabricated GPS coordinate.
