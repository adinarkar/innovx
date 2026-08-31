# Sat2Map training pipeline

Offline training for the **aerial/satellite → map-style** translation model used
as an *auxiliary* representation by the InnovX localization pipeline. This code
is completely separate from the FastAPI server: nothing here runs at application
startup, and no dataset or checkpoint is committed to Git.

```
backend/
  training/sat2map/
    prepare_dataset.py   raw Sat2Maps tiles  -> train/val satellite|map pairs
    dataset.py           paired loader + pair-consistent augmentation
    model.py             UNetTranslator (from app.localization._sat2map_net) + L1/SSIM/edge loss
    train.py             training loop: seed, CPU/GPU, checkpoints, resume, best-ckpt
    evaluate.py          MAE / SSIM / edge-IoU / structure-IoU + qualitative triptychs
    export.py            strip optimizer state; optional TorchScript
  weights/               checkpoints land here (git-ignored)
```

## 1. Data

Uses the data format of
[`taesungp/larger-google-sat2maps-dataset`](https://github.com/taesungp/larger-google-sat2maps-dataset):
~100k tiles, each a `satellite | google-roadmap` image concatenated
side-by-side for the same geographic area.

> **This is not a drone-to-map dataset.** A model trained only on Sat2Maps is
> **not** a production-ready drone localization model. Use it for (1) pretraining
> aerial→map representation translation, (2) learning cross-domain features,
> (3) prototype visualization, (4) preparing the architecture for a later
> fine-tune on real InnovX drone imagery.

Download the tiles yourself (this repo never downloads them at app startup), then:

```bash
cd backend
python -m training.sat2map.prepare_dataset \
    --src ./raw/sat2maps --out ./datasets/sat2maps --val-split 0.05 --size 256
```

### If the UC Berkeley host is unreachable

`http://efrosgans.eecs.berkeley.edu/datasets/larger_sat2maps_cleaned.tar` is
frequently offline. `fetch_hf_maps.py` pulls the equivalent
satellite&harr;Google-roadmap pairing from the Hugging Face Hub
(`huggan/maps` &mdash; the original pix2pix "maps" set, 1096 train / 1098 val,
600&times;600) into the `--already-split` layout:

```bash
pip install datasets
python -m training.sat2map.fetch_hf_maps --repo huggan/maps --out ./raw/sat2maps
python -m training.sat2map.prepare_dataset \
    --src ./raw/sat2maps --out ./datasets/sat2maps --already-split --size 256
```

This is a smaller, prototype-grade substitute &mdash; treat any model trained on
it as a wiring demo, not a production translator.

Output layout (exact pair correspondence preserved):

```
datasets/sat2maps/
  train/satellite/000001.png   train/map/000001.png
  val/satellite/000001.png     val/map/000001.png
```

`prepare_dataset.py` provides resizing, corruption checks and the train/val
split. Augmentation happens at load time in `dataset.py` and is applied
**identically to both images of a pair** (90/180/270 rotation, flips, small
arbitrary rotation) so input and target never stop describing the same
coordinates. Photometric augmentation (brightness/contrast/noise/blur/JPEG) is
applied to the **aerial image only**.

## 2. Train

```bash
python -m training.sat2map.train \
    --dataset ./datasets/sat2maps \
    --epochs 50 --batch-size 8 --output ./weights/sat2map
```

Deterministic seed, automatic CPU/GPU/MPS detection, `sat2map_last.pt` every
epoch, `sat2map_best.pt` on val-loss improvement, `--resume`, `config.json` and
`history.json` dumps. Objective is L1 + SSIM + edge-consistency — no GAN in the
default path (geometric consistency matters more than photorealism).

## 3. Evaluate

```bash
python -m training.sat2map.evaluate \
    --dataset ./datasets/sat2maps \
    --checkpoint ./weights/sat2map/sat2map_best.pt \
    --out ./weights/sat2map/eval
```

### Training report (optional)

`report.py` bakes `history.json`, `config.json`, the run's preview PNGs and the
`eval/` triptychs into one self-contained `report.html` (loss curves + tables +
images, no server, opens from `file://`):

```bash
python -m training.sat2map.report --run ./weights/sat2map
# -> ./weights/sat2map/report.html
```

It is a training artefact, not part of the app, and is git-ignored with the
rest of `weights/`.

## 4. Export & wire into inference

```bash
python -m training.sat2map.export \
    --checkpoint ./weights/sat2map/sat2map_best.pt \
    --out ./weights/sat2map_best.pt
```

Then set in `.env`:

```
SAT2MAP_ENABLED=true
SAT2MAP_MODEL_PATH=weights/sat2map_best.pt
SAT2MAP_DEVICE=auto
```

The engine loads the checkpoint once and caches it; it never reloads per
request. If the file is missing or `SAT2MAP_ENABLED=false`, the pipeline logs
`Sat2Map translation unavailable - using standard localization pipeline.` and
continues.

## Future: InnovX drone fine-tune

The same code trains a `drone camera → map` model once paired data exists:

```
datasets/drone/
  drone_train/aerial/000001.png   drone_train/map/000001.png
```

```bash
python -m training.sat2map.train --dataset ./datasets/drone \
    --aerial-dir aerial --resume ./weights/sat2map/sat2map_best.pt \
    --output ./weights/drone
```

| Stage | Direction | Data |
|---|---|---|
| Sat2Maps pretrain | satellite → map | Google sat/roadmap pairs |
| InnovX fine-tune | drone camera → map | real drone frames + reference-map crops |
