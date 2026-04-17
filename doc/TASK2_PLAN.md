# Task II Plan — Unsupervised Domain Adaptation for Classification

**Status:** planning
**Created:** 2026-04-15
**Depends on:** Task I checkpoints (see PROGRESS.md)

---

## 1. Problem statement

Train a classifier on a labeled **source** domain, evaluate it on an unlabeled **target** domain. Compare several UDA methods against source-only and oracle baselines, across 5 dataset pairs.

Metric: top-1 accuracy on the target test set.

---

## 2. Datasets and classifier architectures

| # | Source → Target | Resolution | Classes | # train (S/T) | Classifier |
|---|---|---|---|---|---|
| 1 | MNIST → USPS | 32×32 gray→gray (3ch) | 10 | 60000 / 7291 | Small CNN |
| 2 | SVHN → MNIST | 32×32 rgb→gray(3ch) | 10 | 73257 / 60000 | Small CNN |
| 3 | Amazon → Webcam (Office-31) | 64×64 | 31 | 2817 / 795 | ResNet-18 (ImageNet-pretrained) |
| 4 | Art → Real-World (Office-Home) | 64×64 | 65 | 2427 / 4357 | ResNet-18 (ImageNet-pretrained) |
| 5 | Photo → Sketch (PACS) | 64×64 | 7 | 1670 / 3929 | ResNet-18 (ImageNet-pretrained) |

**Architectures:**
- **Small CNN (32×32):** 3 conv blocks (32→64→128 ch, BN+ReLU, stride-2) → global avgpool → FC(num_classes). ~200K params. Trained from scratch, 20 epochs, Adam 1e-3.
- **ResNet-18 (64×64):** `torchvision.models.resnet18(weights=IMAGENET1K_V1)`, replace final FC with `Linear(512, num_classes)`. Fine-tune end-to-end. 30 epochs, Adam 1e-4, cosine decay. Input resized to 64×64 with ImageNet normalization.

Rationale for pretraining on 64×64 pairs: tiny source datasets (Office-Home Art has 2427 images across 65 classes ≈ 37/class). From-scratch guarantees overfitting; pretrained gives a meaningful baseline and is standard UDA practice.

---

## 3. Methods (rows of the final table)

### M0 — Source-only (lower bound)
Train classifier on raw source. Test on target. No adaptation.

### M1 — Target-only oracle (upper bound; optional sanity check)
Train classifier on target train split with target labels. Test on target test split. Same architecture and training recipe as M0. Included only if it fits in time.

### M2 — CycleGAN-pixel adapted
1. Use Task I pixel CycleGAN `G_AB` to translate every source image → target-style.
2. Train classifier on **(translated_source, source_labels)**.
3. Test on real target.

### M3 — CycleGAN-spectral adapted
Same as M2 but using the spectral CycleGAN variant (recombined output: `G(low_src) + high_src`).

### M4 — FDA (Fourier Domain Adaptation)
Two flavors implemented:
- **M4a FDA-at-test (simple)**: at inference, swap the low-frequency FFT amplitude of the test image with that of a random source image, then classify. No retraining — reuses M0 classifier.
- **M4b FDA-at-training (augmentation)**: train a classifier where each source image's low-frequency amplitude is swapped with a random target image's amplitude (the paper's formulation). Classifier sees source content but target low-freq statistics. Test on real target.

Uses our existing `utils/spectral.py` FFT helpers.

### M5 — CyCADA-lite
Full CyCADA jointly trains CycleGAN + classifier with a semantic consistency loss `||f(x) - f(G(x))||` that forces translation to preserve classifier predictions. Full retrain is expensive.

**Lite version:**
1. Start from the Task I pixel CycleGAN checkpoint.
2. Short (5-epoch) fine-tune of `G_AB` with a new loss term: `L_sem = CE(f_M2(G_AB(x_src)), y_src)` where `f_M2` is the frozen M2 classifier.
3. Retranslate source with fine-tuned `G_AB`, retrain classifier on the new translations.
4. Test on real target.

Honest label: **"CyCADA-lite (reusing pretrained CycleGAN)"** in the report — it's a simplified, budget variant of the paper.

---

## 4. File layout (to be created)

```
UDA_transfer/
├── task2/
│   ├── classifier.py          # SmallCNN + get_resnet18 factory
│   ├── train_classifier.py    # Generic trainer; args: --data_dir --arch --epochs --out
│   ├── translate_bulk.py      # Run a CycleGAN ckpt over a whole source set, dump PNGs
│   ├── fda.py                 # FDA amplitude swap (at-test and at-train variants)
│   ├── cycada_lite.py         # 5-epoch semantic-consistency fine-tune of G_AB
│   ├── eval.py                # Load classifier, evaluate on target test split
│   └── run_all.py             # Orchestrates all methods × all datasets, writes table
├── data_translated/
│   └── <exp_name>/            # Bulk-translated source images per checkpoint
├── checkpoints_task2/
│   └── <method>_<dataset>/classifier.pth
├── results/task2/
│   ├── accuracy_table.md      # Markdown table for the report
│   └── accuracy_table.csv     # Raw numbers
└── scripts/
    ├── run_task2_classifiers.sh  # PBS launcher for bulk classifier training
    └── run_task2_translate.sh    # PBS launcher for bulk translation
```

---

## 5. Implementation order

Each step is independent enough to verify before moving on.

### Phase A — Baseline (M0)
1. Write `classifier.py` with `SmallCNN` and `get_resnet18` factory.
2. Write `train_classifier.py` (dataset loader reuses `data/datasets.py`).
3. Write `eval.py`.
4. Submit 5 source-only classifier trainings in parallel. Record M0 row of the table.
5. Sanity check: digit accuracies should be in the literature range (MNIST→USPS source-only ~75-80%, SVHN→MNIST ~60-65%).

**Estimated wall time:** ~30 min for 5 parallel g1 jobs.

### Phase B — Bulk translation + M2 pixel
1. Write `translate_bulk.py`: loads a checkpoint, runs over source train split, writes translated images to `data_translated/<exp_name>/`.
2. Submit 5 bulk translations (pixel, one per dataset).
3. Train 5 classifiers on translated data (reuses `train_classifier.py` with `--data_dir data_translated/...`).
4. Evaluate → M2 row.

**Estimated wall time:** ~10 min translation + 30 min classifier training.

### Phase C — M3 spectral
Same as Phase B but with spectral checkpoints and the recombined inference path.

**Estimated wall time:** ~40 min.

### Phase D — M4 FDA
1. Write `fda.py` — one function for at-test swap, one for at-train augmentation.
2. M4a: reuse M0 classifiers, evaluate with FDA preprocessing on test set. Fast.
3. M4b: retrain classifiers with FDA augmentation. 5 more classifier trainings.
4. Record M4a and M4b rows.

**Estimated wall time:** ~30 min.

### Phase E — M5 CyCADA-lite
1. Write `cycada_lite.py`: load pixel CycleGAN + M2 classifier, add semantic loss, fine-tune 5 epochs.
2. Retranslate source with fine-tuned G.
3. Retrain classifier on new translations.
4. Record M5 row.

**Estimated wall time:** ~60 min.

### Phase F — Reporting
1. Assemble `accuracy_table.md` and `.csv`.
2. Pick 2-3 datasets with the most interesting patterns, generate qualitative translation comparisons.
3. Write report section: what worked, what didn't, why.

**Estimated wall time:** ~1 hr.

---

## 6. Non-goals (explicitly excluded)

- **No full CyCADA retrain** (uses too much budget; CyCADA-lite is acceptable and labeled as such).
- **No self-training / pseudo-label refinement** methods (DANN, MCD, SHOT, etc.) — not on the required method list.
- **No ImageNet pretraining for 32×32 digit classifiers** — standard practice is from-scratch.
- **No data augmentation sweep** — use a single reasonable augmentation recipe across all methods for fair comparison.

---

## 7. Open questions answered (defaults)

| Question | Decision | Rationale |
|---|---|---|
| Pretrained ResNet-18 for 64×64? | **Yes** | Tiny source sets (Office-Home Art: 2427 imgs / 65 classes) will overfit from scratch. Pretrained is standard UDA practice. |
| Full CyCADA or lite? | **Lite** | Full is multi-hour joint retrain. Lite reuses Task I checkpoints, 5-epoch fine-tune. Labeled honestly in report. |
| FDA at-test, at-train, or both? | **Both** | At-test is ~20 lines with no retrain, at-train is the paper's proper version. Both are cheap. |
| Classifier input size for 64×64 runs | **Keep 64×64** | Matches the CycleGAN output; upsampling would throw away the resolution honesty of the comparison. |
| Class-balancing / sampler | **None** (class-balanced sampler optional if Office-Home class imbalance skews results) | Keep default for apples-to-apples; revisit only if needed. |

---

## 8. Success criteria

1. All 5 methods × 5 datasets = **25 accuracy numbers** in the final table (plus optional oracle row = 30).
2. Pattern should roughly follow literature: M2/M3 > M0 on most pairs; M4/M5 match or beat M2/M3 on some.
3. Expected direction (not hard rule): pixel CycleGAN adaptation generally helps digits more than Office-Home (smaller domain gap ⇒ less room for translation-based gains).
4. Qualitative finding: spectral helps on color-dominated shifts, pixel helps on texture-dominated shifts.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Bulk translation disk use (hundreds of MB per dataset) | Save as JPEG q90, not PNG. Or regenerate on-the-fly in the dataloader. |
| Classifier hyperparams tuned per-dataset balloon time | One fixed recipe per architecture (CNN / ResNet). No tuning. |
| M5 CyCADA-lite fails to improve over M2 | Acceptable — report the negative result. Full CyCADA was explicitly excluded. |
| Office-Home class imbalance inflates source-only accuracy | Report both accuracy and macro-F1 if imbalance > 3:1. |
| 64×64 ResNet-18 overfits source in <30 epochs | Early stopping on source val split; no target info used. |
