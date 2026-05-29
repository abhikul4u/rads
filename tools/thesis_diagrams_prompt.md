# RADS Layer 3 Thesis Diagrams — Canonical AI Prompt

> **Purpose**: This prompt is the single source of truth for any AI tool (Claude,
> GPT-4, Gemini, etc.) that needs to generate diagrams for the RADS Layer 3
> thesis. Copy the relevant section to the AI, specify which diagram you want,
> and provide the requested output format.

---

## Project Context (always include this)

You are helping me prepare diagrams for a M.Tech thesis chapter on
**Road Anomaly Detection System (RADS) Layer 3** — a YOLOv8-based detection
pipeline with three architectural enhancements, trained on a 3-class road
anomaly dataset.

### Background — the system in 2 sentences
RADS detects three classes of road anomalies from RGB images: manholes (MH),
potholes (PH), and waterlogged potholes (WLPH). Layer 3 is the model-development
phase: an enhanced YOLOv8 teacher trained on the 3,328-image dataset, knowledge-
distilled to a YOLOv8n student, then INT8-quantized for Android deployment.

### Hard facts (do not invent details beyond these)

**Dataset**: 3,328 images, split 70/20/10 (train/valid/test = 2330/666/332).
Three classes in this YAML order: `[MH, PH, WLPH]`. Mild class imbalance
(2.5× ratio). Median pothole occupies 2% of image area; manholes ~20%. Source:
Roboflow workspace `road-anomalies`, project `road_anamolies_yolov8-phref`, v6.

**Baseline**: YOLOv8l scale 'l' (43.6M params), input 768×768, batch 32, AdamW
optimizer with lr0=5e-4, weight_decay=5e-4, cosine LR schedule, 100 epochs,
patience=25, COCO-pretrained weights from Ultralytics yolov8l.pt.

**Architecture enhancement 1 — CBAM** (Woo et al. ECCV 2018):
- Inserted at the 3 PANet neck merge points (after each C2f in the upsample path)
- Channel attention (reduction ratio 16) followed by spatial attention (7×7 conv)
- ~0.2M additional parameters
- Implemented as a custom Ultralytics module via source-rewriting parser patch

**Architecture enhancement 2 — P2 detection head**:
- A 4th detection head added at stride 4 (192×192 grid at 768px input)
- Existing heads: P3 (stride 8), P4 (stride 16), P5 (stride 32)
- Requires an extra upsample-concat-C2f block in the neck to produce P2 features
- ~1.2M additional parameters
- Goal: improve small-object detection (small distant potholes)

**Architecture enhancement 3 — Size-aware regression loss**:
- Multiplies the standard YOLO box+DFL loss by inverse-sqrt-area weight
- Formula: `weight_i = clamp( 1 / sqrt(area_normalized_i + eps), max=4.0 )`
  where area_normalized = (w * h) and eps = 1e-3
- alpha mixing factor = 0.5 (loss = (1-alpha)·standard_loss + alpha·weighted_loss)
- Zero added parameters
- Installed at training time by monkey-patching Ultralytics' BboxLoss class

**Knowledge distillation**:
- Teacher: combined variant (CBAM + P2 + size-aware), 43.0M params, frozen
- Student: YOLOv8n scale 'n', 3.0M params, trainable
- Composite loss with three terms:
  - 0.4 × task loss (standard YOLO box+cls+dfl on student's own predictions vs ground truth)
  - 0.4 × KL divergence on classification logits at temperature T=4
  - 0.2 × MSE on intermediate features at the P5 (deepest) detection head
- Hooks register forward outputs on the teacher's and student's Detect modules
- Custom DistillationTrainer subclasses Ultralytics' DetectionTrainer

**Quantization & export**:
- Post-training quantization (PTQ) on the distilled student
- 200-image calibration set sampled from training split
- Outputs: FP32 ONNX (cross-platform), INT8 ONNX (optimized), INT8 TFLite (Android)

**Evaluation**:
- 3 seeds per variant: 42, 1337, 2024
- 6 variants: baseline, cbam, p2, sizeaware, combined, distill
- Metrics: mAP@0.5, mAP@0.5:0.95, per-class AP50 (MH, PH, WLPH), Precision, Recall
- 18 training runs total → mean ± std reported per variant
- FPS measured on A100 (server) and Snapdragon 7xx (mobile target)

### Conventions to follow
- Always use the class order `[MH, PH, WLPH]` exactly
- Layer numbering follows Ultralytics YAML convention (0-indexed)
- Stride values: P2=4, P3=8, P4=16, P5=32
- Tensor shapes: `(batch, channels, height, width)` — PyTorch convention
- Refer to enhancements by these names exactly: CBAM, P2 head, size-aware loss

---

## Output format instructions (specify when asking)

Add one of these to your request to control the output:

```
# For Mermaid (renders in GitHub, easy to version control):
"Return as Mermaid code in a ```mermaid block. Use flowchart LR or TB as
appropriate. Use clear node labels with line breaks via <br/>. Apply colors
to highlight modified parts: red (#ffcccc) for CBAM, green (#ccffcc) for
P2 head, yellow (#fff3cd) for size-aware loss."

# For PlantUML (sequence/state):
"Return as PlantUML code in a ```plantuml block. Start with @startuml,
end with @enduml. Use clear participant names. Add notes for important
behaviors."

# For draw.io / diagrams.net:
"Return as a draw.io-compatible XML document (mxGraph format). Provide
node positions explicitly so the layout doesn't break on import."

# For SVG:
"Return inline SVG. Use viewBox='0 0 1200 800'. Use Arial sans-serif.
Apply a consistent color palette: blues for backbone, oranges for neck,
greens for heads, reds for CBAM, yellows for losses."

# For plain ASCII (for quick previews):
"Return as ASCII art / text diagram suitable for monospace rendering.
Use Unicode box-drawing characters."
```

---

## Diagram catalog — request one or more

Each diagram below has a one-line description, then **what should be in it**.

### Diagram 1 — High-Level Algorithm Pipeline
**Purpose**: End-to-end view from input image to final predictions, showing
both training and inference paths.

**Required elements**:
- Input: RGB image (any resolution, gets resized to 768×768)
- Preprocessing block: resize + normalize + (during training only) augmentation
  pipeline (mosaic, hsv jitter, fliplr 0.5, scale 0.5, translate 0.1)
- Training branch:
  - Forward pass through enhanced YOLOv8 model
  - Loss computation (with size-aware mode if applicable)
  - Backpropagation with AdamW optimizer
  - Weight update with cosine LR schedule
  - (Loop back to forward, with early stop at patience=25)
- Inference branch:
  - Forward pass through trained model
  - Multi-head output decoding (P2/P3/P4/P5)
  - NMS at conf=0.25, IoU=0.7
  - Output: bounding boxes + class labels + confidence scores
- Show the branching clearly with a "train or inference?" diamond
- Highlight that the same model is used in both paths

### Diagram 2 — Detailed Network Architecture
**Purpose**: Layer-by-layer view of the model showing where CBAM and P2 head
modifications sit relative to baseline YOLOv8l.

**Required elements**:
- Three vertical sections: Backbone, Neck (PANet), Heads
- **Backbone** (top to bottom, in image stride order):
  - Conv 64ch s/2 → Conv 128ch s/2 → C2f 128ch (3 blocks)
  - Conv 256ch s/2 → C2f 256ch (6 blocks)
  - Conv 512ch s/2 → C2f 512ch (6 blocks)
  - Conv 512ch s/2 → C2f 512ch (3 blocks) → SPPF 512ch
- **Neck** (PANet structure with both top-down and bottom-up paths):
  - Top-down: Upsample → Concat with backbone P4 → C2f → [CBAM] → Upsample
    → Concat with backbone P3 → C2f → [CBAM]
  - In P2/combined variants: additional Upsample → Concat with backbone P2
    → C2f → [CBAM] (this is the new branch)
  - Bottom-up: Conv s/2 → Concat → C2f → Conv s/2 → Concat → C2f
- **Heads**:
  - In baseline/cbam/sizeaware: P3, P4, P5 heads only (3 outputs)
  - In p2/combined variants: P2, P3, P4, P5 heads (4 outputs)
- Visual coding:
  - **Red** for CBAM modules (only in cbam/combined variants)
  - **Green** for P2 head additions (only in p2/combined variants)
  - **Gray/black** for backbone (unchanged)
- Annotate parameter counts: baseline ~43.6M, with CBAM ~43.7M, with P2 ~42.8M,
  combined ~43.0M
- Show input shape entering: 3×768×768
- Show output shapes per head with their stride

### Diagram 3 — CBAM Internal Structure
**Purpose**: Show the two-stage attention mechanism inside a single CBAM module.

**Required elements**:
- Input: feature map of shape C × H × W
- **Channel attention sub-module**:
  - Two parallel paths from input: GlobalAvgPool and GlobalMaxPool (both → 1×1×C)
  - Both feed into a shared MLP: Linear(C → C/16) → ReLU → Linear(C/16 → C)
  - Outputs of both paths are summed
  - Sigmoid activation → channel attention map M_c (shape 1×1×C)
- Element-wise multiplication: input ⊗ M_c → F'
- **Spatial attention sub-module**:
  - Channel-wise pooling on F': AvgPool and MaxPool along C dimension
    (both → 1×H×W)
  - Concatenate the two → 2×H×W tensor
  - Conv 7×7 with padding 3 → 1×H×W
  - Sigmoid → spatial attention map M_s
- Element-wise multiplication: F' ⊗ M_s → output F''
- Output: refined feature map C × H × W (same shape as input)
- Annotate: reduction ratio r = 16, kernel size k = 7
- Show parameter count: roughly 2C(C/r) + 2(C/r) + 2k² ≈ a few thousand

### Diagram 4 — Knowledge Distillation Pipeline
**Purpose**: Show teacher-student training with the three-component composite loss.

**Required elements**:
- Single input: a training image with ground-truth bounding boxes
- Two parallel model paths:
  - **Teacher** (top): YOLOv8l with CBAM + P2 + size-aware (combined variant
    trained for 100 epochs), 43M params, all weights **FROZEN** (no_grad)
  - **Student** (bottom): YOLOv8n, 3M params, **TRAINABLE**
- Both produce two intermediate outputs:
  - **Feature maps** at the P5 (deepest) detection head
  - **Logits** per head (raw classification scores pre-softmax)
- Student additionally produces final predictions
- Three loss streams converging:
  - **Task loss** (weight 0.4): from student's final predictions vs ground truth.
    Standard YOLO box + cls + dfl losses combined.
  - **KL divergence loss** (weight 0.4): KL between teacher and student
    classification logits, computed at softmax temperature T=4
  - **Feature MSE loss** (weight 0.2): mean-squared-error between teacher and
    student P5 feature maps (after optional channel projection if dimensions differ)
- Composite loss: L = 0.4·L_task + 0.4·L_kl + 0.2·L_feat
- Backpropagation arrow: gradients flow only through student parameters (annotate
  this explicitly)
- Teacher is loaded from a frozen .pt checkpoint, never updated

### Diagram 5 — Training Sequence Diagram (single epoch)
**Purpose**: Sequence of operations within one training epoch.

**Required elements** (use sequence diagram convention with lifelines):
- Participants (left to right):
  - DataLoader, Optimizer, Model, LossFn, Scheduler, Logger, Validator
- Sequence:
  1. DataLoader → Model: batch (images, labels) of size 32
  2. Model → Model: forward pass through backbone, neck, heads
  3. Model → LossFn: predictions per head + ground truth
  4. LossFn → LossFn: compute box_loss, cls_loss, dfl_loss (with size-aware
     weighting if active)
  5. LossFn → Model: gradients via backward()
  6. Optimizer → Model: weight update (AdamW step)
  7. Scheduler → Optimizer: LR update (cosine schedule)
  8. (Repeat 1-7 for all batches in the epoch — show as a loop)
  9. End of epoch: Validator runs on valid set
  10. Validator → Logger: epoch metrics (mAP50, mAP50-95)
  11. Logger → checkpoint: save best.pt if val mAP improved
  12. Early-stop check: if no improvement for patience=25 epochs, signal STOP
- Show the per-batch loop vs the per-epoch validation clearly
- Include the W&B logging callback on every batch
- Annotate AMP (automatic mixed precision) where forward happens

### Diagram 6 — Loss Composition
**Purpose**: Visual breakdown of the composite losses used in training.

**Required elements**:
- Two main loss families shown side by side:
  - **Standard YOLOv8 detection loss** (used in baseline, cbam, p2):
    - Box loss: CIoU between predicted and target boxes, weight λ_box = 7.5
    - Classification loss: BCE on logits per class, weight λ_cls = 0.5
    - DFL loss: distribution focal loss on box-regression distributions,
      weight λ_dfl = 1.5
    - Total: L = λ_box · L_box + λ_cls · L_cls + λ_dfl · L_dfl
  - **Size-aware modification** (used in sizeaware, combined):
    - For each ground-truth box, compute area_normalized = w × h
    - Size weight: s_i = min( 1 / sqrt(area + eps), 4.0 )
    - Weighted box loss: L_box' = mean( s_i · CIoU(pred_i, gt_i) )
    - Same weighting applied to DFL loss
    - Final: L = (1 - α) · L_standard + α · L_weighted, with α = 0.5
- For distillation, show the third composite:
  - L_distill = 0.4 · L_task + 0.4 · L_kl + 0.2 · L_feat
  - Where L_kl = T² · KL(softmax(z_student/T), softmax(z_teacher/T)), T=4
  - And L_feat = MSE(f_student^P5, f_teacher^P5)
- Use equation rendering where possible (LaTeX or similar)
- Show the weights as labeled connectors so the reader can trace contributions

### Diagram 7 — Ablation Matrix / Experiment Design
**Purpose**: Visualize the experimental design — what was run, how many seeds,
what's measured.

**Required elements**:
- A 6 × 3 grid:
  - Rows: 6 variants (baseline, cbam, p2, sizeaware, combined, distill)
  - Columns: 3 seeds (42, 1337, 2024)
  - Each cell: one 100-epoch training run
- For each row, show:
  - Which enhancements are active (CBAM, P2, size-aware columns with checkmarks)
  - Total params
  - Whether it's a teacher or a student (distill row is the student)
- Below the grid, show:
  - Aggregation step: mean ± std across the 3 seeds
  - Result columns: mAP50, mAP50-95, AP_MH, AP_PH, AP_WLPH
- Highlight that distill row uses the trained combined_seed* checkpoints as
  teachers
- Show the post-training pipeline: distill teacher → student → quantize → export

### Diagram 8 — Quantization & Export Pipeline
**Purpose**: Show how the distilled student becomes deployment-ready.

**Required elements**:
- Input: distilled_student.pt (FP32, PyTorch, ~3M params, ~12MB)
- Step 1: Export to ONNX FP32 (~12MB)
- Step 2: Calibration data sampler
  - Pick 200 random training images
  - Resize, normalize to model's input format
  - Build a calibration TensorDataset
- Step 3: Static INT8 quantization via ONNX Runtime
  - Per-channel weight quantization
  - Per-tensor activation quantization with calibration stats
  - Output: distilled_student_int8.onnx (~3MB)
- Step 4: Convert to TFLite INT8
  - Use ultralytics' built-in export → goes through onnx2tf → tflite
  - Output: distilled_student_int8.tflite (~3MB)
- Final outputs ready for deployment:
  - FP32 ONNX for cross-platform GPU/CPU inference
  - INT8 ONNX for optimized edge inference (Hexagon NPU, etc.)
  - INT8 TFLite for Android (used in Layer 4)
- Show file sizes at each step to demonstrate the 4× reduction

---

## How to use this prompt

1. **Copy the "Project Context" section verbatim** — it's the same for every diagram
2. **Pick the diagram(s) you want** from the catalog
3. **Specify output format** from the "Output format instructions" section
4. **Optionally constrain style**: "use a 16:9 aspect ratio for a slide", "match
   the IEEE conference paper style", "use grayscale only for printing"
5. **Iterate**: if the first draft is wrong, point out specifics ("the CBAM
   modules should be in the neck, not the backbone") — the AI will correct

## Validation checklist (before accepting any AI output)

- [ ] Class order is exactly [MH, PH, WLPH] (not alphabetical [MH, PH, WLPH] —
      wait, those happen to be the same. The point is: use this order specifically)
- [ ] Stride values are correct: P2=4, P3=8, P4=16, P5=32
- [ ] CBAM is shown in the NECK, not backbone
- [ ] P2 head is shown as a 4th head, not replacing existing ones
- [ ] Size-aware loss applies to box+DFL, NOT to classification loss
- [ ] In distillation: teacher is FROZEN, student is trainable
- [ ] Distillation weights are 0.4 / 0.4 / 0.2 (not 0.5/0.3/0.2 or any other split)
- [ ] Temperature T = 4 for KL divergence
- [ ] Parameter counts match: baseline 43.6M, distill student 3.0M
- [ ] Image size is 768, not 640 (we explicitly use 768)
- [ ] Patience for early stopping is 25, not 50
- [ ] Three seeds shown: 42, 1337, 2024 (not arbitrary)

## What NOT to include in diagrams (common AI mistakes)

- Don't add dropout layers (YOLOv8 doesn't use them; we don't add any)
- Don't add layer norm (YOLOv8 uses batch norm)
- Don't add data augmentation steps to the inference path (only training)
- Don't show the NMS step in training (it's inference-only)
- Don't connect teacher and student weights directly (they're independent
  models; only outputs flow through losses)
- Don't include validation steps within an epoch (validation runs at epoch end)
- Don't claim more than 3 classes (it's strictly MH/PH/WLPH, no background class
  in our setup though YOLO internally has implicit background handling)

---

## Bonus — diagrams 9-10 (if you want extras)

### Diagram 9 — Dataset Distribution Charts
- Bar chart: instance counts per class per split (3 classes × 3 splits = 9 bars)
- Histogram: box area distribution per class
- Optional: aspect ratio distribution per class

### Diagram 10 — Early Stopping State Diagram
- States: Training (improving), Plateau (no improvement), Saved (best.pt updated),
  Stopped (patience exceeded)
- Transitions based on epoch outcomes
- Shows the patience=25 counter explicitly

---

*End of canonical prompt. Save this file as `thesis_diagrams_prompt.md` and
reuse it across AI tools.*
