# ARC-AGI-2 Solver

An open-source three-stage pipeline for solving ARC-AGI-2 tasks, submitted to the [ARC Prize 2026](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2) competition on Kaggle.

---

## Architecture

```
Input Task
    │
    ▼
Stage 1: Deterministic Rule Bank
    │  Fast symbolic solvers covering ~35% of tasks.
    │  Detects structural patterns (connected components,
    │  flood-fill, symmetry, projection, colour remapping).
    │
    ▼  (if unsolved)
Stage 2: Pattern Induction Engine
    │  21 learned-pattern solvers operating on grid abstractions.
    │  Covers noise filtering, bilateral/quadrant symmetry,
    │  spatial scaling, gravitational projection, path tracing,
    │  template inference, majority voting, and more.
    │
    ▼  (if unsolved)
Stage 3: LLM Inference + Test-Time Training
       Qwen3-4B (grid-SFT checkpoint) loaded from Kaggle model hub.
       Per-task gradient update (20 steps) on training pairs with
       colour-permutation and flip augmentations before inference.
       Returns attempt_1 and attempt_2 per test output.
```

---

## Repository Layout

```
arc-agi2-solver/
├── src/
│   ├── grid_utils.py          # Core grid primitives (BFS, CC, transforms, symmetry)
│   ├── pattern_engine.py      # 21 pattern solvers + ensemble dispatcher
│   ├── dsa_solver.py          # Wrapper around the deterministic rule bank
│   ├── dsa_prepass_src.py     # Full deterministic solver bank (~7000 lines)
│   ├── llm_solver.py          # LLM inference: prompt builder + ArcLLMSolver
│   └── pipeline.py            # Three-stage master pipeline + CLI
├── training/
│   └── finetune_qwen.py       # LoRA/PEFT fine-tuning script for Qwen
├── notebooks/
│   └── arc_agi2_submission.ipynb  # Self-contained Kaggle submission notebook
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Solvers

### Stage 1 — Deterministic Rule Bank
Covers tasks that can be solved with exact rules:
- Connected-component labelling and object isolation
- Flood-fill and enclosed-region detection
- Rigid transformations (rotation, reflection, transpose)
- Majority-vote noise filtering
- Object counting and spatial projection

### Stage 2 — Pattern Induction Engine (21 solvers)

| Solver class | Task family it targets |
|---|---|
| `NeighborhoodRuleLearner` | Local cellular-automaton rules |
| `BilateralSymmetrySolver` | Horizontal / vertical mirror completion |
| `QuadrantSymmetrySolver` | Four-fold rotational symmetry |
| `TranspositionSymmetrySolver` | Diagonal / transpose symmetry |
| `RegionExtractionSolver` | Bounded sub-grid extraction |
| `FloodFillEnclosureSolver` | Interior region flooding |
| `ObjectProjectionSolver` | Directional object casting |
| `LocalContextExpansionSolver` | Small-pattern tiling and expansion |
| `PaletteRemapSolver` | Colour permutation / remapping |
| `SpatialScalingSolver` | Zoom / scale factor inference |
| `GravitationalFallSolver` | Directional gravity simulation |
| `ConnectedPathSolver` | Shortest-path / connectivity tracing |
| `ObjectCountSolver` | Count-driven output generation |
| `TemplateInferenceSolver` | Template matching and completion |
| `SpatialDilationSolver` | Morphological dilation / erosion |
| `MajorityVoteFilterSolver` | Majority-consensus filtering |
| `ConditionalReplacementSolver` | Rule-based conditional colour swap |
| `PerimeterOutlineSolver` | Border / outline extraction |
| `RecursiveStructureSolver` | Recursive / fractal pattern detection |
| `SpatialRelationSolver` | Relative-position reasoning |
| `CompositePipelineSolver` | Multi-step composite solver |

### Stage 3 — LLM Inference
- Base model: [sorokin/qwen3_4b_grids15_sft139](https://www.kaggle.com/models/sorokin/qwen3_4b_grids15_sft139)
- Test-time training: 20 gradient steps per task on train pairs before inference
- Augmentations: colour permutation + horizontal/vertical flip
- Generates `attempt_1` (greedy) and `attempt_2` (temperature-sampled)

---

## Running Locally

```bash
pip install -r requirements.txt

# Run the full pipeline on a local challenge file
python src/pipeline.py \
    --challenge_file data/arc-agi-2-challenges.json \
    --model_path /path/to/qwen3_4b_grids15_sft139 \
    --output_file submission.json
```

---

## Kaggle Submission

The notebook `notebooks/arc_agi2_submission.ipynb` is self-contained and runs end-to-end on Kaggle:

1. Reads `/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi-2-challenges.json`
2. Loads the model from `/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1`
3. Runs all three stages
4. Writes `submission.json` to `/kaggle/working/`

**Runtime:** GPU P100 / T4, ≤12 h. Internet disabled.

---

## Fine-tuning

```bash
python training/finetune_qwen.py \
    --model_path /path/to/qwen3_4b \
    --data_path data/arc2_finetune_500.jsonl \
    --output_dir models/arc_qwen_finetuned \
    --epochs 3
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Citation

If you find this repository useful, please star it and cite:

```
@misc{arc_agi2_solver_2026,
  title  = {ARC-AGI-2 Three-Stage Solver},
  author = {joyjeni},
  year   = {2026},
  url    = {https://github.com/joyjeni/arc-agi2-solver}
}
```
