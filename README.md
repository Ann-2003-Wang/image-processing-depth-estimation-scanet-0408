以下是整理好的 `README.md` 内容，可以直接复制使用：

```markdown
# CSC6051 Assignment 2 – Image Processing & Depth Estimation

**Author**: Wang Xuening (225040103)  
**Course**: CSC6051/MDS6004  

This repository contains the implementation and experiments for three tasks: Gaussian filtering, histogram equalization (global & local), and monocular depth estimation on ScanNet.

##  File Structure

```
.
├── task1.py                      # Gaussian filter from scratch
├── task2.py                      # Global & local histogram equalization
├── run.sh                        # Baseline training (ResNet50)
├── test_baseline.sh              # Evaluate baseline checkpoint
├── t3.3/                         # Data scale study configurations
│   ├── run_20k_16ep.sh           # 20k samples, 16 epochs (Config A)
│   ├── run_40k_8ep.sh            # 40k samples, 8 epochs  (Config B)
│   ├── run_80k_4ep.sh            # 80k samples, 4 epochs  (Config C)
│   └── run_160k_2ep.sh           # 160k samples, 2 epochs (Config D)
├── task3/t3.4.sh                 # Foundation model comparison (Depth Anything V2, ZoeDepth)
└── 225040103_WangXuening_Report2.pdf   # Full report
```

##  How to Run

### Task 1 & 2 (no dependencies needed)
```bash
python task1.py
python task2.py
```

### Task 3 – Baseline Training
```bash
bash run.sh
```

### Task 3 – Baseline Evaluation
```bash
bash test_baseline.sh
```

### Task 3 – Data Scale Study (Config A–D)

Each script trains a model with a specific sample/epoch combination:

| Script | Samples | Epochs | Config |
|--------|---------|--------|--------|
| `run_20k_16ep.sh` | 20,000 | 16 | A |
| `run_40k_8ep.sh`  | 40,000 | 8  | B |
| `run_80k_4ep.sh`  | 80,000 | 4  | C |
| `run_160k_2ep.sh` | 160,000| 2  | D |

Run any of them, for example:
```bash
bash t3.3/run_20k_16ep.sh
```

### Task 3 – Foundation Models
```bash
cd task3
conda activate d2oe          # if required
bash t3.4.sh
```

##  Results Summary

- **Baseline ResNet50**: AbsRel = 0.1247  
- **Best data-scale config (160k samples, 2 epochs)**: AbsRel = 0.0817  
- **Depth Anything V2**: AbsRel = 0.1219  
- **ZoeDepth**: AbsRel = 0.4565 (domain mismatch)

For full analysis and visualizations, see the PDF report.

##  Notes

- All code tested with PyTorch 2.0+ and CUDA 12.4.  
- Results (metrics, depth maps) are saved in `output/` directories.
```

