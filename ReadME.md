# Mask-HybridGNet: Contour-Based Medical Image Segmentation via Implicit Anatomical Correspondence Learning

![Pipeline Overview](figs/pipeline.png)

Official PyTorch implementation of "Contour-Based Medical Image Segmentation via Implicit Anatomical Correspondence Learning".

## Overview

Mask-HybridGNet trains graph-based segmentation models using standard pixel-wise segmentation masks, without requiring manually annotated landmarks with point-to-point correspondences.

**Applications:**
- Medical image segmentation across modalities (X-ray, ultrasound, MRI)
- Temporal tracking through cardiac cycles
- Cross-slice reconstruction in MRI sequences
- Population-level morphological analysis

## Online Demo

Inference-ready environment available at: https://huggingface.co/spaces/ngaggion/ContourBasedSegmentation

## Installation

### For Inference

With python 3.13.3, see `requirements.txt` for dependencies. Install with:
```bash
pip install -r requirements.txt
```

### For Training

Training requires additional dependencies:

1. **PyTorch3D**: To install the Chamfer-Loss, follow installation instructions at https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

2. **Differentiable Rasterizer**: Compile the SoftRasterizer (adapted from BoundaryFormer):
   ```bash
   cd losses/diff_ras
   python setup.py install
   ```
   Original implementation: https://github.com/mlpc-ucsd/BoundaryFormer

**Note:** Training requires CUDA for rasterizer compilation.

## Datasets

Evaluated on:
- **Chest X-ray**: JSRT, PadChest, Montgomery, Shenzhen, PAX-Ray++
- **Cardiac Ultrasound**: CAMUS
- **Cardiac MRI**: Sunnybrook
- **Fetal Imaging**: HC18, JNU-IFM, PSFHS

## Usage

### Dataset Preparation

Dataset-specific preprocessing notebooks are provided with prefix `1_Dataset_Generation_*.ipynb`. These must be adapted to your data structure and annotation format.

Examples:
- `1_Dataset_Generation_ChestXRay.ipynb`
- `1_Dataset_Generation_CAMUS.ipynb`
- `1_Dataset_Generation_PAXRay_Front.ipynb`

See paper appendix for configuration system details.

## Project Structure

```
├── data/                          # Dataset loading and transformations
├── models/                        # Network architectures
├── losses/                        # Loss functions and differentiable rasterizer
├── training/                      # Training utilities
├── evaluation/                    # Evaluation metrics
├── utils/                         # Graph operation helpers
├── 1_Dataset_Generation_*.ipynb  # Dataset preprocessing scripts
├── 2_Trainer.py                  # Training script
├── 3_Evaluate.py                 # Evaluation script
├── 4_Eval*.ipynb                 # Dataset-specific evaluation
└── 7_Segment_From_Mask.py        # Atlas extraction from masks
```

## Hardware Requirements

- **Inference**: ~4GB VRAM
- **Training**: 12-24GB VRAM
- **Training Time**: 12-24 hours per dataset (NVIDIA RTX 3090)

## Citation

TO-DO: Add citation information here when available.

## Contact

For questions or issues, please open an issue on GitHub.

## License

See LICENSE file.