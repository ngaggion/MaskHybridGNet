# Mask-HybridGNet: Graph-based segmentation with emergent anatomical correspondence from pixel-level supervision

![Pipeline Overview](figs/pipeline.png)

Official PyTorch implementation of "Mask-HybridGNet: Graph-based segmentation with emergent anatomical correspondence from pixel-level supervision".

Available at: https://arxiv.org/abs/2602.21179

## Overview

Mask-HybridGNet trains graph-based segmentation models using standard pixel-wise segmentation masks, without requiring manually annotated landmarks with point-to-point correspondences. 

![Pipeline Overview](figs/architecture.png)

**Applications:**
- Medical image segmentation across modalities (X-ray, ultrasound, MRI)
- Temporal tracking through cardiac cycles
- Cross-slice reconstruction in MRI sequences
- Population-level morphological analysis

## Online Demo

Inference-ready environment available at: https://huggingface.co/spaces/ngaggion/MaskHybridGNet

---

## Installation

### Option 1: Docker (Recommended)

Due to complex dependencies involving custom CUDA compilations (PyTorch3D and Differentiable Rasterizer), we highly recommend using Docker to run this code. 

You can pull the image:
```bash
docker pull ngaggion/maskhybridgnet

```

**Building the Image:**
Ensure you have the NVIDIA Container Toolkit installed, then build the image from the project root:
```bash
docker build -t maskhybridgnet .

```

**Usage:**
Run the container interactively with GPU access. The `--ipc=host` flag may be required to provide sufficient shared memory for PyTorch's background data loader workers:

```bash
docker run --gpus all -it --rm --ipc=host -v $(pwd):/workspace maskhybridgnet

```

### Option 2: Local Installation

**For Inference:**
Using Python 3.10 (Recommended for PyTorch 2.0.1 compatibility), install the base dependencies:

```bash
pip install -r requirements.txt

```

**For Training:**
Training requires additional custom CUDA dependencies:

1. **PyTorch3D**: To install the Chamfer-Loss, follow the official installation instructions at https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
2. **Differentiable Rasterizer**: Compile the SoftRasterizer (adapted from BoundaryFormer). Note that this requires `nvcc` and a local CUDA toolkit installation:
```bash
cd losses/diff_ras
python setup.py install

```

*Original implementation: https://github.com/mlpc-ucsd/BoundaryFormer*

---

## Datasets

Evaluated on:

* **Chest X-ray**: JSRT, PadChest, Montgomery, Shenzhen, PAX-Ray++
* **Cardiac Ultrasound**: CAMUS
* **Cardiac MRI**: Sunnybrook
* **Fetal Imaging**: HC18, JNU-IFM, PSFHS

## Usage

### Dataset Preparation

Dataset-specific preprocessing notebooks are provided with the prefix `1_Dataset_Generation_*.ipynb`. These must be adapted to your data structure and annotation format.

Examples:

* `1_Dataset_Generation_ChestXRay.ipynb`
* `1_Dataset_Generation_CAMUS.ipynb`
* `1_Dataset_Generation_PAXRay_Front.ipynb`

*See the paper appendix for detailed configuration system instructions.*

### Graph Representations

Models can be trained with one of two graph representations, selected with the `--representation` flag of `2_Trainer.py` (and `8_Trainer_MSE.py`):

* `independent` (default): each organ is modelled by its own closed contour graph (per-organ block-diagonal adjacency). This is the representation previously referred to as "naive".
* `unified`: all organs share a single graph with shared boundary nodes. This is the representation previously referred to as "non-naive".

Adjacency matrices and atlas files for newly generated datasets are written to `Independent/` and `Unified/` subfolders inside the dataset directory.

**Backward compatibility.** The legacy nomenclature is still fully supported:

* The deprecated `--naive` / `--non-naive` flags continue to work as aliases for `--representation independent` / `--representation unified`.
* Checkpoints and `hyperparameters.json` files written by older runs (which only store the boolean `naive` key) load unchanged; the value is mapped to the corresponding representation automatically. New runs additionally store the `naive` flag so older inference scripts keep working.
* Datasets generated with the old `Naive/` and `NonNaive/` folder layout are still read directly. When both the new and legacy folders are present, the new `Independent/` / `Unified/` folders take precedence.

## Project Structure

```text
├── data/                          # Dataset loading and transformations
├── models/                        # Network architectures
├── losses/                        # Loss functions and differentiable rasterizer
├── training/                      # Training utilities
├── evaluation/                    # Evaluation metrics
├── utils/                         # Graph operation helpers
├── 1_Dataset_Generation_*.ipynb   # Dataset preprocessing scripts
├── 2_Trainer.py                   # Training script
├── 3_Evaluate.py                  # Evaluation script
├── 4_Eval*.ipynb                  # Dataset-specific evaluation
└── 7_Segment_From_Mask.py         # Atlas extraction from masks

```

## Hardware Requirements

* **Inference**: ~4GB VRAM
* **Training**: 12-24GB VRAM
* **Training Time**: 12-24 hours per dataset (tested on NVIDIA RTX 3090)

## Citation

If you find this code or our methodology useful in your research, please cite:

```bibtex
@article{gaggion2026mask,
  title={Mask-HybridGNet: Graph-based segmentation with emergent anatomical correspondence from pixel-level supervision},
  author={Gaggion, Nicol{\'a}s and Ledesma-Carbayo, Maria J and Christodoulidis, Stergios and Vakalopoulou, Maria and Ferrante, Enzo},
  journal={arXiv preprint arXiv:2602.21179},
  year={2026}
}

```

## Contact

For questions or issues, please open an issue on GitHub.

## License

See LICENSE file.