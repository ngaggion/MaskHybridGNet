# 1. Use the official PyTorch development image.
# The '-devel' tag is CRITICAL because it contains 'nvcc' (CUDA compiler),
# which you need to compile your Differentiable Rasterizer (and chamferdist/PyTorch3D).
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# 2. Prevent interactive prompts during apt-get installations
ENV DEBIAN_FRONTEND=noninteractive

# 3. Install system dependencies required for compilation
# 'ninja-build' is a massive time-saver for compiling CUDA kernels
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# 4. Set up the working directory
WORKDIR /workspace

# 5. Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# 6. Install the basic requirements
# (torch and torchvision are already in the base image, but running this ensures 
# numpy, opencv, scipy, and matplotlib are installed)
RUN pip install --no-cache-dir -r requirements.txt

# 7. OPTIONAL: If you still want to use full PyTorch3D instead of chamferdist, 
# uncomment the lines below. Setting TORCH_CUDA_ARCH_LIST speeds up the build 
# by only compiling for modern GPUs (e.g., Ampere, Ada).
# ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
# ENV FORCE_CUDA="1"
# RUN pip install --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.4"

# 8. Copy the rest of your repository into the container
COPY . .

# 9. Compile the custom Differentiable Rasterizer
WORKDIR /workspace/losses/diff_ras

# Force PyTorch to compile even without a GPU present
ENV FORCE_CUDA="1"
# Limit architectures to what CUDA 11.7 explicitly supports
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

RUN pip install --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.4"

RUN python setup.py install

# 10. Reset working directory to the project root
WORKDIR /workspace

# 11. Default command to keep the container running interactively
CMD ["/bin/bash"]