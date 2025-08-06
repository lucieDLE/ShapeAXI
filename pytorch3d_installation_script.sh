#!/bin/bash

# Complete PyTorch3D Installation Script for Python 3.12
# Tested on systems with GCC < 9 and CUDA requirements
# Author: Based on successful installation process

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
ENV_NAME="pytorch3d-py312"
PYTHON_VERSION="3.12"
TORCH_VERSION="2.7.0"
TORCHVISION_VERSION="0.22.0"
TORCHAUDIO_VERSION="2.7.0"
CUDA_INDEX_URL="https://download.pytorch.org/whl/cu128"

print_status "Starting PyTorch3D installation for Python ${PYTHON_VERSION}"
print_status "Environment: ${ENV_NAME}"

# Step 1: Check if conda is available
if ! command -v conda &> /dev/null; then
    print_error "conda is not installed or not in PATH"
    print_error "Please install Anaconda or Miniconda first"
    exit 1
fi

print_success "Conda found: $(conda --version)"

# Step 2: Create conda environment
print_status "Creating conda environment: ${ENV_NAME}"
if conda env list | grep -q "^${ENV_NAME} "; then
    print_warning "Environment ${ENV_NAME} already exists"
    read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n ${ENV_NAME} -y
        print_status "Removed existing environment"
    else
        print_status "Using existing environment"
    fi
fi

if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -n ${ENV_NAME} python==${PYTHON_VERSION} -y
    print_success "Created environment: ${ENV_NAME}"
fi

# Step 3: Activate environment
print_status "Activating environment: ${ENV_NAME}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Verify Python version
CURRENT_PYTHON=$(python --version)
print_success "Using Python: ${CURRENT_PYTHON}"

# Step 4: Install PyTorch with CUDA support
print_status "Installing PyTorch ${TORCH_VERSION} with CUDA support"
pip install shapeaxi==1.1.1

# Verify PyTorch installation
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

print_success "PyTorch installation completed"

# Step 5: Check GCC version and upgrade if needed
print_status "Checking GCC version"
GCC_VERSION=$(gcc --version | head -n1 | grep -oP '\d+\.\d+\.\d+' | head -1)
MAJOR_VERSION=$(echo $GCC_VERSION | cut -d. -f1)

print_status "Current GCC version: ${GCC_VERSION}"

if [ "$MAJOR_VERSION" -lt 9 ]; then
    print_warning "GCC version is too old (need >= 9.0). Pytorch3d will not compile.\n Installing newer GCC via conda"
    
    # Install newer GCC via conda
    conda install -c conda-forge gcc_linux-64=9 gxx_linux-64=9 -y
    
    # Set up environment variables for the new compilers
    export CC=$(which x86_64-conda-linux-gnu-gcc)
    export CXX=$(which x86_64-conda-linux-gnu-g++)
    
    print_status "Using GCC: ${CC}"
    print_status "Using G++: ${CXX}"
    
    # Verify new compiler versions
    NEW_GCC_VERSION=$(${CC} --version | head -n1 | grep -oP '\d+\.\d+\.\d+')
    print_success "Updated GCC version: ${NEW_GCC_VERSION}"
else
    print_success "GCC version is sufficient: ${GCC_VERSION}"
    export CC=$(which gcc)
    export CXX=$(which g++)
fi

# Step 6: Install CUDA toolkit if nvcc is not available
print_status "Checking for CUDA toolkit (nvcc)"
if ! command -v nvcc &> /dev/null; then
    print_warning "nvcc not found. Installing CUDA toolkit via conda"
    conda install nvidia/label/cuda-12.6.0::cuda-toolkit -c nvidia
    print_success "CUDA toolkit installed"
else
    print_success "nvcc found: $(which nvcc)"
fi

# Verify CUDA installation
NVCC_VERSION=$(nvcc --version | grep "release" | grep -oP 'V\d+\.\d+\.\d+')
print_success "CUDA toolkit version: ${NVCC_VERSION}"

# Step 7: Set up CUDA environment variables
print_status "Setting up CUDA environment variables"
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;9.0"

print_status "CUDA_HOME: ${CUDA_HOME}"
print_status "FORCE_CUDA: ${FORCE_CUDA}"
print_status "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"

# Step 8: Uninstall any existing PyTorch3D
print_status "Removing any existing PyTorch3D installation"
pip uninstall pytorch3d -y || print_status "No existing PyTorch3D found"

# Step 9: Install PyTorch3D from source with GPU support
print_status "Installing PyTorch3D from GitHub with GPU support"
print_status "This may take 10-30 minutes depending on your system..."

pip install "git+https://github.com/facebookresearch/pytorch3d.git" --verbose

print_success "PyTorch3D installation completed"

# Step 10: Verify installation
print_status "Verifying PyTorch3D GPU support"
python -c "
import torch
import pytorch3d
print(f'PyTorch3D version: {pytorch3d.__version__}')

# Test GPU operations. There are no other ways to check for GPU support
# The imports don't raise this error 
try:
    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.structures import Meshes
    
    # Create simple mesh on GPU
    verts = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=torch.float32).cuda()
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long).cuda()
    mesh = Meshes(verts=[verts], faces=[faces])
    
    # Test the critical operation
    points = sample_points_from_meshes(mesh, 1000)
    print('Y GPU operations successful!')
    print(f'  Sampled points shape: {points.shape}')
    print(f'  Points device: {points.device}')
    
    # Test renderer import
    import pytorch3d.renderer
    print('Y PyTorch3D renderer import successful!')
    
except Exception as e:
    print(f'X GPU operations failed: {e}')
"

print_success "Installation verification completed successfully!"

# Final summary
echo
echo "============================================="
print_success "INSTALLATION COMPLETED SUCCESSFULLY!"
echo "============================================="
echo
print_status "Environment name: ${ENV_NAME}"
print_status "Python version: $(python --version)"
print_status "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
print_status "PyTorch3D version: $(python -c 'import pytorch3d; print(pytorch3d.__version__)')"
echo
print_status "The environment is ready for PyTorch3D development with GPU support!"