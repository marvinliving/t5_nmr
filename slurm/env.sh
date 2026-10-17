# Cluster-specific settings, sourced by submit.sh and by every Slurm job.
# To run on another cluster, change this file rather than the job scripts.

# ===== Submission defaults =====
# sbatch reads SBATCH_* variables as if they were command-line options.
export SBATCH_PARTITION="${SBATCH_PARTITION:-workq}"
# export SBATCH_ACCOUNT="my-project"

# CPU cores requested per GPU. An Isambard-AI node has one 72-core Grace CPU
# per GH200 GPU.
CPUS_PER_GPU="${CPUS_PER_GPU:-72}"


# Dataset used by runs whose config doesn't set DATA_DIR.
export DATA_DIR="${DATA_DIR:-/projects/b5an/alberts_2d}"
# Datasets ./submit.sh check-lengths checks, separated by spaces.
CHECK_DATA_DIRS="${CHECK_DATA_DIRS:-/projects/b5an/alberts_2d /projects/b5an/nmr_expt_data}"
# Dataset ./submit.sh progress scores checkpoints on unless EVAL_DATA_DIR
# is given: the external, experimental spectra.
PROGRESS_DATA_DIR="${PROGRESS_DATA_DIR:-/projects/b5an/nmr_expt_data}"


# ===== Job environment =====

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
CONDA_ENV="${CONDA_ENV:-t5}"

setup_job_env() {
  module load cuda/12.6
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
}

# NCCL between nodes over Slingshot through the aws-ofi-nccl plugin, with the
# settings from the Isambard-AI NCCL guide:
# https://docs.isambard.ac.uk/user-documentation/guides/nccl/
# Within one node NCCL uses NVLink and needs none of this.
setup_multinode_nccl() {
  module load brics/nccl brics/aws-ofi-nccl
  export NCCL_NET="AWS Libfabric"
  export NCCL_SOCKET_IFNAME=hsn
  export NCCL_NET_GDR_LEVEL=PHB
  export NCCL_CROSS_NIC=1
  export NCCL_MIN_NCHANNELS=4
  export NCCL_GDRCOPY_ENABLE=1
  export NCCL_NET_FORCE_FLUSH=0
  export FI_CXI_DEFAULT_CQ_SIZE=131072
  export FI_CXI_DEFAULT_TX_SIZE=2048
  export FI_HMEM_CUDA_USE_GDRCOPY=1
  export FI_CXI_RDZV_PROTO=alt_read
}
