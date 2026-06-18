"""
Central data-path configuration.

These are plain strings -- just edit them to your paths. (If you prefer env
vars, use os.environ.get("CIFAR_DATA_ROOT", DATA_ROOT) -- note the FIRST arg is
the VARIABLE NAME, the path goes in the DEFAULT/second slot.)

IMPORTANT:
  - DATA_ROOT must be the PARENT folder that CONTAINS cifar-10-batches-py/,
    NOT cifar-10-batches-py itself.
  - CIFAR10C_ROOT must directly contain the .npy files (<corruption>.npy + labels.npy).
"""

# Parent folder that contains cifar-10-batches-py/
DATA_ROOT = "/scratch/zdong112/cifar10"

# Folder holding the CIFAR-10-C .npy files
CIFAR10C_ROOT = "/scratch/zdong112/cifar10c"

# Whether torchvision may download CIFAR-10 if missing (False on offline nodes)
DOWNLOAD = False
