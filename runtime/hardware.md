# Runtime benchmark hardware

The dedicated six-station runtime benchmark used:

| Component | Recorded value |
|---|---|
| CPU | Intel Xeon Gold 6240C at 2.60 GHz |
| Logical CPU cores visible | 72 |
| GPU used for Proposed-GPU | one NVIDIA GeForce RTX 4090, 24 GiB |
| Operating system | Linux 5.15, x86-64, glibc 2.31 |
| Python | 3.12.13 |
| PyTorch | 2.5.1+cu121 |
| CUDA runtime | 12.1 |
| cuDNN | 9.1.0 |
| PyTorch intra-op threads | 4 |
| PyTorch inter-op threads | 1 |
| Thread environment | `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4` |

The host contained multiple GPUs, but the Proposed-GPU arm used only one
device. The benchmark did not record a controlled background-load schedule or
CPU-affinity mask, so the results should be interpreted as measurements on
this system rather than hardware-normalized latency guarantees.
