#pragma once

#include <c10/cuda/CUDAStream.h>

// Current PyTorch CUDA stream for this thread.
#define MY_STREAM c10::cuda::getCurrentCUDAStream().stream()
