/*
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <http://unlicense.org/>
*/

#include <cmath>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>

#include "gpu_counts_dist.h"

void getTProbs(
    unsigned int nLandUse,
    unsigned int nPixels,
    unsigned int const* dkCounts,
    unsigned int const* dlCounts,
    float const* phis,
    float* probsOut
);

CUDATransitionCountsDist::CUDATransitionCountsDist(
    size_t nLandUse,
    size_t nPixels,
    float const* beginLogPhis
) : nLandUse_(nLandUse),
    nPixels_(nPixels)
{
    size_t
        nPhis(nLandUse*nLandUse),
        phisSize(nPhis*sizeof(float)),
        luSize(nLandUse*sizeof(unsigned int));

    // put the phis on the device. these remain for the life of the program
    cudaMalloc(&dPhis_, phisSize);
    cudaMemcpy(dPhis_, beginLogPhis, phisSize, cudaMemcpyHostToDevice);

    // put space for the the k counts on the device
    cudaMalloc(&dkCounts_, luSize);
    cudaMalloc(&dlCounts_, luSize);
    cudaMalloc(&dResult_, sizeof(float));
}

CUDATransitionCountsDist::CUDATransitionCountsDist(
    size_t nLandUse,
    size_t nPixels,
    double const* beginPhis
) : nLandUse_(nLandUse),
    nPixels_(nPixels)
{
    size_t
        nPhis(nLandUse*nLandUse),
        phisSize(nPhis*sizeof(float)),
        luSize(nLandUse*sizeof(unsigned int));

    std::vector<float> floatPhis(nPhis);
    std::copy(beginPhis, beginPhis + nPhis, floatPhis.begin());

    // put the phis on the device. these remain for the life of the program
    cudaMalloc(&dPhis_, phisSize);
    cudaMemcpy(dPhis_, &floatPhis[0], phisSize, cudaMemcpyHostToDevice);

    // put space for the the k counts on the device
    cudaMalloc(&dkCounts_, luSize);
    cudaMalloc(&dlCounts_, luSize);
    cudaMalloc(&dResult_, phisSize);
}

CUDATransitionCountsDist::~CUDATransitionCountsDist(void) {
    cudaFree(dPhis_);
    cudaFree(dkCounts_);
    cudaFree(dlCounts_);
    cudaFree(dResult_);
}

void CUDATransitionCountsDist::apply(
    unsigned int* kCounts,
    unsigned int* lCounts,
    unsigned int curk,
    unsigned int curl,
    float* results
) const {
    --kCounts[curk];
    --lCounts[curl];

    cudaMemcpy(dkCounts_, kCounts, nLandUse_*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dlCounts_, kCounts, nLandUse_*sizeof(int), cudaMemcpyHostToDevice);

    getTProbs(nLandUse_, nPixels_, dkCounts_, dlCounts_, dPhis_, dResult_);
    cudaMemcpy(results, dResult_, nLandUse_*nLandUse_*sizeof(float), cudaMemcpyDeviceToHost);

    ++kCounts[curk];
    ++lCounts[curl];
}
