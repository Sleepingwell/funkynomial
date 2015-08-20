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

template <unsigned int prevPower2>
__global__ void transitionKernel(
    size_t nPixels,
    size_t nn1Pixels,
    float mu1Constant,
    float mu2Constant,
    float logNLandUse,
    unsigned int const* dkCounts,
    unsigned int const* dlCounts,
    float const* dPhis,
    float* probsOut
) {
    float
        klt,
        mu1,
        mu2,
        kll_l(0.0f),
        kkl_l(0.0f);

    unsigned int k = threadIdx.x;
    unsigned int l = threadIdx.y;
    unsigned int nLandUse = blockDim.x;
    unsigned int tid = k * nLandUse + l;
    unsigned int bid = blockIdx.x * nLandUse + blockIdx.y;

    // counts in local memory
    __shared__ int kCounts[32];
    __shared__ int lCounts[32];

    // the (log) transition probabilities
    __shared__ float phis[1024];
    __shared__ float kl1[1024];
    __shared__ float kl2[1024];
    __shared__ float kll_kkl[1024];

    // copy the counts to local memory
    phis[tid] = dPhis[tid];
    if(l == 0) {
        // remember the block is always square
        kCounts[k] = dkCounts[k];
        lCounts[k] = dlCounts[k];
    }
    __syncthreads();

    // do the sums
    klt = (kCounts[k] + 1) * (lCounts[l] + 1) * phis[tid];
    kl1[tid] = klt;
    kl2[tid] = klt * phis[tid];
    for(int korl(0); korl < k; ++korl) kkl_l += 2.0 * klt * kCounts[korl] * phis[korl*nLandUse + l];
    kkl_l += klt * (kCounts[k] + 1) * phis[tid];
    for(int korl(0); korl < l; ++korl) kll_l += 2.0 * klt * lCounts[korl] * phis[k*nLandUse + korl];
    kll_l += klt * (lCounts[l] + 1) * phis[tid];

    kll_kkl[tid] = kkl_l + kll_l;
    __syncthreads();

    // write rows beyond the largest power of two that fits into the first lastPower2 rows
    if(tid >= prevPower2) {
        kl1[tid - prevPower2] += kl1[tid];
        kl2[tid - prevPower2] += kl2[tid];
        kll_kkl[tid - prevPower2] += kll_kkl[tid];
    }
    __syncthreads();

    // now run the reduction
    if(prevPower2 >= 1024) {
        if(tid < 512) {
            kl1[tid] += kl1[tid + 512];
            kl2[tid] += kl2[tid + 512];
            kll_kkl[tid] += kll_kkl[tid + 512];
        }
        __syncthreads();
    }
    if(prevPower2 >= 512) {
        if(tid < 256) {
            kl1[tid] += kl1[tid + 256];
            kl2[tid] += kl2[tid + 256];
            kll_kkl[tid] += kll_kkl[tid + 256];
        }
        __syncthreads();
    }
    if(prevPower2 >= 256) {
        if(tid < 128) {
            kl1[tid] += kl1[tid + 128];
            kl2[tid] += kl2[tid + 128];
            kll_kkl[tid] += kll_kkl[tid + 128];
        }
        __syncthreads();
    }
    if(prevPower2 >= 128) {
        if(tid < 64) {
            kl1[tid] += kl1[tid + 64];
            kl2[tid] += kl2[tid + 64];
            kll_kkl[tid] += kll_kkl[tid + 64];
        }
        __syncthreads();
    }
    if(tid < 32) { // if we are in the 
        if(prevPower2 >= 64) {
            kl1[tid] += kl1[tid + 32];
            kl2[tid] += kl2[tid + 32];
            kll_kkl[tid] += kll_kkl[tid + 32];
        }
        if(prevPower2 >= 32) {
            kl1[tid] += kl1[tid + 16];
            kl2[tid] += kl2[tid + 16];
            kll_kkl[tid] += kll_kkl[tid + 16];
        }
        if(prevPower2 >= 16) {
            kl1[tid] += kl1[tid + 8];
            kl2[tid] += kl2[tid + 8];
            kll_kkl[tid] += kll_kkl[tid + 8];
        }
        if(prevPower2 >= 8) {
            kl1[tid] += kl1[tid + 4];
            kl2[tid] += kl2[tid + 4];
            kll_kkl[tid] += kll_kkl[tid + 4];
        }
        if(prevPower2 >= 4) {
            kl1[tid] += kl1[tid + 2];
            kl2[tid] += kl2[tid + 2];
            kll_kkl[tid] += kll_kkl[tid + 2];
        }
        if(prevPower2 >= 2) {
            kl1[tid] += kl1[tid + 1];
            kl2[tid] += kl2[tid + 1];
            kll_kkl[tid] += kll_kkl[tid + 1];
        }
    }

    __syncthreads();

    if(tid == 0) {
        mu1 = mu1Constant + kl1[0] / nPixels;
        mu2 = mu2Constant
            + kl2[0] / nPixels
            + kl2[0] / nn1Pixels
            - kll_kkl[0] / nn1Pixels
            + (kl1[0] * kl1[0]) / nn1Pixels
            - 2.0f * logNLandUse * kl1[0];
        probsOut[bid] = mu1 + (mu2 - mu1*mu1) / 2.0f;
    }
}



void getTProbs(
    unsigned int nLandUse,
    unsigned int nPixels,
    unsigned int const* dkCounts,
    unsigned int const* dlCounts,
    float const* dPhis,
    float* probsOut
) {
    unsigned int
        prevPower2(1),
        nn1Pixels(nPixels * (nPixels - 1));

    float
        logNLandUse(static_cast<float>(std::log(nLandUse))),
        mu1Constant(-(nPixels * logNLandUse)),
        mu2Constant(mu1Constant * mu1Constant);

    dim3
        blockDim(nLandUse, nLandUse);

    while(prevPower2 <= nLandUse*nLandUse) prevPower2 <<= 1;
    prevPower2 >>= 1;

    switch(prevPower2) {
    case 1024:
        transitionKernel<1024> <<<blockDim, blockDim>>>(nPixels, nn1Pixels, mu1Constant, mu2Constant, logNLandUse, dkCounts, dlCounts, dPhis, probsOut);
        break;
    case 512:
        transitionKernel<512> <<<blockDim, blockDim>>>(nPixels, nn1Pixels, mu1Constant, mu2Constant, logNLandUse, dkCounts, dlCounts, dPhis, probsOut);
        break;
    case 256:
        transitionKernel<256> <<<blockDim, blockDim>>>(nPixels, nn1Pixels, mu1Constant, mu2Constant, logNLandUse, dkCounts, dlCounts, dPhis, probsOut);
        break;
    case 128:
        transitionKernel<128> <<<blockDim, blockDim>>>(nPixels, nn1Pixels, mu1Constant, mu2Constant, logNLandUse, dkCounts, dlCounts, dPhis, probsOut);
        break;
    case 64:
        transitionKernel<64> <<<blockDim, blockDim>>>(nPixels, nn1Pixels, mu1Constant, mu2Constant, logNLandUse, dkCounts, dlCounts, dPhis, probsOut);
        break;
    case 32:
        transitionKernel<32> <<<blockDim, blockDim>>>(nPixels, nn1Pixels, mu1Constant, mu2Constant, logNLandUse, dkCounts, dlCounts, dPhis, probsOut);
        break;
    case 16:
        transitionKernel<16> <<<blockDim, blockDim>>>(nPixels, nn1Pixels, mu1Constant, mu2Constant, logNLandUse, dkCounts, dlCounts, dPhis, probsOut);
        break;
    case 8:
        transitionKernel<8> <<<blockDim, blockDim>>>(nPixels, nn1Pixels, mu1Constant, mu2Constant, logNLandUse, dkCounts, dlCounts, dPhis, probsOut);
        break;
    case 4:
        transitionKernel<4> <<<blockDim, blockDim>>>(nPixels, nn1Pixels, mu1Constant, mu2Constant, logNLandUse, dkCounts, dlCounts, dPhis, probsOut);
        break;
    case 2:
        transitionKernel<2> <<<blockDim, blockDim>>>(nPixels, nn1Pixels, mu1Constant, mu2Constant, logNLandUse, dkCounts, dlCounts, dPhis, probsOut);
        break;
    case 1:
        transitionKernel<1> <<<blockDim, blockDim>>>(nPixels, nn1Pixels, mu1Constant, mu2Constant, logNLandUse, dkCounts, dlCounts, dPhis, probsOut);
        break;
    }
}
