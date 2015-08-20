#ifndef CUDA_COUNTS_DIST_HEADER_INCLUDED_SF61DH34D3GHS1DFG313H2DH1N36F45
#define CUDA_COUNTS_DIST_HEADER_INCLUDED_SF61DH34D3GHS1DFG313H2DH1N36F45
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

#ifdef COUNTSDIST_EXPORTS
#define COUNTSDIST_API __declspec(dllexport)
#else
#define COUNTSDIST_API __declspec(dllimport)
#endif



class COUNTSDIST_API CUDATransitionCountsDist {
public:
    CUDATransitionCountsDist(size_t nLandUse, size_t nPixels, float const* beginLogPhis);
    CUDATransitionCountsDist(size_t nLandUse, size_t nPixels, double const* beginPhis);
    ~CUDATransitionCountsDist(void);
    void apply(
        unsigned int* kCounts,
        unsigned int* lCounts,
        unsigned int curk,
        unsigned int curl,
        float* results
    ) const;

private:
    size_t
        nLandUse_,
        nPixels_;

    mutable unsigned int
        *dkCounts_,
        *dlCounts_;

    mutable float
        hResult_, // h for 'host'
        *dResult_;

    float
        *dPhis_;
};

#endif //CUDA_COUNTS_DIST_HEADER_INCLUDED_SF61DH34D3GHS1DFG313H2DH1N36F45
