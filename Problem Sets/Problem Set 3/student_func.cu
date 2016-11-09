#include "utils.h"
#include <stdio.h>

__global__ void shmem_reduce_minmax_kernel(const float* d_in,
                                        float* d_out,
                                        const size_t pixels,
                                        const bool max)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ float sdata[];

    int myId = threadIdx.x + ((blockDim.x) * blockIdx.x);
    int tid  = threadIdx.x;

    if (myId >= pixels){
      return;
    }

    //load shared mem from global mem
    if (max == true)
    {
      sdata[tid] = fmaxf(d_in[myId], d_in[myId + blockDim.x]);
    }
    else
    {
      sdata[tid] = fminf(d_in[myId], d_in[myId + blockDim.x]);
    }
    // sdata[tid] = d_in[myId];
    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            if (max == true)
            {
              sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
            }
            else
            {
              sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
            }
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = sdata[0];
    }
}

 float run_minmax(int blocks, int threads, const float* const d_in,
             float* d_intermediate, float* d_out, int pixels, const bool max){

  shmem_reduce_minmax_kernel<<<blocks,threads, sizeof(float) * threads>>>(d_in, d_intermediate, pixels, max);
  //cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  threads = blocks;
  blocks = 1;

  shmem_reduce_minmax_kernel<<<blocks, threads, sizeof(float) * threads>>>(d_intermediate, d_out, pixels, max);
  //cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  float h_minmax_out;
  checkCudaErrors(cudaMemcpy(&h_minmax_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
  return h_minmax_out;

}

__global__ void d_histogram(const float* const d_in, unsigned int* bins, const float lumRange, const float lumMin, const int numBins, const int pixels){

  int myId = threadIdx.x + (blockDim.x * blockIdx.x);

  if (myId >= pixels){
    return;
  }

  unsigned int bin = min(static_cast<unsigned int>(numBins - 1),
                         static_cast<unsigned int>((d_in[myId] - lumMin) / lumRange * numBins));

  atomicAdd(&(bins[bin]), 1);

}

void h_histogram(int blocks, int threads, const float* const d_in, unsigned int* h_bins, unsigned int* d_bins, const float lumRange, const float lumMin, const int numBins, const int pixels){

  for (int i = 0; i < numBins; i++){
    h_bins[i] = 0;
  }

  checkCudaErrors(cudaMemcpy(d_bins, h_bins, sizeof(int) * numBins, cudaMemcpyHostToDevice));

  d_histogram<<<blocks, threads>>>(d_in, d_bins, lumRange, lumMin, numBins, pixels);

  checkCudaErrors(cudaMemcpy(h_bins, d_bins, sizeof(int) * numBins, cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}

__global__ void d_cdf_calc(const unsigned int* const bins, unsigned int* const d_cdf, const size_t numBins){
  int myId = threadIdx.x + (blockDim.x * blockIdx.x);
  int maxId = numBins - 1;

  if (myId < numBins){
    d_cdf[myId] = bins[myId];
  }
  __syncthreads();

  for (int step = 0; step <= log2((float)numBins) - 1; step++){
    int gap = pow((float)2, (float)step + 1);
    int lookback = pow((float)2, (float)step);
    if (((myId + 1) % gap == 0) && (myId <= maxId)){
      d_cdf[myId] = d_cdf[myId] + d_cdf[myId - lookback];
    }
    __syncthreads();
  }
  if (myId == maxId){
    d_cdf[myId] = 0;
  }
  __syncthreads();
  for (int step = log2((float)numBins) - 1; step >= 0; step--){
    int gap = pow((float)2, (float)step + 1);
    int lookback = pow((float)2, (float)step);
    if (((myId + 1) % gap == 0) && (myId <= maxId)){
      unsigned int temp_cdf = d_cdf[myId];
      d_cdf[myId] = d_cdf[myId] + d_cdf[myId - lookback];
      d_cdf[myId - lookback] = temp_cdf;
    }
    __syncthreads();
  }
}

void h_cdf_calc(const unsigned int* const d_bins,
                unsigned int* const d_cdf,
                const size_t numBins,
                int maxThreadsPerBlock){
  int threads = maxThreadsPerBlock;
  int blocks = numBins / maxThreadsPerBlock;

  unsigned int* h_cdf = (unsigned int *) malloc(sizeof(unsigned int) * numBins);

  d_cdf_calc<<<blocks, threads>>>(d_bins, d_cdf, numBins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(h_cdf, d_cdf, sizeof(int) * numBins, cudaMemcpyDeviceToHost));

  free(h_cdf);
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  const int maxThreadsPerBlock = 1024;
  const size_t pixels = numRows * numCols;
  int threads = maxThreadsPerBlock;
  int blocks = pixels / maxThreadsPerBlock;

  float* d_minmax_intermediate;
  float* d_minmax_out;
  unsigned int* h_bins = (unsigned int *) malloc(sizeof(unsigned int) * numBins);
  unsigned int* d_bins;

  checkCudaErrors(cudaMalloc(&d_minmax_intermediate, sizeof(float) * pixels));
  checkCudaErrors(cudaMalloc(&d_minmax_out, sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_bins, sizeof(int) * numBins));

  min_logLum = run_minmax(blocks, threads, d_logLuminance, d_minmax_intermediate, d_minmax_out, pixels, false);
  max_logLum = run_minmax(blocks, threads, d_logLuminance, d_minmax_intermediate, d_minmax_out, pixels, true);

  float lumRange = max_logLum - min_logLum;

  h_histogram(blocks, threads, d_logLuminance, h_bins, d_bins, lumRange, min_logLum, numBins, pixels);

  h_cdf_calc(d_bins, d_cdf, numBins, maxThreadsPerBlock);
  free(h_bins);
  cudaFree(d_minmax_intermediate);
  cudaFree(d_minmax_out);
  cudaFree(d_bins);
}
