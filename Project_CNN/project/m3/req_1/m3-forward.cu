#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

__global__ void matmul_conv_fused(const float *mask, const float *input, float *output,
                                  int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
     __shared__ half A_s[16][16];
__shared__ half B_s[16][16];//changed to half for tensor core requirement
__shared__ float C_s[16][16]; //Temporary shared memory to hold Tensor Core results

int tx=threadIdx.x;int ty=threadIdx.y; 
int row=blockIdx.y*16+threadIdx.y;
int col=blockIdx.x*16+threadIdx.x;
int numARows=Map_out;
int numAColumns=Channel*K*K;
int numBRows=Channel*K*K;
 int numBColumns=Batch*Height_out*Width_out;

wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b_frag;
wmma::fragment<wmma::accumulator,16,16,16,float> c_frag;
wmma::fill_fragment(c_frag,0.0f);

    int b=col/Height_out/Width_out;//what batch
        int batchoffset=col%(Height_out*Width_out);
        int houtput=batchoffset/Width_out;
        int woutput=batchoffset%Width_out;
        
    int warpId=(threadIdx.y*16+threadIdx.x)/32;

for(int tile=0;tile<(numAColumns+15)/16;tile++){
    int aCol=tile*16+tx;
    if(row<numARows&&numAColumns>aCol)A_s[threadIdx.y][threadIdx.x]=__float2half(mask[(size_t)row*numAColumns+aCol]);
    else A_s[threadIdx.y][threadIdx.x]=__float2half(0.0f);
    
    int bRow=tile*16+ty;
    if(col<numBColumns&&numBRows>bRow){
        
        int c=bRow/K/K;//see which input channel
        int channeloffset=bRow%(K*K);
        int p=channeloffset/K;
        int q=channeloffset%K;//which is the conv mask multiplying
        
       
        int hin=houtput+p;
        int win=woutput+q;
        
        size_t inputidx=(size_t)b*(Channel*Height*Width)+(size_t)c*(Height*Width)+(size_t)hin*Width+(size_t)win;
        B_s[ty][tx] =__float2half(input[inputidx]);
        
    }else B_s[threadIdx.y][threadIdx.x]=__float2half(0.0f);
        __syncthreads();

if(warpId==0){
    wmma::load_matrix_sync(a_frag,&A_s[0][0],16);
    wmma::load_matrix_sync(b_frag,&B_s[0][0],16);
    wmma::mma_sync(c_frag,a_frag,b_frag,c_frag);
}
    __syncthreads();
}

if(warpId==0){
    wmma::store_matrix_sync(&C_s[0][0],c_frag,16,wmma::mem_row_major);
}
__syncthreads();

if(row<numARows&&col<numBColumns){
    //permutee
    size_t image_size=Height_out*Width_out;
      b = col/image_size;
    int x = col%image_size;
    if (x < image_size) {
           
            output[(size_t)b * Map_out * image_size + row * image_size + x] =
                    C_s[ty][tx];
    }
}

}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
      int heightOut=Height-K+1;
    int widthOut=Width-K+1;
cudaMalloc((void**)device_input_ptr,(size_t)Batch*Channel*Height*Width*sizeof(float));
cudaMalloc((void**)device_output_ptr,(size_t)Batch*Map_out*heightOut*widthOut*sizeof(float));
cudaMalloc((void**)device_mask_ptr,(size_t)K*K*Channel*Map_out *sizeof(float));
cudaMemcpy(*device_input_ptr,host_input,(size_t)Batch*Channel*Height*Width*sizeof(float),cudaMemcpyHostToDevice);
cudaMemcpy(*device_mask_ptr,host_mask,(size_t)K*K*Channel*Map_out*sizeof(float),cudaMemcpyHostToDevice);

}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
int Height_out=Height-K+1;
int Width_out=Width-K+1;
int numCRows=Map_out;
    int numCColumns=Batch*Height_out*Width_out;
    dim3 blockDim(16,16,1);
    dim3 gridDim((numCColumns+15)/16,
                 (numCRows+15)/16, 1);
    matmul_conv_fused<<<gridDim, blockDim>>>(device_mask,device_input,device_output,Batch,Map_out,Channel,Height,Width,K);
    cudaDeviceSynchronize();
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
int Height_out=Height-K +1;
int Width_out=Width-K+1;
    cudaMemcpy(host_output,device_output,(size_t)Batch*Map_out*Height_out*Width_out*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
}