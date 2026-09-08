#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 16

__global__ void matmul_conv_fused(const float *mask, const float *input, float *output,
                                  int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
    /*
    TODO: Modify this function to implement the fused unroll-matmul-permute kernel.
    
    Function parameter definitions:
    mask - convolution kernel
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
        const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
     __shared__ float A_s[TILE_WIDTH][TILE_WIDTH];
__shared__ float B_s[TILE_WIDTH][TILE_WIDTH];
int tx=threadIdx.x;int ty=threadIdx.y; 
int row=blockIdx.y*blockDim.y+threadIdx.y;
int col=blockIdx.x*blockDim.x+threadIdx.x;
int numARows=Map_out;
int numAColumns=Channel*K*K;
int numBRows=Channel*K*K;
 int numBColumns=Batch*Height_out*Width_out;
float sum=0;
  int b=col/Height_out/Width_out;//what batch
        int batchoffset=col%(Height_out*Width_out);
        int houtput=batchoffset/Width_out;
        int woutput=batchoffset%Width_out;
#pragma unroll
for(int tile=0;tile<(numAColumns+TILE_WIDTH-1)/TILE_WIDTH;tile++){
    int aCol=tile*TILE_WIDTH+tx;
    if(row<numARows&&numAColumns>aCol)A_s[threadIdx.y][threadIdx.x]=mask[(size_t)row*numAColumns+aCol];
    else A_s[threadIdx.y][threadIdx.x]=0;
    
    int bRow=tile*TILE_WIDTH+ty;
    if(col<numBColumns&&numBRows>bRow){
        
        int c=bRow/K/K;//see which input channel
        int channeloffset=bRow%(K*K);
        int p=channeloffset/K;
        int q=channeloffset%K;//which is the conv mask multiplying
        
       
        int hin=houtput+p;
        int win=woutput+q;
        
        size_t inputidx=(size_t)b*(Channel*Height*Width)+(size_t)c*(Height*Width)+(size_t)hin*Width+(size_t)win;
        B_s[ty][tx] =input[inputidx];
        
    }else B_s[threadIdx.y][threadIdx.x]=0;
        __syncthreads();
#pragma unroll
for(int i=0;i<TILE_WIDTH;i++){
        sum+=A_s[threadIdx.y][i]*B_s[i][threadIdx.x];
    }
    __syncthreads();
}
if(row<numARows&&col<numBColumns){
    //permutee
    size_t image_size=Height_out*Width_out;
      b = col/image_size;
    int x = col%image_size;
    if (x < image_size) {

            output[(size_t)b * Map_out * image_size + row * image_size + x] =
                    sum;
    }
}

}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU
 
    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
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
    // TODO: Set the kernel dimensions and call the fused kernel
int Height_out=Height-K+1;
int Width_out=Width-K+1;
int numCRows=Map_out;
    int numCColumns=Batch*Height_out*Width_out;
    dim3 blockDim(TILE_WIDTH,TILE_WIDTH,1);
    dim3 gridDim((numCColumns+TILE_WIDTH-1)/TILE_WIDTH,
                 (numCRows+TILE_WIDTH-1)/TILE_WIDTH, 1);
    matmul_conv_fused<<<gridDim, blockDim>>>(device_mask,device_input,device_output,Batch,Map_out,Channel,Height,Width,K);
    cudaDeviceSynchronize();
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
int Height_out=Height-K +1;
int Width_out=Width-K+1;
    cudaMemcpy(host_output,device_output,(size_t)Batch*Map_out*Height_out*Width_out*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
    // TODO: Free device memory

}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}

