#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define COARSE_FACTOR 2

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
    __shared__ float B_s[TILE_WIDTH][TILE_WIDTH*COARSE_FACTOR];
    
    int tx=threadIdx.x;int ty=threadIdx.y; 
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col_start=blockIdx.x*blockDim.x*COARSE_FACTOR+threadIdx.x;
    
    int numARows=Map_out;
    int numAColumns=Channel*K*K;
    int numBRows=Channel*K*K;
    int numBColumns=Batch*Height_out*Width_out;
    
    float sum[COARSE_FACTOR];
    int b[COARSE_FACTOR], houtput[COARSE_FACTOR], woutput[COARSE_FACTOR], validCol[COARSE_FACTOR];

#pragma unroll
    for(int cf=0;cf<COARSE_FACTOR;cf++){
        sum[cf]=0;
        int currentCol=col_start+cf*TILE_WIDTH;
        validCol[cf]=(currentCol<numBColumns);
        if(validCol[cf]){
            b[cf]=currentCol/Height_out/Width_out;
            int batchoffset=currentCol%(Height_out*Width_out);
            houtput[cf]=batchoffset/Width_out;
            woutput[cf]=batchoffset%Width_out;
        }
    }

#pragma unroll
    for(int tile=0;tile<(numAColumns+TILE_WIDTH-1)/TILE_WIDTH;tile++){
        int aCol=tile*TILE_WIDTH+tx;
        if(row<numARows&&numAColumns>aCol)A_s[threadIdx.y][threadIdx.x]=mask[(size_t)row*numAColumns+aCol];
        else A_s[threadIdx.y][threadIdx.x]=0;
        
        int bRow=tile*TILE_WIDTH+ty;
        for(int cf=0;cf<COARSE_FACTOR;cf++){
            if(validCol[cf]&&numBRows>bRow){
                int c=bRow/K/K;//see which input channel
                int channeloffset=bRow%(K*K);
                int p=channeloffset/K;
                int q=channeloffset%K;//which is the conv mask multiplying
                
                int hin=houtput[cf]+p;
                int win=woutput[cf]+q;
                
                size_t inputidx=(size_t)b[cf]*(Channel*Height*Width)+(size_t)c*(Height*Width)+(size_t)hin*Width+(size_t)win;
                B_s[ty][tx+cf*TILE_WIDTH]=input[inputidx];
            }else B_s[ty][tx+cf*TILE_WIDTH]=0;
        }
        __syncthreads();
        
#pragma unroll
        for(int i=0;i<TILE_WIDTH;i++){
            float a=A_s[threadIdx.y][i];
            for(int cf=0;cf<COARSE_FACTOR;cf++){
                sum[cf]+=a*B_s[i][tx+cf*TILE_WIDTH];
            }
        }
        __syncthreads();
    }
    
    for(int cf=0;cf<COARSE_FACTOR;cf++){
        int currentCol=col_start+cf*TILE_WIDTH;
        if(row<numARows&&currentCol<numBColumns){
            //permutee
            size_t image_size=Height_out*Width_out;
            int out_b=currentCol/image_size;
            int out_x=currentCol%image_size;
            if(out_x<image_size){
                output[(size_t)out_b*Map_out*image_size+row*image_size+out_x]=sum[cf];
            }
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    int heightOut=Height-K+1;
    int widthOut=Width-K+1;
    cudaMalloc((void**)device_input_ptr,(size_t)Batch*Channel*Height*Width*sizeof(float));
    cudaMalloc((void**)device_output_ptr,(size_t)Batch*Map_out*heightOut*widthOut*sizeof(float));
    cudaMalloc((void**)device_mask_ptr,(size_t)K*K*Channel*Map_out*sizeof(float));
    cudaMemcpy(*device_input_ptr,host_input,(size_t)Batch*Channel*Height*Width*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr,host_mask,(size_t)K*K*Channel*Map_out*sizeof(float),cudaMemcpyHostToDevice);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    int Height_out=Height-K+1;
    int Width_out=Width-K+1;
    int numCRows=Map_out;
    int numCColumns=Batch*Height_out*Width_out;
    
    dim3 blockDim(TILE_WIDTH,TILE_WIDTH,1);
    dim3 gridDim((numCColumns+TILE_WIDTH*COARSE_FACTOR-1)/(TILE_WIDTH*COARSE_FACTOR),
                 (numCRows+TILE_WIDTH-1)/TILE_WIDTH, 1);
                 
    matmul_conv_fused<<<gridDim, blockDim>>>(device_mask,device_input,device_output,Batch,Map_out,Channel,Height,Width,K);
    cudaDeviceSynchronize();
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    int Height_out=Height-K+1;
    int Width_out=Width-K+1;
    cudaMemcpy(host_output,device_output,(size_t)Batch*Map_out*Height_out*Width_out*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
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