#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

__global__ void matmul_conv_fused(const float *__restrict__ mask,const float *__restrict__ input,float *__restrict__ output,
                                  int Batch,int Map_out,int Channel,int Height,int Width,int K)
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
        const int Height_out=Height-K+1;
    const int Width_out=Width-K+1;
     __shared__ float M_s[16][4];
int tx=threadIdx.x; 
int row_start=blockIdx.y*16;
int col=blockIdx.x*64+tx;
int numARows=Map_out;
int numAColumns=Channel*K*K;
int numBRows=Channel*K*K;
 int numBColumns=Batch*Height_out*Width_out;
float sum[16]={0};
  int b=col/Height_out/Width_out;//what batch
        int batchoffset=col%(Height_out*Width_out);
        int houtput=batchoffset/Width_out;
        int woutput=batchoffset%Width_out;
        
        int c=0;//see which input channel
        int p=0;
        int q=0;//which is the conv mask multiplying
        size_t inputidx=(size_t)b*(Channel*Height*Width)+(size_t)c*(Height*Width)+(size_t)houtput*Width+(size_t)woutput;
#pragma unroll
for(int tile=0;tile<(numAColumns+4-1)/4;tile++){
    int m_row=tx/4;
    int m_col=tx%4;
    int global_m_row=row_start+m_row;
    int global_m_col=tile*4+m_col;
    if(global_m_row<numARows&&global_m_col<numAColumns)M_s[m_row][m_col]=mask[(size_t)global_m_row*numAColumns+global_m_col];
    else M_s[m_row][m_col]=0;
    
    float N_reg[4]={0};
    if(col<numBColumns){
        #pragma unroll
        for(int i=0;i<4;i++){
            int bRow=tile*4+i;
            if(bRow<numBRows){
                N_reg[i]=input[inputidx];
                q++;
                inputidx++;
                if(q==K){
                    q=0;
                    p++;
                    inputidx+=Width-K;
                    if(p==K){
                        p=0;
                        c++;
                        inputidx+=(Height-K)*Width;
                    }
                }
            }
        }
    }
        __syncthreads();
#pragma unroll
for(int i=0;i<4;i++){
        #pragma unroll
        for(int j=0;j<16;j++){
            sum[j]+=M_s[j][i]*N_reg[i];
        }
    }
    __syncthreads();
}
if(col<numBColumns){
    //permutee
    size_t image_size=Height_out*Width_out;
      b=col/image_size;
    int x=col%image_size;
    if(x<image_size){
        #pragma unroll
        for(int j=0;j<16;j++){
            int global_row=row_start+j;
            if(global_row<numARows){
                output[(size_t)b*Map_out*image_size+global_row*image_size+x]=sum[j];
            }
        }
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
    dim3 blockDim(64,1,1);
    dim3 gridDim((numCColumns+64-1)/64,
                 (numCRows+16-1)/16, 1);
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