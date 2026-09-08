#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 16
__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
     

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int W_tile=(Width_out+16-1)/16;
    int m=blockIdx.x;
    int b=blockIdx.z;
    int h=(blockIdx.y/W_tile)*16+threadIdx.y;
    int w=(blockIdx.y%W_tile)*16+threadIdx.x;
    if(h<Height_out&&w<Width_out){
        float sum=0;
        for(int c=0;c<Channel;c++){
            for(int p=0;p<K;p++){
                for(int q=0;q<K;q++)sum+=in_4d(b,c,h+p,w+q)*mask_4d(m,c,p,q);
            }
        }
out_4d(b,m, h,w)=sum;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    int H_out=Height-(K-1);
    int W_out=Width-(K-1);
    cudaMalloc((void**) device_input_ptr,Batch*Channel*Height*Width*sizeof(float));
    cudaMalloc((void**) device_mask_ptr,Map_out*Channel*K*K*sizeof(float));
    cudaMalloc((void**) device_output_ptr,Batch*Map_out*H_out*W_out*sizeof(float));
    cudaMemcpy(*device_input_ptr,host_input,Batch*Channel*Height*Width*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr,host_mask,Map_out*Channel*K*K*sizeof(float),cudaMemcpyHostToDevice);
    
    
    
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
int H_out=Height-(K-1);
    int W_out=Width-(K-1);
    int Y=((W_out+16-1)/16)*((H_out+16-1)/16);
    dim3 blockdim(16,16,1);
    dim3 griddim(Map_out,Y,Batch);
    conv_forward_kernel<<<griddim,blockdim>>>(device_output,device_input,device_mask,Batch,Map_out,Channel,Height,Width,K);

   

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    int H_out=Height-(K-1);
    int W_out=Width-(K-1);
   cudaMemcpy(host_output,device_output,Batch*Map_out*H_out*W_out*sizeof(float),cudaMemcpyDeviceToHost);
   cudaFree(device_input);
    cudaFree(device_mask);
    cudaFree(device_output);

}









// #include <cmath>
// #include <iostream>
// #include "gpu-new-forward.h"
// #define TILE_WIDTH 16
// __global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.

//     Function paramter definitions:
//     output - output
//     input - input
//     mask - convolution kernel
//     Batch - batch_size (number of images in x)
//     Map_out - number of output feature maps
//     Channel - number of input feature maps
//     Height - input height dimension
//     Width - input width dimension
//     K - kernel height and width (K x K)
//     */

//     const int Height_out = Height - K + 1;
//     const int Width_out = Width - K + 1;
//      int WUnroll=Height_out*Width_out;
//      int CUnroll=Channel*K*K;
//      int tx=threadIdx.x;int ty=threadIdx.y;
//      int row=blockIdx.y*TILE_WIDTH+ty;
//      int col=blockIdx.x*TILE_WIDTH+tx;
//      int b=blockIdx.z;

//     // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     // An example use of these macros:
//     // float a = in_4d(0,0,0,0)
//     // out_4d(0,0,0,0) = a

//     #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
//     #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
//     #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     // Insert your GPU convolution kernel code here
//     __shared__ float Mmatrix[TILE_WIDTH][TILE_WIDTH];
//     __shared__ float Nmatrix[TILE_WIDTH][TILE_WIDTH];
//     float sum=0;
//     int h_out=col/Width_out;
//         int w_out=col%Width_out;
//     int num_tiles=(CUnroll+TILE_WIDTH-1)/TILE_WIDTH;
//     for(int tile=0;tile<num_tiles;tile++){
//         int mask_col=tile*TILE_WIDTH+tx;
//         if(row<Map_out&&mask_col<CUnroll){
//             int c=mask_col/(K*K);
//             int p=(mask_col%(K*K))/K;
//             int q=mask_col%K;
//             Mmatrix[ty][tx]=mask_4d(row,c,p,q);
//         }
//         else Mmatrix[ty][tx]=0;
//         int in_row=tile*TILE_WIDTH+ty;
//         if(in_row<CUnroll&&col<WUnroll){
//             int c=in_row/(K*K);
//             int p=(in_row%(K*K))/K;
//             int q=in_row%K;
//             int h_out=col/Width_out;
//             int w_out=col%Width_out;
//             Nmatrix[ty][tx]=in_4d(b,c,h_out+p,w_out+q);
//         }
//         else Nmatrix[ty][tx]=0;
//         __syncthreads();
        
//         //finally
//         for(int k=0;k<TILE_WIDTH;k++)sum+=Mmatrix[ty][k]*Nmatrix[k][tx];
//         __syncthreads();
//     }
//     if(row<Map_out&&col<WUnroll){
//         out_4d(b,row,h_out,w_out)=sum;
//     }

//     #undef out_4d
//     #undef in_4d
//     #undef mask_4d
// }

	
// __host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     // Allocate memory and copy over the relevant data structures to the GPU

//     // We pass double pointers for you to initialize the relevant device pointers,
//     //  which are passed to the other two functions.

//     // Useful snippet for error checking
//     int H_out=Height-(K-1);
//     int W_out=Width-(K-1);
//     int input_size=Batch*Channel*Height*Width*sizeof(float);
//     int output_size=Batch*Map_out*H_out*W_out*sizeof(float);
//     int mask_size=Map_out*Channel*K*K*sizeof(float);
//     cudaMalloc((void**)device_input_ptr,input_size);
//     cudaMalloc((void**)device_output_ptr,output_size);
//     cudaMalloc((void**)device_mask_ptr,mask_size);
//     cudaMemcpy(*device_input_ptr,host_input,input_size,cudaMemcpyHostToDevice);
//     cudaMemcpy(*device_mask_ptr,host_mask,mask_size,cudaMemcpyHostToDevice);
    
//     // cudaError_t error = cudaGetLastError();
//     // if(error != cudaSuccess)
//     // {
//     //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
//     //     exit(-1);
//     // }

// }


// __host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     // Set the kernel dimensions and call the kernel
// int H_out=Height-(K-1);
//     int W_out=Width-(K-1);
//     int unrollmatrix=H_out*W_out;
//     int block_x=(unrollmatrix+TILE_WIDTH-1)/TILE_WIDTH;
//     int block_y=(Map_out+TILE_WIDTH-1)/TILE_WIDTH;
//     dim3 blockdim(TILE_WIDTH,TILE_WIDTH,1);
//     dim3 griddim(block_x,block_y,Batch);
//     conv_forward_kernel<<<griddim,blockdim>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
//     cudaDeviceSynchronize();

// }


// __host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     // Copy the output back to host
//     cudaMemcpy(host_output,device_output,Batch*Map_out*(Height-K+1)*(Width-K+1)*sizeof(float),cudaMemcpyDeviceToHost);
//     // Free device memory
//     cudaFree(device_input);
//     cudaFree(device_output);
//     cudaFree(device_mask);

// }


// __host__ void GPUInterface::get_device_properties()
// {
//     int deviceCount;
//     cudaGetDeviceCount(&deviceCount);

//     for(int dev = 0; dev < deviceCount; dev++)
//     {
//         cudaDeviceProp deviceProp;
//         cudaGetDeviceProperties(&deviceProp, dev);

//         std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
//         std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
//         std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
//         std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
//         std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
//         std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
//         std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
//         std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
//         std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
//     }
// }



