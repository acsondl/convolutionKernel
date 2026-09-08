#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include "matmul.h"

#define PERMUTE_BLOCK_SIZE 256

__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K) {
    /*
    Modify this function to implement the input matrix unrolling kernel.

    Function paramter definitions:
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
   size_t heightunroll=Channel*K*K;
   size_t widthunroll=Batch*Height_out*Width_out;
   size_t col=threadIdx.x+blockIdx.x*blockDim.x;
   size_t row=threadIdx.y+blockIdx.y*blockDim.y;//
    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]

    // TODO: Insert your input matrix unrolling kernel code here
    if(col<widthunroll&&row<heightunroll){
        size_t c=row/K/K;//see which input channel
        size_t channeloffset=row%(K*K);
        size_t p=channeloffset/K;
        size_t q=channeloffset%K;//which is the conv mask multiplying
        
        size_t b=col/Height_out/Width_out;//what batch
        size_t batchoffset=col%(Height_out*Width_out);
        size_t houtput=batchoffset/Width_out;
        size_t woutput=batchoffset%Width_out;
        size_t hin=houtput+p;
        size_t win=woutput+q;
        output[row*widthunroll+col]= in_4d(b,c,hin,win);
        
    }
    

    #undef in_4d
}


// Permutes the matmul result.
// The output feature map after matmul is of shape Map_out x Batch x Height_out x Width_out,
// and we need to permute it into Batch x Map_out x Height_out x Width_out.
// You don't need to modify this kernel.
__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
                                      int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    input[m * Batch * image_size + b * image_size + x];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU
  //left unchanged define all the constant and malloc   
    int Height_out=Height-K+1;
    int Width_out=Width-K+1;
    size_t input_size=(size_t)Batch*Channel*Height*Width*sizeof(float);
    size_t output_size=(size_t)Batch*Map_out*Height_out*Width_out*sizeof(float);
    size_t mask_size=(size_t)K*K*Channel*Map_out*sizeof(float);
    cudaMalloc((void**)device_input_ptr,input_size);
    cudaMalloc((void**)device_output_ptr,output_size);
    cudaMalloc((void**)device_mask_ptr,mask_size);

//pinning and copy the host memmory
    cudaHostRegister((void*)host_input,input_size,cudaHostRegisterDefault);
    cudaHostRegister((void*)host_output,output_size,cudaHostRegisterDefault);

//copy the mask
    cudaMemcpy(*device_mask_ptr,host_mask,mask_size,cudaMemcpyHostToDevice);

//initialize the 4 streams
    int num_streams=4;
    cudaStream_t streams[num_streams];
    for(int i=0;i<num_streams;i++)cudaStreamCreate(&streams[i]);

    //divide into 4 chunks for streams and define neccesary variables
    int chunk_size=(Batch+num_streams-1)/num_streams; 
    int Height_unrolled=Channel*K*K;
    int max_Width_unrolled_chunk=chunk_size*Height_out*Width_out;

    //allocate memory for first kernel launch(matmul)
    float *unrolled_matrix;// Pointer to device memory for storing the unrolled matrix
    float *matmul_output;// Pointer to device memory for storing the result of matrix multiplication
    cudaMalloc((void**)&unrolled_matrix,(size_t)num_streams*Height_unrolled*max_Width_unrolled_chunk*sizeof(float));
    cudaMalloc((void**)&matmul_output,(size_t)num_streams*chunk_size*Map_out*Height_out*Width_out*sizeof(float));

//go through the 4 streams and figure out what to put in for each of the 4 streams, first divide the input into 4 segment then copy to stream
    for(int i=0;i<num_streams;i++){
        int current_chunk=std::min(chunk_size,Batch-i*chunk_size);
        if(current_chunk<=0) continue;
        size_t in_offset=(size_t)i*chunk_size*Channel*Height*Width;//devide the input into 4 segment each with i*chunksize
        //copy to stream
        cudaMemcpyAsync((*device_input_ptr)+in_offset,host_input+in_offset,(size_t)current_chunk*Channel*Height*Width*sizeof(float),cudaMemcpyHostToDevice,streams[i]);
    }
    
    //launch the kernels for streams
    for(int i=0;i<num_streams;i++){
        int current_chunk=std::min(chunk_size,Batch-i*chunk_size);
        if(current_chunk<=0) continue;
        
        size_t in_offset=(size_t)i*chunk_size*Channel*Height*Width;
        size_t out_offset=(size_t)i*chunk_size*Map_out*Height_out*Width_out;
        size_t unroll_offset=(size_t)i*Height_unrolled*max_Width_unrolled_chunk;
        int current_width_unrolled=current_chunk*Height_out*Width_out;

        //launch the unroll kernel
        dim3 blockdim(16,16,1);
        dim3 griddim((current_width_unrolled+15)/16,(Height_unrolled+15)/16,1);
        matrix_unrolling_kernel<<<griddim,blockdim,0,streams[i]>>>((*device_input_ptr)+in_offset,unrolled_matrix+unroll_offset,current_chunk,Channel,Height,Width,K);

        //launch the matmul kernel
        dim3 matmul_grid_dim((current_width_unrolled-1)/16+1,(Map_out-1)/16+1,1);
        dim3 matmul_block_dim(16,16,1);
        matrixMultiplyShared<<<matmul_grid_dim,matmul_block_dim,0,streams[i]>>>(*device_mask_ptr,unrolled_matrix+unroll_offset,matmul_output+out_offset,Map_out,Height_unrolled,Height_unrolled,current_width_unrolled,Map_out,current_width_unrolled);

        //launch the permutation kernel
        const int out_image_size=Height_out*Width_out;
        dim3 permute_grid_dim((out_image_size-1)/PERMUTE_BLOCK_SIZE+1,current_chunk,1);
        matrix_permute_kernel<<<permute_grid_dim,PERMUTE_BLOCK_SIZE,0,streams[i]>>>(matmul_output+out_offset,(*device_output_ptr)+out_offset,Map_out,current_chunk,out_image_size);
    }

    //copy result to pinned memory output in streams
    for(int i=0;i<num_streams;i++){
        int current_chunk=std::min(chunk_size,Batch-i*chunk_size);
        if(current_chunk<=0) continue;
        size_t out_offset=(size_t)i*chunk_size*Map_out*Height_out*Width_out;
        cudaMemcpyAsync((float*)host_output+out_offset,(*device_output_ptr)+out_offset,(size_t)current_chunk*Map_out*Height_out*Width_out*sizeof(float),cudaMemcpyDeviceToHost,streams[i]);
    }

    cudaDeviceSynchronize();
    
    //free the streams, device pointers,pinned memory
    for(int i=0;i<num_streams;i++){
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(unrolled_matrix);
    cudaFree(matmul_output);

    //Free the pinned host memory
    cudaHostUnregister((void*)host_input);
    cudaHostUnregister((void*)host_output);
    //////////////////////////////////////////////////////////////////////


    

  

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
  
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
cudaMemcpy(host_output,device_output,Batch*(Height-K+1)*(Width-K+1)*Map_out*sizeof(float),cudaMemcpyDeviceToHost);
    // TODO: Free device memory
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