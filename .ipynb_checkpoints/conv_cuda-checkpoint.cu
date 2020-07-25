
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

namespace forward {
    template<typename scalar_t>
    __device__ __forceinline__ scalar_t computeOutputPixel(
        torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits,size_t> input,
        torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits,size_t> weights,
        torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits,size_t> bias,
        size_t batchIdx, size_t channelIdx, size_t heightIdx, size_t widthIdx
    ) {
        scalar_t output = 0;
        for(size_t c = 0; c < input.size(1); c++) {
            for(size_t h = 0; h < weights.size(2); h++) {
                for(size_t w = 0; w < weights.size(3); w++) {
                    output += input[batchIdx][c][heightIdx + h][widthIdx + w] 
                            * weights[channelIdx][c][h][w];
                }
            }
        }
        output += bias[channelIdx];
        return output;
    }

    template<typename scalar_t>
    __global__ void convolution_cuda_forward_kernel(
        const torch::PackedTensorAccessor<scalar_t, 4 , torch::RestrictPtrTraits, size_t> input,
        const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> weights,
        const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> bias,
        torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> output
    ) {
        /*Input format
            * input (torch::Tensor): Shape (B, C, H, W)
            * weights (torch::Tensor): Shape (C_out, C_in, H, W)
            * bias (torch::Tensor): Shape (C_out, )
            * output (torch::Tensor): Shape (B, C, H, W)
        */
        const int totalThreads = gridDim.x * blockDim.x;
        const int globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
        
        /*Workload assignment
            1. Flatten the output tensor
            2. The work of computing output[n] is assigned to thread n%n_threads
        */
        for(size_t b = 0; b < output.size(0); b++){
            for(size_t c = 0; c < output.size(1); c++){
                for(size_t h = 0; h < output.size(2); h++){
                    for(size_t w = 0; w < output.size(3); w++){
                        size_t pixelFlattenedIdx = b*output.stride(0) + c*output.stride(1) 
                                                 + h*output.stride(2) + w*output.stride(3);
                        if(pixelFlattenedIdx % totalThreads == globalThreadIdx){
                            output[b][c][h][w] = computeOutputPixel(
                                input, weights, bias, b, c, h, w
                            );
                        }
                    }
                }
            }
        }
    }
}

namespace backward {
    template<typename scalar_t>
    __device__ __forceinline__ scalar_t computeGradWeightPixel(
        const torch::PackedTensorAccessor<scalar_t, 4 , torch::RestrictPtrTraits, size_t> gradOutput,
        const torch::PackedTensorAccessor<scalar_t, 4 , torch::RestrictPtrTraits, size_t> input,
        size_t channelOutIdx, size_t channelInIdx, size_t heightIdx, size_t widthIdx
    ) {
        scalar_t output = 0;
        for(size_t b = 0; b < gradOutput.size(0); b++){
            for(size_t h = 0; h < gradOutput.size(2); h++){
                for(size_t w = 0; w < gradOutput.size(3); w++){
                    output += gradOutput[b][channelOutIdx][h][w] 
                            * input[b][channelInIdx][h + heightIdx][w + widthIdx];
                }
            }
        }
        return output;
    }

    template<typename scalar_t>
    __global__ void convolution_cuda_backward_by_weights_kernel(
        const torch::PackedTensorAccessor<scalar_t, 4 , torch::RestrictPtrTraits, size_t> gradOutput,
        const torch::PackedTensorAccessor<scalar_t, 4 , torch::RestrictPtrTraits, size_t> input,
        torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> gradWeights
    ) {
        /*Input format
            * gradOutput (torch::Tensor): Shape (B, C, H, W)
            * input (torch::Tensor): Shape (B, C, H, W)
            * gradWeights (torch::Tensor): Shape (C_out, C_in, H, W)
        */
        const int totalThreads = gridDim.x * blockDim.x;
        const int globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
        
        /*Workload assignment for weight gradient computation
            1. Flatten the output tensor
            2. The work of computing output[n] is assigned to thread n%n_threads
        */
        for(size_t c_out = 0; c_out < gradWeights.size(0); c_out++){
            for(size_t c_in = 0; c_in < gradWeights.size(1); c_in++){
                for(size_t h = 0; h < gradWeights.size(2); h++){
                    for(size_t w = 0; w < gradWeights.size(3); w++){
                        size_t pixelFlattenedIdx = c_out*gradWeights.stride(0)
                                                 + c_in*gradWeights.stride(1)
                                                 + h*gradWeights.stride(2)
                                                 + w*gradWeights.stride(3);
                        if(pixelFlattenedIdx % totalThreads == globalThreadIdx){
                            gradWeights[c_out][c_in][h][w] = computeGradWeightPixel(
                                gradOutput, input, c_out, c_in, h, w
                            );
                        };
                    }
                }
            }
        }
    }

    template<typename scalar_t>
    __device__ __forceinline__ scalar_t computeGradInputPixel(
        const torch::PackedTensorAccessor<scalar_t, 4 , torch::RestrictPtrTraits, size_t> gradOutput,
        const torch::PackedTensorAccessor<scalar_t, 4 , torch::RestrictPtrTraits, size_t> weights,
        size_t batchIdx, size_t channelInIdx, size_t heightIdx, size_t widthIdx
    ) {
        scalar_t output = 0;
        size_t minH = 0, maxH = weights.size(2), minW = 0, maxW = weights.size(3);
        if(heightIdx + 1 < maxH) maxH = heightIdx + 1;
        if(heightIdx >= gradOutput.size(2) + minH) minH = heightIdx - gradOutput.size(2) + 1;
        if(widthIdx + 1 < maxW) maxW = widthIdx + 1;
        if(widthIdx >= gradOutput.size(3) + minW) minW = widthIdx - gradOutput.size(3) + 1;
        
        for(size_t c_out = 0; c_out < weights.size(0); c_out++){
            for(size_t h = minH; h < maxH; h++){
                for(size_t w = minW; w < maxW; w++){
                    output += weights[c_out][channelInIdx][h][w]
                            * gradOutput[batchIdx][c_out][heightIdx - h][widthIdx - w];
                }
            }
        }
        return output;
    }
        

    template<typename scalar_t>
    __global__ void convolution_cuda_backward_by_input_kernel(
        const torch::PackedTensorAccessor<scalar_t, 4 , torch::RestrictPtrTraits, size_t> gradOutput,
        const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> weights,
        torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> gradInput
    ) {
        /*Input format
            * gradOutput (torch::Tensor): Shape (B, C, H, W)
            * weights (torch::Tensor): Shape (C_out, C_in, H, W)
            * gradInput (torch::Tensor): Shape (B, C, H, W)
        */
        const int totalThreads = gridDim.x * blockDim.x;
        const int globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;

        /*Workload assignment for input gradient computation
            1. Flatten the output tensor
            2. The work of computing output[n] is assigned to thread n%n_threads
        */
        for(size_t b = 0; b < gradInput.size(0); b++){
            for(size_t c = 0; c < gradInput.size(1); c++){
                for(size_t h = 0; h < gradInput.size(2); h++){
                    for(size_t w = 0; w < gradInput.size(3); w++){
                        size_t pixelFlattenedIdx = b*gradInput.stride(0)
                                                 + c*gradInput.stride(1)
                                                 + h*gradInput.stride(2)
                                                 + w*gradInput.stride(3);
                        if(pixelFlattenedIdx % totalThreads == globalThreadIdx){
                            gradInput[b][c][h][w] = computeGradInputPixel(
                                gradOutput, weights, b, c, h, w
                            );
                        }
                    }
                }
            }
        }
    }
}

std::vector<torch::Tensor> convolution_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias
) {
    /* 
    Notes
        * The number of threads is not infinite. This should be about 128 - 512
            https://forums.developer.nvidia.com/t/how-to-choose-how-many-threads-blocks-to-have/55529/2
        * Output tensor to be created must be located within the same device as in other tensors
    Weird behavior
        * Required to return std::vector<torch::Tensor>
        * Another CUDA process pop up
        * Use torch::Tensor.stride(idx) is slower than computing the stride by hand
    */
    auto output = torch::zeros({
        input.size(0), weights.size(0), 
        input.size(2) - weights.size(2) + 1, input.size(3) - weights.size(3) + 1
    }, input.options());
    
    cudaSetDevice(0);
    const dim3 threads(128, 1, 1);
    const dim3 blocks(128, 1, 1);
    AT_DISPATCH_FLOATING_TYPES(input.type(), "convolution forward", ([&] {
        forward::convolution_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            weights.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            bias.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            output.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>()
        );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }));
    
    return {output};
}

std::vector<torch::Tensor> convolution_cuda_backward(
    torch::Tensor gradOutput,
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias
) {
    auto gradInput = torch::zeros_like(input);
    auto gradWeights = torch::zeros_like(weights);
    auto gradBias = torch::ones_like(bias) * gradOutput.size(0)
                  * gradOutput.size(2) * gradOutput.size(3);

    cudaSetDevice(0);
    const dim3 threads(128, 1, 1);
    const dim3 blocks(128, 1, 1);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "convolution backward by weights", ([&] {
        // Main code
        backward::convolution_cuda_backward_by_weights_kernel<scalar_t><<<blocks, threads>>>(
            gradOutput.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            input.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            gradWeights.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>()
        );
        // Debug
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }));

    AT_DISPATCH_FLOATING_TYPES(input.type(), "convolution backward by input", ([&] {
        // Main code
        backward::convolution_cuda_backward_by_input_kernel<scalar_t><<<blocks, threads>>>(
            gradOutput.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            weights.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            gradInput.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>()
        );
        // Debug
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }));

    return {gradInput, gradWeights, gradBias};
}