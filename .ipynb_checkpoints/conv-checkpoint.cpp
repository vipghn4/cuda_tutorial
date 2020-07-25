
#include <torch/extension.h>
#include <vector>
#include <iostream>

/*CUDA declarations*/
std::vector<torch::Tensor> convolution_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias
);

std::vector<torch::Tensor> convolution_cuda_backward(
    torch::Tensor gradOutput,
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias
);

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> convolution_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weights);
    CHECK_INPUT(bias);
    return convolution_cuda_forward(input, weights, bias);
}

std::vector<torch::Tensor> convolution_backward(
    torch::Tensor gradOutput,
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias
) {
    CHECK_INPUT(gradOutput);
    CHECK_INPUT(input);
    CHECK_INPUT(weights);
    CHECK_INPUT(bias);
    return convolution_cuda_backward(gradOutput, input, weights, bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &convolution_forward, "Convolution forward CUDA");
    m.def("backward", &convolution_backward, "Convolution backward CUDA");
}