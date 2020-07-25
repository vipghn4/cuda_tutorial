
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="convolution_cuda",
    ext_modules=[
        CUDAExtension("conv_cuda", [
            "conv.cpp",
            "conv_cuda.cu",
        ]),
    ],
    cmdclass={"build_ext": BuildExtension}
)