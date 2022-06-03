//
// Created by locks on 02.06.2022.
//

#ifndef CUDA_DENOISE_METHODS_H
#define CUDA_DENOISE_METHODS_H

#define CHECK_BOUNDS(y, x, h, w) \
    ((x) < 0 || (y) < 0 || (x) >= (w) || (y) >= (h))

void parseCudaResult(std::string label, cudaError_t res) {
    if (res) {
        std::cout << label << ": " << cudaGetErrorString(res) << std::endl;
    }
}

__global__ void zero_pad(
        const float* src,
        const int src_h,
        const int src_w,
        const int src_c,
        float* dst,
        const int dst_h,
        const int dst_w,
        const int dst_c
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int xoffset = (dst_h - src_h) / 2;
    int yoffset = (dst_w - src_w) / 2;
    if (CHECK_BOUNDS(i, j, src_h, src_w) || CHECK_BOUNDS(i + yoffset, j + xoffset, dst_h, dst_w)) {
        return;
    }

    for (int k = 0; k < dst_c; ++k) {
        dst[(i + yoffset) * dst_w * dst_c + (j + xoffset) * dst_c + k] = src[i * src_w * src_c + j * src_c + k];
    }
}

__global__ void convkxk(
        const float* src,
        const int src_h,
        const int src_w,
        const int src_c,
        const float* kernel,
        const float* bias,
        const int kh,
        const int kw,
        float* dst,
        const int dst_h,
        const int dst_w,
        const int dst_c,
        const int relu
) {
    // Using channels-last format (H, W, C)
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (CHECK_BOUNDS(i, j, dst_h, dst_w)) {
        return;
    }

    float* dst_ = dst + i * dst_w * dst_c + j * dst_c;
    const float* src_ = src + i * src_w * src_c + j * src_c;

    for (int oc = 0; oc < dst_c; ++oc) {
        float tmp = 0;
        for (int k = 0; k < kh; ++k) {
            for (int l = 0; l < kw; ++l) {
                for (int n = 0; n < src_c; ++n) {
                    tmp += *kernel++ * src_[k * src_w * src_c + l * src_c + n];
                }
            }
        }
        dst_[oc] = tmp + bias[oc];
        dst_[oc] = relu * MAX(dst_[oc], 0) + (1 - relu) * dst_[oc];
    }
}

__global__ void maxpool2d2x2stride2(
        const float* src,
        const int src_h,
        const int src_w,
        const int src_c,
        float* dst,
        const int dst_h,
        const int dst_w,
        const int dst_c
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (CHECK_BOUNDS(i, j, dst_h, dst_w)) {
        return;
    }

    float* dst_ = dst + i * dst_w * dst_c + j * dst_c;
    const float* src_ = src + 2 * i * src_w * src_c + 2 * j * src_c;

    for (int oc = 0; oc < dst_c; ++oc) {
        dst_[oc] = MAX(MAX(src_[oc], src_[src_c + oc]), MAX(src_[src_w * src_c + oc], src_[src_w * src_c + src_c + oc]));
    }
}

__global__ void transposepad(
        const float* src,
        const int src_h,
        const int src_w,
        const int src_c,
        float* dst,
        const int dst_h,
        const int dst_w,
        const int dst_c
        ) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (CHECK_BOUNDS(i, j, src_h, src_w)) {
        return;
    }

    float *dst_ = dst + (i * 2 + 2) * dst_w * dst_c + (j * 2 + 2) * dst_c;
    const float *src_ = src + i * src_w * src_c + j * src_c;

    for (int oc = 0; oc < dst_c; ++oc) {
        dst_[oc] = src_[oc];
    }
}

__global__ void sigmoid(
        const float *src,
        const int src_h,
        const int src_w,
        const int src_c,
        float *dst,
        const int dst_h,
        const int dst_w,
        const int dst_c
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (CHECK_BOUNDS(i, j, dst_h, dst_w)) {
        return;
    }

    const float* src_ = src + i * src_w * src_c + j * src_c;
    float* dst_ = dst + i * dst_w * dst_c + j * dst_c;

    for (int oc = 0; oc < dst_c; ++oc) {
        dst_[oc] = 1 / (1 + expf(-src_[oc]));
    }
}

__global__ void relu(
        const float* src,
        const int src_h,
        const int src_w,
        const int src_c,
        float* dst,
        const int dst_h,
        const int dst_w,
        const int dst_c
        ) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (CHECK_BOUNDS(i, j, dst_h, dst_w)) {
        return;
    }

    const float* src_ = src + i * src_w * src_c + j * src_c;
    float* dst_ = dst + i * dst_w * dst_c + j * dst_c;

    for (int oc = 0; oc < dst_c; ++oc) {
        dst_[oc] = MAX(0, src_[oc]);
    }
}

#endif //CUDA_DENOISE_METHODS_H
