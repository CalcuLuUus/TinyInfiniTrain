#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include <cmath>

namespace infini_train::kernels::cuda {

__global__ void AccumulateGradKernel(const float *grad_ptr, float rate, float *tensor_ptr, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        tensor_ptr[idx] += rate * grad_ptr[idx];
    }
}

__global__ void AdamAccumulateGradKernel(const float *grad_ptr, float *param_ptr, float *m_ptr, float *v_ptr,
                                        size_t num_elements, float learning_rate, float beta1, float beta2, float eps,
                                        float bc1, float bc2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) {
        return;
    }

    const float g = grad_ptr[idx];
    const float one_minus_beta1 = 1.0f - beta1;
    const float one_minus_beta2 = 1.0f - beta2;

    const float mi = beta1 * m_ptr[idx] + one_minus_beta1 * g;
    const float vi = beta2 * v_ptr[idx] + one_minus_beta2 * g * g;
    m_ptr[idx] = mi;
    v_ptr[idx] = vi;

    const float m_hat = mi / bc1;
    const float v_hat = vi / bc2;
    param_ptr[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + eps);
}

void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    size_t num_elements = gradient->NumElements();

    const float *grad_ptr = static_cast<const float *>(gradient->DataPtr());
    float *tensor_ptr = static_cast<float *>(tensor->DataPtr());

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AccumulateGradKernel<<<num_blocks, threads_per_block>>>(grad_ptr, rate, tensor_ptr, num_elements);
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== 作业 ===================================
    // TODO：实现Adam优化器的梯度累积和参数更新
    // REF:
    // =================================== 作业 ===================================
    CHECK(grad);
    CHECK(param);
    CHECK(m);
    CHECK(v);
    CHECK_EQ(static_cast<int>(grad->Dtype()), static_cast<int>(DataType::kFLOAT32));
    CHECK_EQ(static_cast<int>(param->Dtype()), static_cast<int>(DataType::kFLOAT32));
    CHECK_EQ(static_cast<int>(m->Dtype()), static_cast<int>(DataType::kFLOAT32));
    CHECK_EQ(static_cast<int>(v->Dtype()), static_cast<int>(DataType::kFLOAT32));
    CHECK_EQ(grad->NumElements(), param->NumElements());
    CHECK_EQ(grad->NumElements(), m->NumElements());
    CHECK_EQ(grad->NumElements(), v->NumElements());
    CHECK_GE(t, 1);

    const float bc1 = 1.0f - powf(beta1, static_cast<float>(t));
    const float bc2 = 1.0f - powf(beta2, static_cast<float>(t));
    CHECK_NE(bc1, 0.0f);
    CHECK_NE(bc2, 0.0f);

    size_t num_elements = grad->NumElements();
    const float *grad_ptr = static_cast<const float *>(grad->DataPtr());
    float *param_ptr = static_cast<float *>(param->DataPtr());
    float *m_ptr = static_cast<float *>(m->DataPtr());
    float *v_ptr = static_cast<float *>(v->DataPtr());

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    AdamAccumulateGradKernel<<<num_blocks, threads_per_block>>>(grad_ptr, param_ptr, m_ptr, v_ptr, num_elements,
                                                                learning_rate, beta1, beta2, eps, bc1, bc2);
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                              \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL
