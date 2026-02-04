#include <cmath>
#include <cstddef>
#include <memory>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    for (int64_t idx = 0; idx < gradient->NumElements(); ++idx) {
        static_cast<float *>(tensor->DataPtr())[idx] += rate * static_cast<const float *>(gradient->DataPtr())[idx];
    }
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

    const float beta1_pow_t = static_cast<float>(std::pow(static_cast<double>(beta1), static_cast<double>(t)));
    const float beta2_pow_t = static_cast<float>(std::pow(static_cast<double>(beta2), static_cast<double>(t)));
    const float bc1 = 1.0f - beta1_pow_t;
    const float bc2 = 1.0f - beta2_pow_t;
    CHECK_NE(bc1, 0.0f);
    CHECK_NE(bc2, 0.0f);

    const float *g_ptr = static_cast<const float *>(grad->DataPtr());
    float *p_ptr = static_cast<float *>(param->DataPtr());
    float *m_ptr = static_cast<float *>(m->DataPtr());
    float *v_ptr = static_cast<float *>(v->DataPtr());

    const float one_minus_beta1 = 1.0f - beta1;
    const float one_minus_beta2 = 1.0f - beta2;

    for (int64_t idx = 0; idx < grad->NumElements(); ++idx) {
        const float g = g_ptr[idx];
        const float mi = beta1 * m_ptr[idx] + one_minus_beta1 * g;
        const float vi = beta2 * v_ptr[idx] + one_minus_beta2 * g * g;
        m_ptr[idx] = mi;
        v_ptr[idx] = vi;

        const float m_hat = mi / bc1;
        const float v_hat = vi / bc2;
        p_ptr[idx] -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
    }
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                               \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CPU_ACCUMULATE_GRAD_KERNEL
