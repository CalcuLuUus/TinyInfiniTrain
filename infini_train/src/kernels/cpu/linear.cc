#include <cstdint>
#include <fcntl.h>
#include <memory>
#include <numeric>
#include <tuple>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    // =================================== 作业 ===================================
    // TODO：实现CPU上的矩阵乘法前向计算
    // REF:
    // =================================== 作业 ===================================
    CHECK(input);
    CHECK(other);
    CHECK_EQ(static_cast<int>(input->Dtype()), static_cast<int>(DataType::kFLOAT32));
    CHECK_EQ(static_cast<int>(other->Dtype()), static_cast<int>(DataType::kFLOAT32));

    const auto &a_dims = input->Dims();
    const auto &b_dims = other->Dims();
    CHECK_GE(a_dims.size(), 2);
    CHECK_EQ(a_dims.size(), b_dims.size());

    const int64_t ndim = static_cast<int64_t>(a_dims.size());
    for (int64_t i = 0; i < ndim - 2; ++i) {
        CHECK_EQ(a_dims[i], b_dims[i]) << "Batch dim mismatch at dim=" << i;
    }

    const int64_t M = a_dims[ndim - 2];
    const int64_t K = a_dims[ndim - 1];
    CHECK_EQ(K, b_dims[ndim - 2]);
    const int64_t N = b_dims[ndim - 1];

    std::vector<int64_t> out_dims = a_dims;
    out_dims[ndim - 1] = N;
    auto output = std::make_shared<Tensor>(out_dims, DataType::kFLOAT32);

    const int64_t batch
        = (ndim == 2) ? 1 : std::accumulate(a_dims.begin(), a_dims.end() - 2, 1LL, std::multiplies<int64_t>());

    const auto *a_ptr = static_cast<const float *>(input->DataPtr());
    const auto *b_ptr = static_cast<const float *>(other->DataPtr());
    auto *c_ptr = static_cast<float *>(output->DataPtr());

    using MatrixRM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using ConstMap = Eigen::Map<const MatrixRM>;
    using Map = Eigen::Map<MatrixRM>;

    const int64_t a_stride = M * K;
    const int64_t b_stride = K * N;
    const int64_t c_stride = M * N;

    for (int64_t b = 0; b < batch; ++b) {
        ConstMap A(a_ptr + b * a_stride, M, K);
        ConstMap B(b_ptr + b * b_stride, K, N);
        Map C(c_ptr + b * c_stride, M, N);
        C.noalias() = A * B;
    }

    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    // =================================== 作业 ===================================
    // TODO：实现CPU上的矩阵乘法反向传播
    // REF:
    // =================================== 作业 ===================================
    CHECK(input);
    CHECK(other);
    CHECK(grad_output);
    CHECK_EQ(static_cast<int>(input->Dtype()), static_cast<int>(DataType::kFLOAT32));
    CHECK_EQ(static_cast<int>(other->Dtype()), static_cast<int>(DataType::kFLOAT32));
    CHECK_EQ(static_cast<int>(grad_output->Dtype()), static_cast<int>(DataType::kFLOAT32));

    const auto &a_dims = input->Dims();
    const auto &b_dims = other->Dims();
    const auto &c_dims = grad_output->Dims();
    CHECK_GE(a_dims.size(), 2);
    CHECK_EQ(a_dims.size(), b_dims.size());
    CHECK_EQ(a_dims.size(), c_dims.size());

    const int64_t ndim = static_cast<int64_t>(a_dims.size());
    for (int64_t i = 0; i < ndim - 2; ++i) {
        CHECK_EQ(a_dims[i], b_dims[i]);
        CHECK_EQ(a_dims[i], c_dims[i]);
    }

    const int64_t M = a_dims[ndim - 2];
    const int64_t K = a_dims[ndim - 1];
    CHECK_EQ(K, b_dims[ndim - 2]);
    const int64_t N = b_dims[ndim - 1];

    CHECK_EQ(c_dims[ndim - 2], M);
    CHECK_EQ(c_dims[ndim - 1], N);

    auto grad_input = std::make_shared<Tensor>(a_dims, DataType::kFLOAT32);
    auto grad_other = std::make_shared<Tensor>(b_dims, DataType::kFLOAT32);

    const int64_t batch
        = (ndim == 2) ? 1 : std::accumulate(a_dims.begin(), a_dims.end() - 2, 1LL, std::multiplies<int64_t>());

    const auto *a_ptr = static_cast<const float *>(input->DataPtr());
    const auto *b_ptr = static_cast<const float *>(other->DataPtr());
    const auto *dc_ptr = static_cast<const float *>(grad_output->DataPtr());

    auto *da_ptr = static_cast<float *>(grad_input->DataPtr());
    auto *db_ptr = static_cast<float *>(grad_other->DataPtr());

    using MatrixRM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using ConstMap = Eigen::Map<const MatrixRM>;
    using Map = Eigen::Map<MatrixRM>;

    const int64_t a_stride = M * K;
    const int64_t b_stride = K * N;
    const int64_t c_stride = M * N;

    for (int64_t b = 0; b < batch; ++b) {
        ConstMap A(a_ptr + b * a_stride, M, K);
        ConstMap B(b_ptr + b * b_stride, K, N);
        ConstMap dC(dc_ptr + b * c_stride, M, N);

        Map dA(da_ptr + b * a_stride, M, K);
        Map dB(db_ptr + b * b_stride, K, N);

        dA.noalias() = dC * B.transpose();
        dB.noalias() = A.transpose() * dC;
    }

    return {grad_input, grad_other};
}

std::shared_ptr<Tensor> LinearForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      bool transpose, const std::shared_ptr<Tensor> &bias) {
    /*
    transpose:  output = input * weight^T + bias
    output[*, out_features] = input[*, in_features] * weight[out_features, in_features]^T + bias[out_features]

    !transpose: output = input * weight + bias
    output[*, out_features] = input[*, in_features] * weight[in_features, out_features] + bias[out_features]
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    const int out_features = weight_dims[transpose ? 0 : 1];

    if (bias) {
        const auto &bias_dims = bias->Dims();
        CHECK_EQ(bias_dims.size(), 1);
        CHECK_EQ(bias_dims[0], out_features);
    }

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);

    if (transpose) {
        output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix().transpose();
    } else {
        output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix();
    }

    if (bias) {
        output->EigenMatrix().rowwise() += bias->EigenVector();
    }

    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LinearBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight, bool transpose,
               int64_t out_features, const std::shared_ptr<Tensor> &grad_output, const bool bias) {
    /*
    transpose: grad_input = grad_output * weight
    grad_input[*, in_features] = grad_output[*, out_features] * weight[out_features, in_features]
    grad_weight[out_features, in_features] = grad_output[*, out_features]^T * input[*, in_features]
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)

    !transpose: grad_input = grad_output * weight^T
    grad_input[*, in_features] = grad_output[_, out_features] * weight[in_features, out_features]^T
    grad_weight[in_features, out_features] = input[*, in_features]^T * grad_output[*, out_features]
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    CHECK_EQ(out_features, weight_dims[transpose ? 0 : 1]);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32);
    std::shared_ptr<Tensor> grad_bias = nullptr;
    if (bias) {
        grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, DataType::kFLOAT32);
    }

    if (transpose) {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix();
        grad_weight->EigenMatrix() = grad_output->EigenMatrix().transpose() * input->EigenMatrix();
    } else {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix().transpose();
        grad_weight->EigenMatrix() = input->EigenMatrix().transpose() * grad_output->EigenMatrix();
    }
    if (bias) {
        grad_bias->EigenVector() = grad_output->EigenMatrix().colwise().sum();
    }

    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_LINEAR_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_LINEAR_KERNEL(MatmulForward)
REGISTER_CPU_LINEAR_KERNEL(MatmulBackward)
REGISTER_CPU_LINEAR_KERNEL(LinearForward)
REGISTER_CPU_LINEAR_KERNEL(LinearBackward)

#undef REGISTER_CPU_LINEAR_KERNEL
