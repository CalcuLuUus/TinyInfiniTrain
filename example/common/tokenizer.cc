#include "example/common/tokenizer.h"

#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef USE_CUDA
#include "cuda_runtime_api.h"
#endif

#include "glog/logging.h"

namespace infini_train {

constexpr uint32_t kGpt2Eot = 50256;
constexpr uint32_t kLLaMA3Eot = 128001;
constexpr uint64_t kRandomU32Multiplier = 0x2545F4914F6CDD1Dull;
constexpr float kF32Divisor = 16777216.0f; // 2^24
constexpr uint64_t kRngState = 1337;

using Version = Tokenizer::Version;

const std::unordered_map<uint32_t, uint32_t> kEotMap = {
    {20240328, kGpt2Eot},   // GPT-2
    {20240801, kLLaMA3Eot}, // LLaMA-3
};

const std::unordered_map<uint32_t, std::vector<uint32_t>> kPromptMap = {
    // e.g. "The meaning of life is"
    // ref: https://tiktokenizer.vercel.app/
    {20240328, std::vector<uint32_t>{464, 3616, 286, 1204, 318}}, // GPT-2
    {20240801, std::vector<uint32_t>{791, 7438, 315, 2324, 374}}, // LLaMA-3
};

std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs) {
    std::vector<uint8_t> result(num_bytes);
    ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
    return result;
}

template <typename T> T BytesToType(const std::vector<uint8_t> &bytes, size_t offset) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    T value;
    std::memcpy(&value, &bytes[offset], sizeof(T));
    return value;
}

unsigned int RandomU32(uint64_t &state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (state * kRandomU32Multiplier) >> 32;
}

float RandomF32(uint64_t &state) { // random float32 in [0,1)
    return (RandomU32(state) >> 8) / kF32Divisor;
}

int SampleMult(float *probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from RandomF32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

Tokenizer::Tokenizer(const std::string &filepath) {
    /* ===================================== 作业 =====================================
    TODO：实现Tokenizer二进制文件加载

    文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | VOCAB TABLE                           |
    | magic(4B) | version(4B) | vocab_size(4B) | reserved(1012B) | token词表数据       |
    ----------------------------------------------------------------------------------
    ===================================== 作业 ===================================== */
    if (!std::filesystem::exists(filepath)) {
        LOG(FATAL) << "File not found: " << filepath;
    }
    std::ifstream ifs(filepath, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open tokenizer file: " << filepath;

    const auto header = ReadSeveralBytesFromIfstream(1024, &ifs);
    CHECK_GE(header.size(), 12);

    magic_number_ = BytesToType<uint32_t>(header, 0);
    const auto version_u32 = BytesToType<uint32_t>(header, 4);
    vocab_size_ = BytesToType<uint32_t>(header, 8);
    CHECK_GT(vocab_size_, 0);

    CHECK(kEotMap.find(magic_number_) != kEotMap.end()) << "Unknown tokenizer magic: " << magic_number_;
    eot_token_ = kEotMap.at(magic_number_);

    token_table_.clear();
    token_table_.reserve(vocab_size_);

    for (uint32_t i = 0; i < vocab_size_; ++i) {
        uint32_t len = 0;
        if (version_u32 == static_cast<uint32_t>(Version::kV1)) {
            uint16_t l16 = 0;
            ifs.read(reinterpret_cast<char *>(&l16), sizeof(uint16_t));
            CHECK(ifs.good()) << "Tokenizer file truncated while reading token length";
            len = static_cast<uint32_t>(l16);
        } else if (version_u32 == static_cast<uint32_t>(Version::kV2)) {
            ifs.read(reinterpret_cast<char *>(&len), sizeof(uint32_t));
            CHECK(ifs.good()) << "Tokenizer file truncated while reading token length";
        } else {
            LOG(FATAL) << "Unsupported tokenizer version: " << version_u32;
        }

        std::string token(len, '\0');
        if (len > 0) {
            ifs.read(token.data(), static_cast<std::streamsize>(len));
            CHECK(ifs.good()) << "Tokenizer file truncated while reading token bytes";
        }
        token_table_.push_back(std::move(token));
    }
}

std::string Tokenizer::Decode(uint32_t token_id) const {
    /* ===================================== 作业 =====================================
    TODO：实现token_id到文本的转换
    功能描述：根据token_id返回对应的文本片段
    ===================================== 作业 ===================================== */
    CHECK_LT(token_id, token_table_.size());
    return token_table_[token_id];
}

void Tokenizer::GenerateText(infini_train::nn::Module &model, uint32_t batch_size, uint32_t sequence_length,
                             uint32_t text_length, Device device) const {
    std::vector<int64_t> dims;
    dims.assign({batch_size, sequence_length});
    // x_tensor (FLAGS_batch_size, FLAGS_sequence_length) eq:(4, 64)
    infini_train::Tensor x_tensor = infini_train::Tensor(dims, DataType::kINT64);
    int64_t *x_buff = static_cast<int64_t *>(x_tensor.DataPtr());
    for (int i = 0; i < batch_size * sequence_length; ++i) { x_buff[i] = eot_token_; }

    // Give some contexts: "The meaning of life is "
    auto prompt = kPromptMap.at(magic_number_);
    auto prompt_len = prompt.size();
    for (int i = 0; i < prompt_len; ++i) { x_buff[i] = prompt[i]; }
    std::cout << "The meaning of life is";

    auto x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
    uint64_t rng_state = kRngState;
    LOG(INFO) << "start generate text:";
    for (int t = prompt_len; t < text_length; t++) {
        /* ===================================== 作业 =====================================
        TODO：实现单步文本生成逻辑
        HINT：调用model.Forward推理获取logits，根据推理结果进行随机采样，调用Decode获取文本结果
        ===================================== 作业 ===================================== */
        auto logits = model.Forward({x})[0];
        CHECK(logits);
        CHECK_EQ(logits->Dims().size(), 3);
        CHECK_EQ(logits->Dims()[0], batch_size);
        CHECK_EQ(logits->Dims()[1], sequence_length);

        // Read logits on CPU for sampling.
        auto cpu_logits = logits->To(Device(DeviceType::kCPU, 0));
#ifdef USE_CUDA
        if (device.Type() == DeviceType::kCUDA) {
            cudaDeviceSynchronize();
        }
#endif
        const float *logits_ptr = static_cast<const float *>(cpu_logits.DataPtr());
        const int64_t vocab_size = cpu_logits.Dims()[2];
        CHECK_GT(vocab_size, 0);

        // Sample from batch 0, position (t-1).
        const int64_t pos = static_cast<int64_t>(t - 1);
        const float *row = logits_ptr + (pos * vocab_size);

        // Softmax on CPU.
        float max_logit = row[0];
        for (int64_t i = 1; i < vocab_size; ++i) {
            if (row[i] > max_logit) {
                max_logit = row[i];
            }
        }
        std::vector<float> probs(static_cast<size_t>(vocab_size));
        float sum = 0.0f;
        for (int64_t i = 0; i < vocab_size; ++i) {
            const float v = std::exp(row[i] - max_logit);
            probs[static_cast<size_t>(i)] = v;
            sum += v;
        }
        CHECK_GT(sum, 0.0f);
        for (int64_t i = 0; i < vocab_size; ++i) {
            probs[static_cast<size_t>(i)] /= sum;
        }

        const int next_token = SampleMult(probs.data(), static_cast<int>(vocab_size), RandomF32(rng_state));
        x_buff[t] = static_cast<int64_t>(next_token);

        // Update device tensor for next step.
        x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));

        std::cout << Decode(static_cast<uint32_t>(next_token));
    }
    std::cout << std::endl;
}
} // namespace infini_train
