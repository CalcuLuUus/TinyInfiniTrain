#include "example/common/tiny_shakespeare_dataset.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace {
using DataType = infini_train::DataType;
using TinyShakespeareType = TinyShakespeareDataset::TinyShakespeareType;
using TinyShakespeareFile = TinyShakespeareDataset::TinyShakespeareFile;

const std::unordered_map<int, TinyShakespeareType> kTypeMap = {
    {20240520, TinyShakespeareType::kUINT16}, // GPT-2
    {20240801, TinyShakespeareType::kUINT32}, // LLaMA 3
};

const std::unordered_map<TinyShakespeareType, size_t> kTypeToSize = {
    {TinyShakespeareType::kUINT16, 2},
    {TinyShakespeareType::kUINT32, 4},
};

const std::unordered_map<TinyShakespeareType, DataType> kTypeToDataType = {
    {TinyShakespeareType::kUINT16, DataType::kUINT16},
    {TinyShakespeareType::kUINT32, DataType::kINT32},
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

TinyShakespeareFile ReadTinyShakespeareFile(const std::string &path, size_t sequence_length) {
    /* =================================== 作业 ===================================
       TODO：实现二进制数据集文件解析
       文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | DATA (tokens)                        |
    | magic(4B) | version(4B) | num_toks(4B) | reserved(1012B) | token数据           |
    ----------------------------------------------------------------------------------
       =================================== 作业 =================================== */
    if (!std::filesystem::exists(path)) {
        LOG(FATAL) << "File not found: " << path;
    }
    CHECK_GT(sequence_length, 0);

    std::ifstream ifs(path, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open dataset file: " << path;

    const auto header = ReadSeveralBytesFromIfstream(1024, &ifs);
    CHECK_GE(header.size(), 12);

    const auto magic = BytesToType<uint32_t>(header, 0);
    const auto version = BytesToType<uint32_t>(header, 4);
    (void)version;
    const auto num_toks = BytesToType<uint32_t>(header, 8);
    CHECK_GT(num_toks, 0) << "Empty dataset: " << path;

    const auto type_it = kTypeMap.find(static_cast<int>(magic));
    CHECK(type_it != kTypeMap.end()) << "Unknown dataset magic: " << magic;
    const auto type = type_it->second;
    const auto elem_size = kTypeToSize.at(type);

    const uint64_t total_bytes = static_cast<uint64_t>(num_toks) * elem_size;
    std::vector<uint8_t> raw(static_cast<size_t>(total_bytes));
    ifs.read(reinterpret_cast<char *>(raw.data()), static_cast<std::streamsize>(total_bytes));
    CHECK_EQ(static_cast<uint64_t>(ifs.gcount()), total_bytes) << "Failed to read token data from: " << path;

    const int64_t num_chunks = static_cast<int64_t>(num_toks) / static_cast<int64_t>(sequence_length);
    CHECK_GE(num_chunks, 2) << "Dataset too small for seq_len=" << sequence_length << ": num_toks=" << num_toks;

    std::vector<int64_t> dims{num_chunks, static_cast<int64_t>(sequence_length)};
    auto tensor = infini_train::Tensor(dims, DataType::kINT64);
    auto *dst = static_cast<int64_t *>(tensor.DataPtr());

    const int64_t to_copy = num_chunks * static_cast<int64_t>(sequence_length);
    if (type == TinyShakespeareType::kUINT16) {
        CHECK_EQ(elem_size, sizeof(uint16_t));
        for (int64_t i = 0; i < to_copy; ++i) {
            uint16_t tok;
            std::memcpy(&tok, raw.data() + static_cast<size_t>(i) * sizeof(uint16_t), sizeof(uint16_t));
            dst[i] = static_cast<int64_t>(tok);
        }
    } else if (type == TinyShakespeareType::kUINT32) {
        CHECK_EQ(elem_size, sizeof(uint32_t));
        for (int64_t i = 0; i < to_copy; ++i) {
            uint32_t tok;
            std::memcpy(&tok, raw.data() + static_cast<size_t>(i) * sizeof(uint32_t), sizeof(uint32_t));
            dst[i] = static_cast<int64_t>(tok);
        }
    } else {
        LOG(FATAL) << "Unsupported dataset token type";
    }

    return TinyShakespeareFile{.type = type, .dims = std::move(dims), .tensor = std::move(tensor)};
}
} // namespace

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length)
    : text_file_(ReadTinyShakespeareFile(filepath, sequence_length)),
      sequence_length_(sequence_length),
      sequence_size_in_bytes_(sequence_length * sizeof(int64_t)),
      num_samples_(static_cast<size_t>(text_file_.dims[0] - 1)) {
    // =================================== 作业 ===================================
    // TODO：初始化数据集实例
    // HINT: 调用ReadTinyShakespeareFile加载数据文件
    // =================================== 作业 ===================================
    CHECK_EQ(text_file_.dims.size(), 2);
    CHECK_EQ(text_file_.dims[1], static_cast<int64_t>(sequence_length_));
    CHECK_GE(text_file_.dims[0], 2);
}

std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
TinyShakespeareDataset::operator[](size_t idx) const {
    CHECK_LT(idx, text_file_.dims[0] - 1);
    std::vector<int64_t> dims = std::vector<int64_t>(text_file_.dims.begin() + 1, text_file_.dims.end());
    // x: (seq_len), y: (seq_len) -> stack -> (bs, seq_len) (bs, seq_len)
    return {std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_, dims),
            std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_ + sizeof(int64_t),
                                                   dims)};
}

size_t TinyShakespeareDataset::Size() const { return num_samples_; }
