#include "framelog.hpp"

#include <cstddef>
#include <cstring>
#include <type_traits>

namespace spark {

namespace {

template <class T> constexpr const char *typeName() {
    if constexpr (std::is_same_v<T, float>) return "f32";
    else if constexpr (std::is_same_v<T, uint32_t>) return "u32";
    else if constexpr (std::is_same_v<T, uint16_t>) return "u16";
    else if constexpr (std::is_same_v<T, uint8_t>) return "u8";
    else static_assert(sizeof(T) == 0, "unsupported field type");
}

const FieldDesc kFields[] = {
#define SPARK_X(type, name, unit, when, desc) {#name, typeName<type>(), unit, when, desc, offsetof(FrameRecord, name), sizeof(type)},
    SPARK_FRAME_FIELDS(SPARK_X)
#undef SPARK_X
};

constexpr size_t kBufferRows = 4096;

} // namespace

std::span<const FieldDesc> frameFields() { return kFields; }

std::string jsonEscape(const std::string &s) {
    std::string o;
    o.reserve(s.size() + 2);
    for (char c : s) {
        if (c == '"' || c == '\\') {
            o += '\\';
            o += c;
        } else if (static_cast<unsigned char>(c) < 0x20) {
            char b[8];
            std::snprintf(b, sizeof b, "\\u%04x", unsigned(static_cast<unsigned char>(c)));
            o += b;
        } else {
            o += c;
        }
    }
    return o;
}

std::string frameLogHeaderJson(const std::string &machineJson, const std::string &sessionJson) {
    std::string j = "{\"format\":\"spark-framelog\",\"format_version\":" + std::to_string(kFrameLogVersion) +
                    ",\"record_size\":" + std::to_string(sizeof(FrameRecord)) + ",\"byte_order\":\"little\",\"fields\":[";
    bool first = true;
    for (const FieldDesc &f : kFields) {
        if (!first) j += ',';
        first = false;
        j += "{\"name\":\"" + std::string(f.name) + "\",\"type\":\"" + f.type + "\",\"offset\":" +
             std::to_string(f.offset) + ",\"unit\":\"" + f.unit + "\",\"when\":\"" + f.when + "\",\"desc\":\"" +
             jsonEscape(f.desc) + "\"}";
    }
    j += "],\"machine\":" + (machineJson.empty() ? std::string("{}") : machineJson);
    j += ",\"session\":" + (sessionJson.empty() ? std::string("{}") : sessionJson) + "}";
    return j;
}

bool TableWriter::open(const std::string &path, const char magic[8], uint32_t version, uint32_t recordSize,
                       const std::string &json, std::string &err) {
    close();
    f_ = std::fopen(path.c_str(), "wb");
    if (!f_) {
        err = "cannot open " + path + " for writing";
        return false;
    }
    recordSize_ = recordSize;
    uint32_t hdr[4] = {version, recordSize, uint32_t(json.size()), uint32_t((24 + json.size() + 7) / 8 * 8)};
    unsigned char pad[8] = {};
    bool ok = std::fwrite(magic, 1, 8, f_) == 8 && std::fwrite(hdr, 4, 4, f_) == 4 &&
              std::fwrite(json.data(), 1, json.size(), f_) == json.size() &&
              std::fwrite(pad, 1, hdr[3] - 24 - json.size(), f_) == hdr[3] - 24 - json.size();
    if (!ok) {
        err = "write failed: " + path;
        close();
        return false;
    }
    buf_.assign(kBufferRows * recordSize, 0);
    used_ = 0;
    rows_ = 0;
    return true;
}

void TableWriter::write(const void *row) {
    if (!f_) return;
    std::memcpy(buf_.data() + used_, row, recordSize_);
    used_ += recordSize_;
    ++rows_;
    if (used_ == buf_.size()) flush();
}

void TableWriter::flush() {
    if (!f_ || !used_) return;
    std::fwrite(buf_.data(), 1, used_, f_);
    std::fflush(f_);
    used_ = 0;
}

void TableWriter::close() {
    if (!f_) return;
    flush();
    std::fclose(f_);
    f_ = nullptr;
}

bool readTable(const std::string &path, TableData &out, std::string &err) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    std::vector<unsigned char> all;
    unsigned char chunk[65536];
    size_t n;
    while ((n = std::fread(chunk, 1, sizeof chunk, f)) > 0) all.insert(all.end(), chunk, chunk + n);
    std::fclose(f);
    if (all.size() < 24) {
        err = "file too short for a header";
        return false;
    }
    uint32_t hdr[4];
    std::memcpy(hdr, all.data() + 8, 16);
    out.magic.assign(reinterpret_cast<const char *>(all.data()), strnlen(reinterpret_cast<const char *>(all.data()), 8));
    out.version = hdr[0];
    out.recordSize = hdr[1];
    if (hdr[1] == 0 || hdr[3] < 24 + hdr[2] || hdr[3] > all.size()) {
        err = "corrupt header";
        return false;
    }
    out.json.assign(reinterpret_cast<const char *>(all.data()) + 24, hdr[2]);
    out.rowCount = (all.size() - hdr[3]) / hdr[1];
    out.rows.assign(all.begin() + hdr[3], all.begin() + hdr[3] + long(out.rowCount * hdr[1]));
    return true;
}

} // namespace spark
