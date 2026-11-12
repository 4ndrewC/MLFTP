// Frame-time models loaded from .spkm files (research/frametime/MODEL_CONTRACT.md).
//
// Families: linear, rls (online recursive least squares), mlp, gbt (quantile
// tree ensembles), gru and tcn (sequence models over the frame history).
//
// Per-frame protocol (the engine and the parity tool both follow it exactly):
//   model.beginFrame(history)      once, before predictions (sequence encoding)
//   model.predict(x, out)          any number of times (one per candidate quality)
//   history.push(h); model.endFrame(x, h, y)   after the frame ran, with the
//                                  features of the quality actually used, the
//                                  history vector and the measured t_total in ms
#pragma once

#include <memory>
#include <span>
#include <string>
#include <vector>

#include "features.hpp"

namespace spark::ft {

inline constexpr uint32_t kSpkmVersion = 2; // v2: feature set spark-ft-v2, OUTS transform mulexp + add column

struct OutputSpec {
    std::string name;      // t_mean_ms | t_pNN_ms | p_miss
    int transform = 0;     // 0 identity, 1 sigmoid, 2 exp, 3 softplus, 4 mulexp
    int baseFeature = -1;  // raw feature: added before the transform, or (mulexp) the multiplier; -1 none
    int addFeature = -1;   // raw feature added after the transform (e.g. a noise/tail term), -1 none
};

class Model {
public:
    virtual ~Model() = default;

    /**
     * Load and validate a .spkm file against the engine's feature list.
     * Returns nullptr and sets `err` on any problem (bad magic/version/CRC,
     * truncation, feature-set or hash mismatch, inconsistent shapes).
     */
    static std::unique_ptr<Model> load(const std::string &path, std::string &err);
    /** Same, from bytes in memory. */
    static std::unique_ptr<Model> loadBytes(const std::vector<unsigned char> &bytes, std::string &err);

    void beginFrame(const HistoryRing &h);
    void predict(const float *x, float *out);
    void endFrame(const float *x, const float *h, float yMs);

    const std::string &family() const { return family_; }
    const std::vector<OutputSpec> &outputs() const { return outputs_; }
    int numOutputs() const { return int(outputs_.size()); }
    int outputIndex(const std::string &name) const;
    int historyLen() const { return historyLen_; }
    double budgetMs() const { return budgetMs_; }
    const std::string &info(const std::string &key) const;
    size_t numParameters() const { return params_; }

protected:
    virtual void encode(const HistoryRing &, const float * /*hMean*/, const float * /*hInvStd*/) {}
    virtual void raw(const float *xn, const float *x, float *out) = 0;
    virtual void observe(const float * /*xn*/, const float * /*x*/, const float * /*hn*/, float /*yMs*/) {}

    friend struct SpkmParser;
    std::string family_;
    std::vector<OutputSpec> outputs_;
    std::vector<std::pair<std::string, std::string>> info_;
    int historyLen_ = 0;
    double budgetMs_ = 1000.0 / 60.0;
    size_t params_ = 0;
    std::vector<float> mean_, invStd_, hMean_, hInvStd_;
    float xn_[kNumFeatures] = {}, hn_[kHistoryDim] = {}, raw_[16] = {};
};

/** CRC-32 (IEEE, as zlib.crc32). */
uint32_t crc32(const unsigned char *data, size_t n, uint32_t crc = 0);

} // namespace spark::ft
