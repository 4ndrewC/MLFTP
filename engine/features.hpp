// Feature vector for learned frame-time prediction ("spark-ft-v2").
//
// research/frametime/MODEL_CONTRACT.md is the specification; this file and
// research/frametime/train/features.py implement it (checked against each
// other by research/frametime/tests/test_features_parity.py).
//
// x(k+1) is computed at the start of frame k+1, before its sim steps run, from
//   - the frame's planned-work record (FrameRecord fields marked "before"/"system"),
//   - a CANDIDATE quality setting (so the controller can ask "what if"),
//   - FeatureState: summaries of frames <= k (updated after each frame).
#pragma once

#include <cstdint>

#include "fingerprint.hpp"
#include "framelog.hpp"
#include "profile.hpp"

namespace spark::ft {

inline constexpr const char *kFeatureSet = "spark-ft-v2";
inline constexpr int kNumFeatures = 64;
extern const char *const kFeatureNames[kNumFeatures];

/** History vectors (sequence models): x(j) followed by these measured times of frame j. */
inline constexpr int kHistoryExtra = 3;
inline constexpr int kHistoryDim = kNumFeatures + kHistoryExtra;
extern const char *const kHistoryExtraNames[kHistoryExtra];
inline constexpr int kMaxHistory = 64;

/** FNV-1a 64 over the feature-set id, feature names and history names ('\n'-joined). */
uint64_t featureHash();

/**
 * Fingerprint-derived constants (LIT_REVIEW_HW §3.3 A/B), computed once per session
 * from the spark-hw-v1 JSON. All zero when no fingerprint is loaded.
 */
struct FpCoef {
    bool valid = false;
    double stepBase = 0, stepSlope = 0, stepNormal = 0;      // ms/step = base + slope * entities
    double nsClear = 0, nsRect = 0, nsPoly = 0, nsGlow = 0;  // ns per counted pixel
    double nsCircle[2] = {}, nsRing[2] = {}, nsLine[2] = {}; // [0] AA on, [1] aliased
    double calib = 1, cliff = 1, px1 = 1, px32 = 1;           // canned-frame calibration, cache cliff
    double noiseP99Ms = 0, noiseP999Ms = 0, sleepP99Ms = 0;   // tail terms
    double logKStep = 0, logKRpx = 0, logCliff = 0, logKGlow = 0, logKPart = 0, logGhz = 0, logLatL3 = 0,
           logLatDram = 0, logBwDram = 0;                    // reduced fingerprint (log10)
    static FpCoef from(const fp::Fingerprint &f);
};

/** Session constants the features need. */
struct FrameContext {
    int logicalW = 960, logicalH = 540;
    float outScale = 1; // stress: base pixels per logical unit before the quality render scale
    FpCoef fp;
};

/** Effective pixels per logical unit for a quality level. */
inline float effectiveScale(const FrameContext &c, const Quality &q) { return c.outScale * q.renderScale(); }

class FeatureState {
public:
    void reset() { *this = FeatureState(); }
    /** Fill x[kNumFeatures] for the coming frame `planned` at candidate quality `q`. */
    void compute(const FrameRecord &planned, const Quality &q, const FrameContext &ctx, float *x) const;
    /** Fold in a finished frame (its record now has timings and render counts). */
    void update(const FrameRecord &done);

    // Summaries of finished frames (double so Python can reproduce them exactly).
    double tPrev = 0, tPrev2 = 0, tEma = 0, tStepsPrev = 0, tRenderPrev = 0;
    double last8[8] = {};
    int n8 = 0, i8 = 0, stepsPrev = 0;
    double emaStep = 0, emaRpw = 0;
    bool tEmaInit = false, emaStepInit = false, emaRpwInit = false;
    double rasterLpx = 0, glowReqLpx = 0; // previous frame's pixel work in logical units
    double lpxRect = 0, lpxCircle = 0, lpxRing = 0, lpxLine = 0, lpxPoly = 0; // per class, logical units
};

/** Fixed-capacity ring of history vectors (oldest first when read). */
class HistoryRing {
public:
    void clear() { n_ = head_ = 0; }
    void push(const float *v);
    int size() const { return n_; }
    /** i = 0 is the oldest stored vector, size()-1 the newest. */
    const float *at(int i) const { return buf_[(head_ + kMaxHistory - n_ + i) % kMaxHistory]; }

private:
    float buf_[kMaxHistory][kHistoryDim] = {};
    int n_ = 0, head_ = 0;
};

} // namespace spark::ft
