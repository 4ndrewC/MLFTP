// Quality controllers for frame-time research (MODEL_CONTRACT.md, "Controller").
//
//   Fixed     never changes quality (default)
//   Reactive  non-ML baseline in the style of dynamic resolution: EMA of past
//             frame times vs the budget, rate-limited steps, panic drop
//   Model     predict-then-optimise: asks the model for the predicted frame
//             time of candidate ladder levels and picks the best level
//             predicted to fit, with hysteresis on the way back up
//   Random    exploration policy for data collection: random levels/levers
//             at random intervals (gives the data quality transitions)
#pragma once

#include <cstdint>
#include <functional>

#include "math.hpp"
#include "profile.hpp"

namespace spark::ft {

enum class CtlMode : uint8_t { Fixed, Reactive, Model, Random };
const char *ctlName(CtlMode m);
bool parseCtl(const char *s, CtlMode &m);

struct CtlParams {
    double budgetMs = 1000.0 / 60.0;
    double margin = 0.10;   // aim for budget * (1 - margin)
    // Model controller
    double upFrac = 0.80;   // raise quality only if the better level is predicted <= target * upFrac ...
    int upHold = 30;        // ... for this many consecutive frames
    int upCooldown = 60;    // and not within this many frames of a drop
    // Reactive controller
    double emaAlpha = 0.25;
    int downEvery = 6;      // at most one step down per this many frames
    int panicFrames = 3;    // this many consecutive misses: drop two levels at once
    double reactUpFrac = 0.65;
    int reactUpHold = 60;
    // Random controller
    int randMinFrames = 90, randMaxFrames = 900;
    double offLadderProb = 0.5;
    uint64_t seed = 1;
};

struct CtlDecision {
    Quality q;
    int action = 0;  // 0 keep, 1 lowered, 2 raised
    int evals = 0;   // model evaluations used
    float value = -1; // predicted decision value of the chosen level (model mode)
};

class Controller {
public:
    /** Predicted decision value (ms) for a candidate quality. */
    using Predict = std::function<float(const Quality &)>;

    void configure(CtlMode mode, const CtlParams &p);
    CtlMode mode() const { return mode_; }
    const CtlParams &params() const { return p_; }

    /**
     * Decide the quality for the coming frame. `lastMs` is the previous frame's
     * measured time (reactive), `predict` may be empty unless mode == Model.
     * `budgetMs` is the effective budget for this frame (budget minus present).
     */
    CtlDecision decide(const Quality &current, double lastMs, double budgetMs, const Predict &predict);

private:
    CtlDecision decideModel(const Quality &cur, double budgetMs, const Predict &predict);
    CtlDecision decideReactive(const Quality &cur, double lastMs, double budgetMs);
    CtlDecision decideRandom(const Quality &cur);
    static int ladderIndex(const Quality &q); // nearest ladder level

    CtlMode mode_ = CtlMode::Fixed;
    CtlParams p_;
    Rng rng_{1};
    int upCount_ = 0, sinceChange_ = 1 << 20, sinceDrop_ = 1 << 20, missRun_ = 0, nextRandom_ = 0;
    double ema_ = 0;
    bool emaInit_ = false;
};

} // namespace spark::ft
