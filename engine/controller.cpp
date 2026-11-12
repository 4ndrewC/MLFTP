#include "controller.hpp"

#include <cstring>

namespace spark::ft {

const char *ctlName(CtlMode m) {
    switch (m) {
    case CtlMode::Fixed: return "fixed";
    case CtlMode::Reactive: return "reactive";
    case CtlMode::Model: return "model";
    case CtlMode::Random: return "random";
    }
    return "?";
}

bool parseCtl(const char *s, CtlMode &m) {
    for (CtlMode c : {CtlMode::Fixed, CtlMode::Reactive, CtlMode::Model, CtlMode::Random})
        if (std::strcmp(s, ctlName(c)) == 0 || (c == CtlMode::Fixed && std::strcmp(s, "none") == 0)) {
            m = c;
            return true;
        }
    return false;
}

void Controller::configure(CtlMode mode, const CtlParams &p) {
    mode_ = mode;
    p_ = p;
    rng_.seed(p.seed * 0x9E3779B97F4A7C15ull + 17);
    upCount_ = missRun_ = 0;
    sinceChange_ = sinceDrop_ = 1 << 20;
    nextRandom_ = 0;
    emaInit_ = false;
    if (mode == CtlMode::Random) { // keep the starting quality for a first random interval
        sinceChange_ = 0;
        nextRandom_ = rng_.range(p_.randMinFrames, p_.randMaxFrames);
    }
}

int Controller::ladderIndex(const Quality &q) {
    int i = QualityLadder::find(q);
    if (i >= 0) return i;
    // Off the ladder: the first level that is no better on any lever.
    for (int l = 0; l < QualityLadder::kCount; ++l) {
        const Quality &c = QualityLadder::kLevels[l];
        if (c.scale >= q.scale && c.particles >= q.particles && c.glow >= q.glow && c.aa >= q.aa) return l;
    }
    return QualityLadder::kCount - 1;
}

CtlDecision Controller::decide(const Quality &cur, double lastMs, double budgetMs, const Predict &predict) {
    ++sinceChange_;
    ++sinceDrop_;
    CtlDecision d;
    switch (mode_) {
    case CtlMode::Fixed: d.q = cur; break;
    case CtlMode::Reactive: d = decideReactive(cur, lastMs, budgetMs); break;
    case CtlMode::Model: d = predict ? decideModel(cur, budgetMs, predict) : CtlDecision{cur, 0, 0, -1}; break;
    case CtlMode::Random: d = decideRandom(cur); break;
    }
    if (!(d.q == cur)) {
        sinceChange_ = 0;
        if (d.action == 1) sinceDrop_ = 0;
    }
    return d;
}

CtlDecision Controller::decideModel(const Quality &cur, double budgetMs, const Predict &predict) {
    CtlDecision d;
    const double target = budgetMs * (1 - p_.margin);
    int c = ladderIndex(cur);
    const Quality &atC = QualityLadder::kLevels[c];
    float vc = predict(atC);
    d.evals = 1;
    if (vc > target) {
        // Over budget: the best (lowest index) cheaper level predicted to fit, else the cheapest.
        int pick = QualityLadder::kCount - 1;
        float pv = vc;
        for (int l = c + 1; l < QualityLadder::kCount; ++l) {
            pv = predict(QualityLadder::kLevels[l]);
            ++d.evals;
            if (pv <= target) {
                pick = l;
                break;
            }
        }
        upCount_ = 0;
        d.q = QualityLadder::kLevels[pick];
        d.value = pv;
        d.action = pick != c || !(atC == cur) ? 1 : 0;
        return d;
    }
    d.q = atC;
    d.value = vc;
    d.action = atC == cur ? 0 : 1;
    if (c > 0 && sinceDrop_ >= p_.upCooldown) {
        float vu = predict(QualityLadder::kLevels[c - 1]);
        ++d.evals;
        upCount_ = vu <= target * p_.upFrac ? upCount_ + 1 : 0;
        if (upCount_ >= p_.upHold) {
            upCount_ = 0;
            d.q = QualityLadder::kLevels[c - 1];
            d.value = vu;
            d.action = 2;
        }
    } else {
        upCount_ = 0;
    }
    return d;
}

CtlDecision Controller::decideReactive(const Quality &cur, double lastMs, double budgetMs) {
    CtlDecision d;
    d.q = cur;
    if (lastMs <= 0) return d; // nothing measured yet
    ema_ = emaInit_ ? (1 - p_.emaAlpha) * ema_ + p_.emaAlpha * lastMs : lastMs;
    emaInit_ = true;
    missRun_ = lastMs > budgetMs ? missRun_ + 1 : 0;
    const double target = budgetMs * (1 - p_.margin);
    int c = ladderIndex(cur);
    int pick = c;
    if (missRun_ >= p_.panicFrames) {
        pick = std::min(QualityLadder::kCount - 1, c + 2);
        missRun_ = 0;
        upCount_ = 0;
    } else if (ema_ > target && sinceChange_ >= p_.downEvery) {
        pick = std::min(QualityLadder::kCount - 1, c + 1);
        upCount_ = 0;
    } else if (ema_ < target * p_.reactUpFrac) {
        if (++upCount_ >= p_.reactUpHold && c > 0) {
            pick = c - 1;
            upCount_ = 0;
        }
    } else {
        upCount_ = 0;
    }
    d.q = QualityLadder::kLevels[pick];
    d.action = pick > c ? 1 : pick < c ? 2 : (d.q == cur ? 0 : 1);
    return d;
}

CtlDecision Controller::decideRandom(const Quality &cur) {
    CtlDecision d;
    d.q = cur;
    if (sinceChange_ < nextRandom_) return d;
    nextRandom_ = rng_.range(p_.randMinFrames, p_.randMaxFrames);
    Quality q;
    if (rng_.chance(float(p_.offLadderProb))) {
        // Each lever: full quality 40% of the time (where the budget pressure is), else any level.
        auto lever = [&](int levels) { return uint8_t(rng_.chance(0.4f) ? 0 : rng_.range(0, levels - 1)); };
        q.scale = lever(Quality::kScaleLevels);
        q.particles = lever(Quality::kParticleLevels);
        q.glow = lever(Quality::kGlowLevels);
        q.aa = uint8_t(rng_.chance(0.5f) ? 0 : 1);
    } else {
        q = QualityLadder::kLevels[rng_.range(0, QualityLadder::kCount - 1)];
    }
    if (q == cur) return d;
    int a = ladderIndex(cur), b = ladderIndex(q);
    d.q = q;
    d.action = b >= a ? 1 : 2;
    return d;
}

} // namespace spark::ft
