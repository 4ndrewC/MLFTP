// spkm_eval: run .spkm models through the engine's predictor code.
//
//   spkm_eval check MODEL.spkm              load + validate; prints OK/REJECT (exit 0/1)
//   spkm_eval run MODEL.spkm CASES OUT      parity: replay a frame sequence through the per-frame protocol
//   spkm_eval bench MODEL.spkm [ITERS]      inference cost per call (random inputs)
//
// CASES: u32 n, u32 F, u32 D, then n x (f32 x[F], f32 h[D], f32 y)
// OUT:   n x f32 outputs[n_out] (predictions made before each frame's end_frame)
// Built by `make parity` / `make bench-predictor` (see the Makefile).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "predictor.hpp"

using namespace spark::ft;
using Clock = std::chrono::steady_clock;

namespace {

int check(const char *path) {
    std::string err;
    auto m = Model::load(path, err);
    if (!m) {
        std::printf("REJECT %s\n", err.c_str());
        return 1;
    }
    std::printf("OK %s params=%zu outputs=%d history_len=%d\n", m->family().c_str(), m->numParameters(),
                m->numOutputs(), m->historyLen());
    return 0;
}

int run(const char *path, const char *cases, const char *outPath) {
    std::string err;
    auto m = Model::load(path, err);
    if (!m) {
        std::fprintf(stderr, "REJECT %s\n", err.c_str());
        return 1;
    }
    FILE *f = std::fopen(cases, "rb");
    if (!f) return 1;
    uint32_t hdr[3];
    if (std::fread(hdr, 4, 3, f) != 3 || hdr[1] != uint32_t(kNumFeatures) || hdr[2] != uint32_t(kHistoryDim)) {
        std::fprintf(stderr, "bad cases header\n");
        return 1;
    }
    std::vector<float> row(size_t(kNumFeatures + kHistoryDim + 1)), out(size_t(m->numOutputs()));
    std::vector<float> all;
    HistoryRing ring;
    for (uint32_t i = 0; i < hdr[0]; ++i) {
        if (std::fread(row.data(), 4, row.size(), f) != row.size()) return 1;
        const float *x = row.data(), *h = x + kNumFeatures, y = row.back();
        m->beginFrame(ring);
        m->predict(x, out.data());
        all.insert(all.end(), out.begin(), out.end());
        ring.push(h);
        m->endFrame(x, h, y);
    }
    std::fclose(f);
    FILE *o = std::fopen(outPath, "wb");
    if (!o) return 1;
    std::fwrite(all.data(), 4, all.size(), o);
    std::fclose(o);
    return 0;
}

double pct(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, size_t(p * double(v.size())))];
}

int bench(const char *path, int iters) {
    std::string err;
    auto m = Model::load(path, err);
    if (!m) {
        std::printf("REJECT %s\n", err.c_str());
        return 1;
    }
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0, 1);
    HistoryRing ring;
    float h[kHistoryDim], x[kNumFeatures], out[16];
    for (int i = 0; i < kMaxHistory; ++i) {
        for (float &v : h) v = nd(rng);
        ring.push(h);
    }
    std::vector<double> tBegin, tPredict, tEnd;
    double sink = 0;
    for (int it = 0; it < iters; ++it) {
        for (float &v : x) v = nd(rng);
        for (float &v : h) v = nd(rng);
        auto a = Clock::now();
        m->beginFrame(ring);
        auto b = Clock::now();
        m->predict(x, out);
        auto c = Clock::now();
        ring.push(h);
        m->endFrame(x, h, 3.0f + nd(rng));
        auto d = Clock::now();
        sink += out[0];
        tBegin.push_back(std::chrono::duration<double, std::micro>(b - a).count());
        tPredict.push_back(std::chrono::duration<double, std::micro>(c - b).count());
        tEnd.push_back(std::chrono::duration<double, std::micro>(d - c).count());
    }
    std::printf("%-8s params=%-7zu predict p50 %7.2f p99 %7.2f us | begin_frame p50 %7.2f p99 %7.2f | end_frame "
                "p50 %6.2f p99 %6.2f  (sink %.1g)\n",
                m->family().c_str(), m->numParameters(), pct(tPredict, 0.5), pct(tPredict, 0.99), pct(tBegin, 0.5),
                pct(tBegin, 0.99), pct(tEnd, 0.5), pct(tEnd, 0.99), sink * 0);
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    if (argc >= 3 && !std::strcmp(argv[1], "check")) return check(argv[2]);
    if (argc >= 5 && !std::strcmp(argv[1], "run")) return run(argv[2], argv[3], argv[4]);
    if (argc >= 3 && !std::strcmp(argv[1], "bench")) return bench(argv[2], argc >= 4 ? std::atoi(argv[3]) : 20000);
    std::fprintf(stderr, "usage: spkm_eval check MODEL | run MODEL CASES OUT | bench MODEL [ITERS]\n");
    return 2;
}
