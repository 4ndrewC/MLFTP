#include "fingerprint.hpp"

#include <sched.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>

#include "canvas.hpp"
#include "engine.hpp"
#include "framelog.hpp"
#include "particles.hpp"
#include "profile.hpp"

namespace spark::fp {

namespace {

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

struct Stat {
    double med = 0, riqr = 0;
};
Stat stat(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    auto q = [&](double p) {
        double i = p * double(v.size() - 1);
        size_t lo = size_t(i);
        size_t hi = std::min(v.size() - 1, lo + 1);
        return v[lo] + (v[hi] - v[lo]) * (i - double(lo));
    };
    Stat s;
    s.med = q(0.5);
    s.riqr = s.med != 0 ? (q(0.75) - q(0.25)) / std::fabs(s.med) : 0;
    return s;
}
double quantile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, size_t(p * double(v.size() - 1) + 0.5))];
}

/** Record of fp values (insertion-ordered for readable JSON). */
struct Out {
    std::vector<std::pair<std::string, double>> vals, iqr;
    void put(const std::string &k, double v) { vals.emplace_back(k, v); }
    void put(const std::string &k, const Stat &s) {
        vals.emplace_back(k, s.med);
        iqr.emplace_back(k, s.riqr);
    }
};

template <class F> Stat repeat(int k, F &&f) {
    std::vector<double> v;
    for (int i = 0; i < k; ++i) v.push_back(f());
    return stat(v);
}

volatile uint64_t gSink = 0;

// F1: dependent chain of single-cycle integer adds -> adds per second ~ core clock.
double ghzOnce() {
    const long n = 4'000'000;
    uint64_t x = gSink, y = gSink | 1;
    auto t0 = Clock::now();
    for (long i = 0; i < n; ++i) {
#if defined(__x86_64__)
        // Register-register adds: Golden Cove eliminates dependent add-immediate chains at rename.
        asm volatile("add %1, %0\n\tadd %1, %0\n\tadd %1, %0\n\tadd %1, %0\n\t"
                     "add %1, %0\n\tadd %1, %0\n\tadd %1, %0\n\tadd %1, %0"
                     : "+r"(x)
                     : "r"(y));
#else
        for (int j = 0; j < 8; ++j) asm volatile("" : "+r"(x)), x += y;
#endif
    }
    double dt = secs(t0, Clock::now());
    gSink = x;
    return double(n) * 8 / dt / 1e9;
}

// F2: pointer chase over a random cyclic permutation of 64-byte nodes (lmbench lat_mem_rd style).
double chaseNs(std::vector<uint64_t> &buf, size_t bytes, long loads, uint64_t seed) {
    size_t nodes = std::max<size_t>(2, bytes / 64);
    std::vector<uint32_t> perm(nodes);
    for (size_t i = 0; i < nodes; ++i) perm[i] = uint32_t(i);
    Rng rng(seed);
    for (size_t i = nodes - 1; i > 0; --i) std::swap(perm[i], perm[size_t(rng.next() % i)]); // Sattolo: one cycle
    for (size_t i = 0; i < nodes; ++i) buf[size_t(perm[i]) * 8] = uint64_t(perm[(i + 1) % nodes]) * 8;
    uint64_t p = 0;
    for (long i = 0; i < std::min<long>(loads, long(nodes)); ++i) p = buf[p]; // warm
    auto t0 = Clock::now();
    for (long i = 0; i < loads; ++i) p = buf[p];
    double dt = secs(t0, Clock::now());
    gSink = gSink + p;
    return dt / double(loads) * 1e9;
}

// F3: STREAM-like triad a = b + s*c, GB/s (24 bytes per element).
double triadGBs(std::vector<double> &a, std::vector<double> &b, std::vector<double> &c, size_t n, int passes) {
    auto t0 = Clock::now();
    for (int p = 0; p < passes; ++p) {
        for (size_t i = 0; i < n; ++i) a[i] = b[i] + 1.5 * c[i];
        std::swap(a, b);
    }
    double dt = secs(t0, Clock::now());
    gSink = gSink + uint64_t(a[n / 2]);
    return 24.0 * double(n) * passes / dt / 1e9;
}

// F4: 8 independent multiply-add chains (float and int32).
double flopsG() {
    float x[8] = {1, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f};
    const float m = 0.999999f, c = 0.000001f;
    const long n = 2'000'000;
    auto t0 = Clock::now();
    for (long i = 0; i < n; ++i)
        for (float &v : x) v = v * m + c;
    double dt = secs(t0, Clock::now());
    float s = 0;
    for (float v : x) s += v;
    gSink = gSink + uint64_t(s);
    return 16.0 * double(n) / dt / 1e9;
}
double intopsG() {
    uint32_t x[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const long n = 2'000'000;
    auto t0 = Clock::now();
    for (long i = 0; i < n; ++i)
        for (uint32_t &v : x) v = v * 5u + 3u;
    double dt = secs(t0, Clock::now());
    uint32_t s = 0;
    for (uint32_t v : x) s ^= v;
    gSink = gSink + s;
    return 16.0 * double(n) / dt / 1e9;
}

// F5: data-dependent branch over random vs sorted bytes; cost per mispredict.
double branchPass(const std::vector<uint8_t> &v) {
    uint64_t a = 0, b = 0;
    auto t0 = Clock::now();
    for (uint8_t x : v) {
        if (x >= 128) {
            asm volatile("" ::: "memory"); // keep it a real branch (no cmov)
            a += x;
        } else {
            b ^= x;
        }
    }
    double dt = secs(t0, Clock::now());
    gSink = gSink + a + b;
    return dt;
}

// F6: malloc + first touch of 64 MB in 4 KB pages, us per MB.
double firstTouchUsPerMB() {
    const size_t bytes = size_t(64) << 20;
    auto t0 = Clock::now();
    char *p = static_cast<char *>(std::malloc(bytes));
    if (!p) return 0;
    for (size_t i = 0; i < bytes; i += 4096) p[i] = char(i);
    double dt = secs(t0, Clock::now());
    gSink = gSink + uint64_t(p[bytes / 2]);
    std::free(p);
    return dt * 1e6 / 64.0;
}

// F8 micro scenes: draw a fixed, seeded set of one primitive class; return ns per counted pixel.
double rasterClassNsPerPx(Canvas &cv, Prim cls, bool aa, int reps, Stat *out) {
    cv.setAntialias(aa);
    std::vector<double> v;
    for (int r = 0; r < reps; ++r) {
        Rng rng(1234);
        cv.resetStats();
        auto pos = [&] { return Vec2{rng.range(0.0f, 960.0f), rng.range(0.0f, 540.0f)}; };
        auto t0 = Clock::now();
        switch (cls) {
        case Prim::Clear:
            for (int i = 0; i < 10; ++i) cv.clear(Color(10, 12, 20));
            break;
        case Prim::Rect:
            for (int i = 0; i < 3000; ++i) {
                Vec2 p = pos();
                cv.rect(p.x, p.y, rng.range(1.0f, 30.0f), rng.range(1.0f, 30.0f), Color(200, 120, 60, i % 2 ? 255 : 160));
            }
            break;
        case Prim::Circle:
            for (int i = 0; i < 1500; ++i) cv.circle(pos(), rng.range(2.0f, 22.0f), Color(90, 200, 255, 220));
            break;
        case Prim::Ring:
            for (int i = 0; i < 800; ++i) cv.ring(pos(), rng.range(4.0f, 50.0f), rng.range(1.5f, 4.0f), Color(255, 90, 200, 200));
            break;
        case Prim::Line:
            for (int i = 0; i < 3000; ++i) {
                Vec2 a = pos();
                cv.line(a, a + Vec2::fromAngle(rng.angle(), rng.range(4.0f, 60.0f)), rng.range(1.0f, 5.0f),
                        Color(255, 255, 200, 200));
            }
            break;
        case Prim::Poly: {
            const Vec2 d[] = {{1.2f, 0}, {0, 1.2f}, {-1.2f, 0}, {0, -1.2f}};
            for (int i = 0; i < 1500; ++i)
                cv.shape(d, pos(), rng.angle(), rng.range(4.0f, 24.0f), Color(120, 255, 120, 64), Color(0u), 0);
            break;
        }
        case Prim::Glow:
            for (int i = 0; i < 300; ++i) cv.glow(pos(), rng.range(10.0f, 60.0f), Color(255, 220, 80, 90));
            break;
        default: break;
        }
        double dt = secs(t0, Clock::now());
        double px = double(cv.stats().px[size_t(cls)]);
        v.push_back(px > 0 ? dt * 1e9 / px : 0);
    }
    cv.setAntialias(true);
    *out = stat(v);
    return out->med;
}

std::string fmtNum(double v) {
    char b[40];
    std::snprintf(b, sizeof b, "%.9g", std::isfinite(v) ? v : 0.0);
    return b;
}

uint64_t fnv(const std::string &s, uint64_t h = 1469598103934665603ull) {
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h;
}

std::string binaryHash() {
    std::ifstream f("/proc/self/exe", std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    char b[20];
    std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(fnv(ss.str())));
    return b;
}

std::string cpuModel() {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("model name", 0) == 0) return line.substr(line.find(':') + 2);
    return "unknown";
}

struct State {
    std::string host, model, core, power, profile, epp, maxf, gov, corun, bin;
    int cpu = 0;
};
State readState(const std::string &corunner) {
    State s;
    char host[256] = {};
    gethostname(host, sizeof host - 1);
    s.host = host;
    s.model = cpuModel();
    s.cpu = std::max(0, sched_getcpu());
    s.core = coreType(s.cpu);
    s.power = powerSource();
    s.profile = platformProfile();
    char p[128];
    std::snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/cpufreq/energy_performance_preference", s.cpu);
    s.epp = readFirstLine(p);
    std::snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_max_freq", s.cpu);
    s.maxf = readFirstLine(p);
    std::snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor", s.cpu);
    s.gov = readFirstLine(p);
    s.corun = corunner.empty() ? "none" : corunner;
    static const std::string bin = binaryHash();
    s.bin = bin;
    return s;
}
std::string keyOf(const State &s) {
    return "host=" + s.host + "|cpu=" + s.model + "|core=" + s.core + "|power=" + s.power + "|profile=" + s.profile +
           "|epp=" + s.epp + "|maxf=" + s.maxf + "|gov=" + s.gov + "|corunner=" + s.corun + "|bin=" + s.bin;
}

} // namespace

std::string stateKey(const std::string &corunner) { return keyOf(readState(corunner)); }

std::string cachePath(const std::string &dir, const std::string &key) {
    char b[40];
    std::snprintf(b, sizeof b, "/fp-%016llx.json", static_cast<unsigned long long>(fnv(key)));
    return dir + b;
}

std::string defaultCacheDir() {
    if (const char *d = std::getenv("SPARK_FP_CACHE")) return d;
    if (const char *x = std::getenv("XDG_CACHE_HOME")) return std::string(x) + "/spark_warden";
    const char *h = std::getenv("HOME");
    return std::string(h ? h : ".") + "/.cache/spark_warden";
}

std::string run(const Config &base, const Hooks &hooks, const std::string &corunner) {
    auto start = Clock::now();
    State st = readState(corunner);
    Out o;
    // Warm-up: let turbo / the governor ramp up before measuring.
    for (auto t = Clock::now(); secs(t, Clock::now()) < 0.3;) ghzOnce();

    o.put("ghz_start", repeat(5, ghzOnce));

    { // F2 latency
        std::vector<uint64_t> buf((size_t(128) << 20) / 8);
        const size_t sizes[] = {16u << 10, 256u << 10, 1536u << 10, 12u << 20, 128u << 20};
        const char *names[] = {"lat_l1_ns", "lat_l2_ns", "lat_l3_ns", "lat_llc12_ns", "lat_dram_ns"};
        for (int i = 0; i < 5; ++i) {
            long loads = i == 4 ? 200000 : 1000000;
            o.put(names[i], repeat(5, [&] { return chaseNs(buf, sizes[i], loads, 77 + uint64_t(i)); }));
        }
    }
    { // F3 bandwidth (triad), per-array sizes: L2-, L3-, DRAM-resident
        const size_t big = (size_t(32) << 20) / 8;
        std::vector<double> a(big, 1.0), b(big, 2.0), c(big, 3.0);
        o.put("bw_l2_gbs", repeat(5, [&] { return triadGBs(a, b, c, (size_t(256) << 10) / 24, 200); }));
        o.put("bw_l3_gbs", repeat(5, [&] { return triadGBs(a, b, c, (size_t(8) << 20) / 24, 8); }));
        o.put("bw_dram_gbs", repeat(5, [&] { return triadGBs(a, b, c, big, 1); }));
    }
    o.put("flops_g", repeat(5, flopsG)); // F4
    o.put("intops_g", repeat(5, intopsG));
    { // F5
        std::vector<uint8_t> v(1 << 20);
        Rng rng(5);
        for (auto &x : v) x = uint8_t(rng.next());
        std::vector<uint8_t> sorted = v;
        std::sort(sorted.begin(), sorted.end());
        o.put("branch_miss_ns", repeat(5, [&] {
                  return (branchPass(v) - branchPass(sorted)) * 1e9 / (double(v.size()) * 0.5);
              }));
    }
    o.put("first_touch_us_per_mb", repeat(5, firstTouchUsPerMB)); // F6
    { // F7: fixed work quanta (~50 us) and 1 ms sleeps
        uint64_t x = 1;
        long iters = 20000;
        auto t0 = Clock::now();
        for (long i = 0; i < iters; ++i) x = x * 6364136223846793005ull + 1442695040888963407ull;
        double per = secs(t0, Clock::now()) / double(iters);
        long q = std::max(1000L, long(50e-6 / std::max(per, 1e-10)));
        std::vector<double> d;
        for (int k = 0; k < 4000; ++k) {
            auto a = Clock::now();
            for (long i = 0; i < q; ++i) x = x * 6364136223846793005ull + 1442695040888963407ull;
            d.push_back(secs(a, Clock::now()) * 1e6);
        }
        gSink = gSink + x;
        double p50 = quantile(d, 0.5);
        o.put("noise_quantum_p50_us", p50);
        o.put("noise_excess_p99_us", quantile(d, 0.99) - p50);
        o.put("noise_excess_p999_us", quantile(d, 0.999) - p50);
        o.put("noise_excess_max_us", *std::max_element(d.begin(), d.end()) - p50);
        std::vector<double> sl;
        for (int k = 0; k < 200; ++k) {
            auto a = Clock::now();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            sl.push_back(secs(a, Clock::now()) * 1e6 - 1000.0);
        }
        o.put("sleep_over_p50_us", quantile(sl, 0.5));
        o.put("sleep_over_p99_us", quantile(sl, 0.99));
    }
    { // F8a raster micro kernels (960x540, pixel scale 1): ns per counted pixel
        Canvas cv(960, 540);
        Stat s;
        auto cls = [&](const char *name, Prim p, bool aa) {
            rasterClassNsPerPx(cv, p, aa, 5, &s);
            o.put(name, s);
        };
        cls("ns_px_clear", Prim::Clear, true);
        cls("ns_px_rect", Prim::Rect, true);
        cls("ns_px_poly", Prim::Poly, true);
        cls("ns_px_glow", Prim::Glow, true);
        cls("ns_px_circle", Prim::Circle, true);
        cls("ns_px_circle_aliased", Prim::Circle, false);
        cls("ns_px_ring", Prim::Ring, true);
        cls("ns_px_ring_aliased", Prim::Ring, false);
        cls("ns_px_line", Prim::Line, true);
        cls("ns_px_line_aliased", Prim::Line, false);
    }
    // F8b/F9: canned game state (fixed seed, bot), render at out-scale 1.0 and 3.2; sim ms/step.
    auto canned = [&](float spawn, float burst, int advanceSteps, bool raster) {
        Config c;
        c.headless = true;
        c.width = base.width;
        c.height = base.height;
        c.step = base.step;
        c.burstMult = burst;
        Engine *ep = nullptr;
        c.onStep = [&](long f) {
            if (hooks.botStep) hooks.botStep(*ep, f);
        };
        if (hooks.setSpawnMult) hooks.setSpawnMult(spawn);
        Engine e(c);
        ep = &e;
        e.startScene(hooks.makeScene(e));
        e.advance(advanceSteps);
        if (raster) {
            for (float scale : {1.0f, 3.2f}) {
                e.canvas().setPixelScale(scale);
                e.render(); // warm
                std::vector<double> ms;
                CanvasStats cs;
                for (int r = 0; r < (scale == 1.0f ? 9 : 5); ++r) {
                    e.canvas().resetStats();
                    auto t0 = Clock::now();
                    e.render();
                    ms.push_back(secs(t0, Clock::now()) * 1e3);
                    cs = e.canvas().stats();
                }
                Stat s = stat(ms);
                const char *tag = scale == 1.0f ? "s1" : "s32";
                o.put(std::string("frame_ms_") + tag, s);
                auto px = [&](Prim p) { return double(cs.px[size_t(p)]); };
                o.put(std::string("frame_px_") + tag, px(Prim::Clear));
                o.put(std::string("frame_pxw_clear_") + tag, px(Prim::Clear));
                o.put(std::string("frame_pxw_rect_") + tag, px(Prim::Rect));
                o.put(std::string("frame_pxw_poly_") + tag, px(Prim::Poly));
                o.put(std::string("frame_pxw_circle_") + tag, px(Prim::Circle));
                o.put(std::string("frame_pxw_ring_") + tag, px(Prim::Ring));
                o.put(std::string("frame_pxw_line_") + tag, px(Prim::Line));
                o.put(std::string("frame_pxw_glow_") + tag, double(cs.glowReqPx));
            }
            e.canvas().setPixelScale(1.0f);
        }
        FrameRecord r{};
        if (e.top()) e.top()->planWork(r);
        std::vector<double> ms;
        for (int rep = 0; rep < 5; ++rep) {
            auto t0 = Clock::now();
            e.advance(40);
            ms.push_back(secs(t0, Clock::now()) * 1e3 / 40.0);
        }
        if (hooks.setSpawnMult) hooks.setSpawnMult(-1.0f); // restore the caller's setting
        return std::make_pair(stat(ms), double(r.entities));
    };
    if (hooks.makeScene) {
        // 300 steps in (wave 2); the autopilot survives past the 500 steps used here on this seed.
        auto [norm, eN] = canned(1.0f, 1.0f, 300, true);
        o.put("step_ms_normal", norm);
        o.put("step_entities_normal", eN);
        auto [heavy, eH] = canned(3.0f, 4.0f, 300, false);
        o.put("step_ms_heavy", heavy);
        o.put("step_entities_heavy", eH);
    }
    { // F10 particles: update + draw cost per particle
        Particles ps(100000, 3);
        Burst b;
        b.count = 4000;
        b.life = 100;
        ps.burst({480, 270}, b);
        o.put("part_update_ns", repeat(5, [&] {
                  auto t0 = Clock::now();
                  for (int i = 0; i < 50; ++i) ps.update(0.0001f);
                  return secs(t0, Clock::now()) * 1e9 / (50.0 * double(ps.size()));
              }));
        Canvas cv(960, 540);
        o.put("part_draw_ns", repeat(5, [&] {
                  auto t0 = Clock::now();
                  ps.draw(cv);
                  return secs(t0, Clock::now()) * 1e9 / double(ps.size());
              }));
    }
    o.put("ghz_end", repeat(5, ghzOnce));
    double dur = secs(start, Clock::now());

    // Calibration of the micro-kernel cost model on the canned frame (LIT_REVIEW_HW §3.3 A).
    auto val = [&](const std::string &k) {
        for (auto &[n, v] : o.vals)
            if (n == k) return v;
        return 0.0;
    };
    for (const char *tag : {"s1", "s32"}) {
        std::string t = tag;
        double pred = 1e-6 * (val("ns_px_clear") * val("frame_pxw_clear_" + t) + val("ns_px_rect") * val("frame_pxw_rect_" + t) +
                              val("ns_px_poly") * val("frame_pxw_poly_" + t) + val("ns_px_circle") * val("frame_pxw_circle_" + t) +
                              val("ns_px_ring") * val("frame_pxw_ring_" + t) + val("ns_px_line") * val("frame_pxw_line_" + t) +
                              val("ns_px_glow") * val("frame_pxw_glow_" + t));
        o.put("frame_pred_ms_" + t, pred);
    }
    double calib1 = val("frame_pred_ms_s1") > 0 ? val("frame_ms_s1") / val("frame_pred_ms_s1") : 1.0;
    double calib32 = val("frame_pred_ms_s32") > 0 ? val("frame_ms_s32") / val("frame_pred_ms_s32") : calib1;
    o.put("raster_calib", calib1);
    o.put("cliff_ratio", calib1 > 0 ? calib32 / calib1 : 1.0);
    double eN = val("step_entities_normal"), eH = val("step_entities_heavy");
    double slope = eH > eN ? (val("step_ms_heavy") - val("step_ms_normal")) / (eH - eN) : 0.0;
    o.put("step_ms_per_entity", std::max(0.0, slope));
    o.put("step_ms_base", std::max(0.0, val("step_ms_normal") - std::max(0.0, slope) * eN));
    o.put("duration_s", dur);

    std::string j = "{\"format\":\"" + std::string(kFormat) + "\",\"key\":\"" + jsonEscape(keyOf(st)) + "\",\"state\":{" +
                    "\"host\":\"" + jsonEscape(st.host) + "\",\"cpu_model\":\"" + jsonEscape(st.model) +
                    "\",\"cpu\":" + std::to_string(st.cpu) + ",\"core_type\":\"" + st.core + "\",\"power\":\"" + st.power +
                    "\",\"platform_profile\":\"" + jsonEscape(st.profile) + "\",\"epp\":\"" + jsonEscape(st.epp) +
                    "\",\"scaling_max_freq\":\"" + st.maxf + "\",\"governor\":\"" + jsonEscape(st.gov) +
                    "\",\"corunner\":\"" + jsonEscape(st.corun) + "\",\"binary\":\"" + st.bin + "\",\"compiler\":\"" +
                    jsonEscape(__VERSION__) + "\"},\"fp\":{";
    for (size_t i = 0; i < o.vals.size(); ++i)
        j += (i ? "," : "") + std::string("\"") + o.vals[i].first + "\":" + fmtNum(o.vals[i].second);
    j += "},\"rel_iqr\":{";
    for (size_t i = 0; i < o.iqr.size(); ++i)
        j += (i ? "," : "") + std::string("\"") + o.iqr[i].first + "\":" + fmtNum(o.iqr[i].second);
    j += "}}";
    return j;
}

bool parse(const std::string &json, Fingerprint &out, std::string &err) {
    out = Fingerprint{};
    if (json.find(std::string("\"format\":\"") + kFormat + "\"") == std::string::npos) {
        err = std::string("not a ") + kFormat + " fingerprint";
        return false;
    }
    auto k = json.find("\"key\":\"");
    if (k != std::string::npos) {
        auto e = json.find('"', k + 7);
        out.key = json.substr(k + 7, e - k - 7);
    }
    auto f = json.find("\"fp\":{");
    if (f == std::string::npos) {
        err = "fingerprint has no fp section";
        return false;
    }
    size_t p = f + 6, end = json.find('}', p);
    while (p < end) {
        auto q1 = json.find('"', p);
        if (q1 == std::string::npos || q1 > end) break;
        auto q2 = json.find('"', q1 + 1);
        std::string name = json.substr(q1 + 1, q2 - q1 - 1);
        char *e = nullptr;
        double v = std::strtod(json.c_str() + q2 + 2, &e);
        out.v[name] = v;
        p = size_t(e - json.c_str()) + 1;
    }
    for (const char *req : {"step_ms_base", "raster_calib", "ns_px_clear", "ns_px_glow", "lat_dram_ns"})
        if (!out.v.count(req)) {
            err = std::string("fingerprint is missing ") + req;
            return false;
        }
    out.json = json;
    out.valid = true;
    return true;
}

Fingerprint resolve(const std::string &file, const std::string &cacheDir, const Config &base, const Hooks &hooks,
                    const std::string &corunner, bool allowRun) {
    Fingerprint fp;
    std::string err;
    auto readFile = [](const std::string &path, std::string &txt) {
        std::ifstream f(path);
        if (!f) return false;
        std::stringstream ss;
        ss << f.rdbuf();
        txt = ss.str();
        return true;
    };
    if (file == "none") return fp;
    std::string txt;
    if (!file.empty()) {
        if (!readFile(file, txt) || !parse(txt, fp, err))
            std::fprintf(stderr, "spark: fingerprint %s unusable: %s\n", file.c_str(), err.empty() ? "unreadable" : err.c_str());
        return fp;
    }
    std::string key = stateKey(corunner), dir = cacheDir.empty() ? defaultCacheDir() : cacheDir;
    std::string path = cachePath(dir, key);
    if (readFile(path, txt) && parse(txt, fp, err) && fp.key == key) {
        std::fprintf(stderr, "spark: fingerprint from cache %s\n", path.c_str());
        return fp;
    }
    fp = Fingerprint{};
    if (!allowRun || !hooks.makeScene) return fp;
    std::fprintf(stderr, "spark: no cached fingerprint for this machine state; measuring (~4 s)...\n");
    txt = run(base, hooks, corunner);
    if (!parse(txt, fp, err)) {
        std::fprintf(stderr, "spark: fingerprint failed: %s\n", err.c_str());
        return Fingerprint{};
    }
    for (size_t i = 1; i <= dir.size(); ++i) // mkdir -p
        if (i == dir.size() || dir[i] == '/') ::mkdir(dir.substr(0, i).c_str(), 0755);
    std::ofstream o(path);
    if (o << txt << "\n") std::fprintf(stderr, "spark: fingerprint cached at %s\n", path.c_str());
    return fp;
}

} // namespace spark::fp
