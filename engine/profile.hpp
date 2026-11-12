// Frame-time research support: subsystem timers, quality levels, machine
// state sampling and the artificial CPU-contention stress knob.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace spark {

/** Timed subsystems. Game code wraps work in ProfScope; the engine reads the totals per frame. */
enum class Prof : uint8_t { OnStep, Update, Collision, ParticlesUpd, Flush, RBackground, RWorld, RParticles, RHud, Count };

class Profiler {
public:
    bool enabled = false; // when false, ProfScope does not even read the clock
    float us[size_t(Prof::Count)] = {};
    void reset() {
        for (float &v : us) v = 0;
    }
};

class ProfScope {
public:
    using Clock = std::chrono::steady_clock;
    ProfScope(Profiler &p, Prof slot) : p_(p), slot_(slot) {
        if (p_.enabled) t0_ = Clock::now();
    }
    ~ProfScope() {
        if (p_.enabled) p_.us[size_t(slot_)] += std::chrono::duration<float, std::micro>(Clock::now() - t0_).count();
    }
    ProfScope(const ProfScope &) = delete;
    ProfScope &operator=(const ProfScope &) = delete;

private:
    Profiler &p_;
    Prof slot_;
    Clock::time_point t0_;
};

/**
 * Runtime quality levers, each with discrete levels (0 = best). The controller
 * moves along a ladder of combinations; data collection also samples
 * off-ladder combinations.
 */
struct Quality {
    static constexpr float kScale[] = {1.0f, 0.8f, 0.65f, 0.5f};   // render scale (internal resolution)
    static constexpr float kParticles[] = {1.0f, 0.6f, 0.35f, 0.15f}; // burst particle density
    static constexpr float kGlow[] = {1.0f, 0.75f, 0.5f, 0.0f};     // glow radius multiplier (0 = off)
    static constexpr float kAA[] = {1.0f, 0.0f};                     // anti-aliasing on / off
    static constexpr int kScaleLevels = 4, kParticleLevels = 4, kGlowLevels = 4, kAALevels = 2;

    uint8_t scale = 0, particles = 0, glow = 0, aa = 0;

    float renderScale() const { return kScale[scale]; }
    float particleDensity() const { return kParticles[particles]; }
    float glowScale() const { return kGlow[glow]; }
    bool antialias() const { return aa == 0; }
    bool operator==(const Quality &) const = default;
};

/**
 * Quality ladder, best first, levels (scale, particles, glow, aa). Measured at
 * out-scale 2 (PHASE1_REPORT.md): render scale and glow are the big levers,
 * anti-aliasing a medium one (large once glow is reduced), particle density a
 * small one. Each step lowers at least one lever and raises none.
 */
struct QualityLadder {
    static constexpr Quality kLevels[] = {
        {0, 0, 0, 0}, {0, 1, 1, 0}, {0, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 2, 1},
        {2, 2, 2, 1}, {2, 2, 3, 1}, {3, 2, 3, 1}, {3, 3, 3, 1},
    };
    static constexpr int kCount = int(sizeof kLevels / sizeof kLevels[0]);
    /** Ladder index of q, or -1 if q is not on the ladder. */
    static int find(const Quality &q) {
        for (int i = 0; i < kCount; ++i)
            if (kLevels[i] == q) return i;
        return -1;
    }
};

/** Parse "S,P,G[,A]" (levels; A defaults to 0 = AA on) or "L<n>" (ladder index). False on bad input. */
bool parseQuality(const std::string &s, Quality &q);

// ---------------------------------------------------------------- machine state

/** Machine description for log headers (CPU model, cores, kernel, governor, RAM). JSON object. */
std::string machineInfoJson();
/** First line of a (sysfs) file, "" if unreadable. */
std::string readFirstLine(const char *path);
/** /sys/firmware/acpi/platform_profile, read once per process ("unreadable" if it fails). */
std::string platformProfile();
/** "AC", "battery" or "unknown" (laptops change CPU power limits on battery). */
std::string powerSource();
/** "P" or "E" on hybrid Intel CPUs (sysfs cpu_core / cpu_atom lists), "uniform" otherwise. */
std::string coreType(int cpu);

/** Cheap per-frame system sampling: CPU id, core frequency, context switches, page faults. */
class SysSampler {
public:
    ~SysSampler();
    /** Call once per frame before it runs. Frequency is re-read every `freqEvery` calls. */
    void sample(int freqEvery = 8);
    int cpu = 0;
    int mhz = 0;
    long ctxsw = 0, minflt = 0; // deltas since the previous sample

private:
    int freqFd_ = -1, freqCpu_ = -1;
    long frames_ = 0;
    long lastNivcsw_ = -1, lastMinflt_ = -1;
};

/**
 * Stress knob for data collection: background threads that alternate busy
 * spinning and sleeping (exponentially distributed periods). Run the game under
 * `taskset -c N` and they compete with the main thread for that one core, the
 * way other processes do on a loaded machine.
 */
/**
 * "Virtual hardware" co-runners (LIT_REVIEW_HW §3.4): threads pinned to OTHER cores
 * that either stream through a large buffer (memory-bandwidth hog, Bandwidth-Bandit
 * style) or randomly touch an x-MB working set (cache pirate: shrinks the effective
 * L3). Spec: "bw:<threads>@cpu,cpu" or "cache:<MB>@cpu,cpu" (one thread per cpu).
 */
class CoRunner {
public:
    ~CoRunner() { stop(); }
    bool start(const std::string &spec, std::string &err);
    void stop();
    const std::string &spec() const { return spec_; }

private:
    std::vector<std::thread> threads_;
    std::atomic<bool> run_{false};
    std::string spec_;
};

class CpuContention {
public:
    ~CpuContention() { stop(); }
    void start(int threads, double meanBusyMs, double meanIdleMs, uint64_t seed);
    void stop();
    bool busy() const { return busy_.load(std::memory_order_relaxed) > 0; }

private:
    std::vector<std::thread> threads_;
    std::atomic<bool> run_{false};
    std::atomic<int> busy_{0};
};

} // namespace spark
