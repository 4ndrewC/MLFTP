#include "profile.hpp"

#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#include "framelog.hpp"
#include "math.hpp"

namespace spark {

bool parseQuality(const std::string &s, Quality &q) {
    if (!s.empty() && (s[0] == 'L' || s[0] == 'l')) {
        int i = std::atoi(s.c_str() + 1);
        if (i < 0 || i >= QualityLadder::kCount) return false;
        q = QualityLadder::kLevels[i];
        return true;
    }
    int a, b, c, d = 0;
    int n = std::sscanf(s.c_str(), "%d,%d,%d,%d", &a, &b, &c, &d);
    if (n != 3 && n != 4) return false;
    if (a < 0 || a >= Quality::kScaleLevels || b < 0 || b >= Quality::kParticleLevels || c < 0 ||
        c >= Quality::kGlowLevels || d < 0 || d >= Quality::kAALevels)
        return false;
    q = {uint8_t(a), uint8_t(b), uint8_t(c), uint8_t(d)};
    return true;
}

// ---------------------------------------------------------------- machine info

std::string readFirstLine(const char *path) {
    std::ifstream f(path);
    std::string line;
    std::getline(f, line);
    return line;
}

/** "AC" / "battery" from /sys/class/power_supply (type Mains, online), "unknown" if there is no adapter. */
std::string powerSource() {
    for (const char *name : {"ADP1", "ADP0", "AC", "ACAD", "AC0"}) {
        std::string base = std::string("/sys/class/power_supply/") + name;
        if (readFirstLine((base + "/type").c_str()) != "Mains") continue;
        std::string on = readFirstLine((base + "/online").c_str());
        if (on == "1") return "AC";
        if (on == "0") return "battery";
    }
    return "unknown";
}

namespace {
/** True if cpu is listed in a sysfs cpu list like "0-7,12". */
bool inCpuList(const std::string &list, int cpu) {
    std::stringstream ss(list);
    std::string part;
    while (std::getline(ss, part, ',')) {
        int a = -1, b = -1;
        if (std::sscanf(part.c_str(), "%d-%d", &a, &b) == 2 ? (cpu >= a && cpu <= b) : (a == cpu)) return true;
    }
    return false;
}
} // namespace

std::string coreType(int cpu) {
    std::string p = readFirstLine("/sys/devices/cpu_core/cpus"), e = readFirstLine("/sys/devices/cpu_atom/cpus");
    if (p.empty() && e.empty()) return "uniform"; // not a hybrid CPU
    if (inCpuList(p, cpu)) return "P";
    if (inCpuList(e, cpu)) return "E";
    return "unknown";
}

std::string platformProfile() {
    // The ACPI read can block for ~0.2 s (and fail with EINVAL on some firmware): read once per process.
    static const std::string p = [] {
        std::string s = readFirstLine("/sys/firmware/acpi/platform_profile");
        return s.empty() ? std::string("unreadable") : s;
    }();
    return p;
}

std::string machineInfoJson() {
    std::string model;
    {
        std::ifstream f("/proc/cpuinfo");
        std::string line;
        while (std::getline(f, line))
            if (line.rfind("model name", 0) == 0) {
                model = line.substr(line.find(':') + 2);
                break;
            }
    }
    long memKb = 0;
    {
        std::ifstream f("/proc/meminfo");
        std::string key;
        f >> key >> memKb;
    }
    utsname u{};
    uname(&u);
    cpu_set_t set;
    CPU_ZERO(&set);
    std::string affinity;
    if (sched_getaffinity(0, sizeof set, &set) == 0)
        for (int i = 0; i < CPU_SETSIZE; ++i)
            if (CPU_ISSET(i, &set)) affinity += (affinity.empty() ? "" : ",") + std::to_string(i);
    int firstCpu = sched_getcpu();
    char path[128];
    std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor", firstCpu);
    std::string gov = readFirstLine(path);
    std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", firstCpu);
    std::string maxf = readFirstLine(path);
    std::string j = "{\"cpu_model\":\"" + jsonEscape(model) + "\",\"logical_cpus\":" +
                    std::to_string(sysconf(_SC_NPROCESSORS_ONLN)) + ",\"mem_kb\":" + std::to_string(memKb) +
                    ",\"kernel\":\"" + jsonEscape(std::string(u.sysname) + " " + u.release) + "\",\"arch\":\"" +
                    jsonEscape(u.machine) + "\",\"affinity\":\"" + affinity + "\",\"start_cpu\":" +
                    std::to_string(firstCpu) + ",\"governor\":\"" + jsonEscape(gov) + "\",\"start_cpu_max_khz\":" +
                    (maxf.empty() ? std::string("0") : maxf);
    // Power source and firmware profile change CPU power limits (and so frame times).
    char host[256] = {};
    gethostname(host, sizeof host - 1);
    j += ",\"hostname\":\"" + jsonEscape(host) + "\",\"power\":\"" + powerSource() + "\",\"platform_profile\":\"" +
         jsonEscape(platformProfile()) + "\",\"start_cpu_type\":\"" + coreType(firstCpu) +
         "\",\"cpu_core_list\":\"" + jsonEscape(readFirstLine("/sys/devices/cpu_core/cpus")) +
         "\",\"cpu_atom_list\":\"" + jsonEscape(readFirstLine("/sys/devices/cpu_atom/cpus")) + "\"}";
    return j;
}

// ---------------------------------------------------------------- per-frame sampling

SysSampler::~SysSampler() {
    if (freqFd_ >= 0) close(freqFd_);
}

void SysSampler::sample(int freqEvery) {
    int c = sched_getcpu();
    cpu = c < 0 ? 0 : c;
    if (frames_++ % freqEvery == 0 || cpu != freqCpu_) {
        if (cpu != freqCpu_) {
            if (freqFd_ >= 0) close(freqFd_);
            char path[96];
            std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", cpu);
            freqFd_ = open(path, O_RDONLY | O_CLOEXEC);
            freqCpu_ = cpu;
        }
        char buf[32];
        ssize_t n = freqFd_ >= 0 ? pread(freqFd_, buf, sizeof buf - 1, 0) : -1;
        if (n > 0) {
            buf[n] = 0;
            mhz = int(std::atol(buf) / 1000);
        }
    }
    rusage ru{};
    if (getrusage(RUSAGE_THREAD, &ru) == 0) {
        ctxsw = lastNivcsw_ < 0 ? 0 : ru.ru_nivcsw - lastNivcsw_;
        minflt = lastMinflt_ < 0 ? 0 : ru.ru_minflt - lastMinflt_;
        lastNivcsw_ = ru.ru_nivcsw;
        lastMinflt_ = ru.ru_minflt;
    }
}

// ---------------------------------------------------------------- co-runners

bool CoRunner::start(const std::string &spec, std::string &err) {
    stop();
    auto at = spec.find('@'), colon = spec.find(':');
    if (colon == std::string::npos || at == std::string::npos || at < colon) {
        err = "co-runner spec must look like bw:2@4,6 or cache:8@4";
        return false;
    }
    std::string kind = spec.substr(0, colon);
    double mb = std::atof(spec.substr(colon + 1, at - colon - 1).c_str());
    std::vector<int> cpus;
    std::stringstream ss(spec.substr(at + 1));
    std::string tok;
    while (std::getline(ss, tok, ',')) cpus.push_back(std::atoi(tok.c_str()));
    if ((kind != "bw" && kind != "cache") || cpus.empty() || mb <= 0) {
        err = "bad co-runner spec '" + spec + "'";
        return false;
    }
    spec_ = spec;
    run_ = true;
    for (int cpu : cpus) {
        threads_.emplace_back([this, kind, mb, cpu] {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpu, &set);
            pthread_setaffinity_np(pthread_self(), sizeof set, &set);
            if (kind == "bw") { // stream-copy two 64 MB buffers (far beyond any L3): bandwidth pressure
                const size_t n = size_t(64) << 20 >> 3;
                std::vector<uint64_t> a(n, 1), b(n, 2);
                while (run_.load(std::memory_order_relaxed)) {
                    std::copy(a.begin(), a.end(), b.begin());
                    std::swap(a, b);
                }
            } else { // cache pirate: random cache-line writes inside `mb` megabytes, keeps them L3-resident
                const size_t lines = std::max<size_t>(1, size_t(mb * 1024 * 1024) / 64);
                std::vector<uint64_t> buf(lines * 8, 0);
                Rng rng(uint64_t(cpu) + 99);
                volatile uint64_t sink = 0;
                while (run_.load(std::memory_order_relaxed))
                    for (int i = 0; i < 4096; ++i) {
                        size_t l = size_t(rng.next() % lines) * 8;
                        buf[l] += 1;
                        sink = sink + buf[l];
                    }
            }
        });
    }
    return true;
}

void CoRunner::stop() {
    run_ = false;
    for (auto &t : threads_) t.join();
    threads_.clear();
}

// ---------------------------------------------------------------- contention

void CpuContention::start(int threads, double meanBusyMs, double meanIdleMs, uint64_t seed) {
    stop();
    run_ = true;
    for (int t = 0; t < threads; ++t) {
        threads_.emplace_back([this, meanBusyMs, meanIdleMs, seed, t] {
            using Clock = std::chrono::steady_clock;
            Rng rng(seed * 31 + uint64_t(t) + 1);
            auto expo = [&](double mean) { return -mean * std::log(1.0 - double(rng.uniform())); };
            volatile uint64_t sink = 0;
            while (run_.load(std::memory_order_relaxed)) {
                auto until = Clock::now() + std::chrono::duration<double, std::milli>(expo(meanBusyMs));
                busy_.fetch_add(1, std::memory_order_relaxed);
                while (Clock::now() < until && run_.load(std::memory_order_relaxed))
                    for (int i = 0; i < 2000; ++i) sink = sink + uint64_t(i) * 2654435761u;
                busy_.fetch_sub(1, std::memory_order_relaxed);
                std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(expo(meanIdleMs)));
            }
        });
    }
}

void CpuContention::stop() {
    run_ = false;
    for (auto &t : threads_) t.join();
    threads_.clear();
}

} // namespace spark
