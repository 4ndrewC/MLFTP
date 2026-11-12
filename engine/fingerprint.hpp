// Hardware fingerprint "spark-hw-v1" (research/frametime/LIT_REVIEW_HW.md §3.2).
//
// A <= ~5 s, no-root microbenchmark run by the engine binary itself (`--fingerprint`):
//   F1 effective GHz, F2 pointer-chase latency, F3 streaming bandwidth, F4 FP/int
//   throughput, F5 branch-miss cost, F6 first-touch cost, F7 noise (FWQ) + sleep
//   overshoot, F8 raster kernels (per primitive class, AA on/off, and a canned game
//   frame at out-scale 1.0 and 3.2), F9 sim kernel (normal and heavy load), F10
//   particle kernel, F11 state key.
// Every timed entry is the median of >= 5 repeats with its relative IQR.
//
// The engine turns the engine-kernel entries into per-subsystem reference times
// (t_ref, feature set spark-ft-v2), the PMaC/Funkhouser-style "machine rates x
// planned work" normaliser recommended by LIT_REVIEW_HW §3.3 option A, plus noise
// terms for option B.
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>

namespace spark {

class Engine;
class Scene;
struct Config;

namespace fp {

inline constexpr const char *kFormat = "spark-hw-v1";

/** Fingerprint values the features use (all doubles, parsed from the JSON for exact parity). */
struct Fingerprint {
    bool valid = false;
    std::string key, json; // state key and the full JSON record
    std::map<std::string, double> v; // every fp_* number by name
    double get(const std::string &k, double def = 0) const {
        auto it = v.find(k);
        return it == v.end() ? def : it->second;
    }
};

/** What the game provides so F8/F9 can use a real (canned) game state. */
struct Hooks {
    std::function<std::unique_ptr<Scene>(Engine &)> makeScene; // fixed-seed gameplay scene
    std::function<void(Engine &, long)> botStep;                // drives input each step
    std::function<void(float)> setSpawnMult;                     // heavy-load variant for F9
};

/** The state key: host, CPU, core type, power, platform profile, EPP, max freq, co-runner, binary. */
std::string stateKey(const std::string &corunner);
/** Run all probes (pinned to the current CPU) and return the JSON record. */
std::string run(const Config &base, const Hooks &hooks, const std::string &corunner);
/** Parse a JSON record produced by run(). */
bool parse(const std::string &json, Fingerprint &out, std::string &err);
/** Cache file for a key inside `dir`. */
std::string cachePath(const std::string &dir, const std::string &key);
/** Default cache dir: $SPARK_FP_CACHE, else $XDG_CACHE_HOME/spark_warden, else ~/.cache/spark_warden. */
std::string defaultCacheDir();
/**
 * Resolve the fingerprint for this process: `file` if given ("none" disables),
 * else the cache entry for the current state key, else run the probes (<= ~5 s)
 * and store them in the cache. Prints what it did to stderr.
 */
Fingerprint resolve(const std::string &file, const std::string &cacheDir, const Config &base, const Hooks &hooks,
                    const std::string &corunner, bool allowRun);

} // namespace fp
} // namespace spark
