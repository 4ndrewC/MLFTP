// Spark Warden entry point.
//
//   spark_warden                         play
//   spark_warden --scale 2               bigger window
//   spark_warden --headless --bot --frames 7200 [--seed N] [--shot FRAME:file.ppm]...
//       run with no window, driven by the autopilot; prints a summary.
//   Frame-time research options (research/frametime/PHASE1_REPORT.md): --log, --model,
//   --ctl, --quality, --loop, stress knobs. See usage().
#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "game.hpp"
#include "scenes.hpp"

using namespace warden;

namespace {

void bindControls(Input &in) {
    in.bind(MoveUp, {Key::W});
    in.bind(MoveDown, {Key::S});
    in.bind(MoveLeft, {Key::A});
    in.bind(MoveRight, {Key::D});
    in.bind(AimUp, {Key::Up});
    in.bind(AimDown, {Key::Down});
    in.bind(AimLeft, {Key::Left});
    in.bind(AimRight, {Key::Right});
    in.bind(Fire, {Key::MouseLeft});
    in.bind(Dash, {Key::Space, Key::Shift, Key::MouseRight});
    in.bind(Pause, {Key::Escape, Key::P});
    in.bind(Confirm, {Key::Enter, Key::MouseLeft});
    in.bind(Back, {Key::Escape});
    in.bind(Mute, {Key::M});
}

void usage() {
    std::puts("usage: spark_warden [--scale N] [--seed N] [--mute]\n"
              "                    [--headless] [--bot] [--frames N] [--shot FRAME:path.ppm]...\n"
              "frame-time research (all off by default):\n"
              "  --log PATH             per-frame binary log (research/frametime/DATA_FORMAT.md)\n"
              "  --log-features PATH    also dump each frame's model feature vector (parity checks)\n"
              "  --model DIR|FILE       checkpoint folder (model.spkm) or .spkm file\n"
              "  --ctl MODE             quality controller: fixed (default) | reactive | model | random\n"
              "  --ctl-output NAME      model output compared with the budget (default t_p95_ms)\n"
              "  --budget-ms X          frame budget (default 16.667)\n"
              "  --ctl-margin X         model controller aims for budget x (1 - X) (default 0.10)\n"
              "  --quality S,P,G[,A]    starting quality: scale/particle/glow/AA levels (AA optional, 0 = on)\n"
              "  --quality Ln           or ladder level n (0 = best, 8 = lowest)\n"
              "  --loop MODE            headless timing: fast (default) | emulate | paced\n"
              "  --overshoot-us X       emulate: mean sleep overshoot (default 70)\n"
              "stress knobs (data collection):\n"
              "  --out-scale X          render at X pixels per logical pixel (2 = 1920x1080)\n"
              "  --burst-mult X         particle burst multiplier\n"
              "  --spawn-mult X         enemies per wave / spawn-rate multiplier\n"
              "  --contend N[:BUSY:IDLE] N threads alternating busy/idle (mean ms; run under taskset)\n"
              "  --tag KEY=VALUE        extra session metadata for the log header (repeatable)\n"
              "  --max-seconds S        stop after S seconds of wall time\n"
              "hardware fingerprint (research/frametime/LIT_REVIEW_HW.md):\n"
              "  --fingerprint OUT.json measure this machine/core/power state (~4 s, pin with taskset), write, exit\n"
              "  --fp FILE|none         use this fingerprint for --log/--model (default: cache lookup, measure if missing)\n"
              "  --fp-cache DIR         fingerprint cache (default $SPARK_FP_CACHE or ~/.cache/spark_warden)\n"
              "  --corunner SPEC        virtual hardware: bw:N@cpu,.. (bandwidth hog) or cache:MB@cpu,.. (cache pirate)\n"
              "  --stats                start with the F3 overlay on (predicted vs actual, quality, predictor cost)");
}

double parseNum(const char *s, double lo, double hi, const char *what) {
    char *end = nullptr;
    double v = std::strtod(s, &end);
    if (!end || *end || !(v >= lo && v <= hi)) {
        std::fprintf(stderr, "bad value for %s: %s (range %g..%g)\n", what, s, lo, hi);
        std::exit(2);
    }
    return v;
}

} // namespace

int main(int argc, char **argv) {
    Config cfg;
    cfg.title = "Spark Warden";
    cfg.width = kWidth;
    cfg.height = kHeight;
    cfg.showCursor = false;
    bool bot = false;
    uint64_t seed = 0;
    bool researchSeedSet = false;
    std::vector<std::pair<long, std::string>> shots;
    std::string tags, args, fingerprintOut;
    for (int i = 0; i < argc; ++i) args += std::string(i ? " " : "") + argv[i];

    for (int i = 1; i < argc; ++i) {
        auto arg = [&](const char *name) { return std::strcmp(argv[i], name) == 0; };
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                usage();
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg("--headless")) cfg.headless = true;
        else if (arg("--bot")) bot = true;
        else if (arg("--mute")) cfg.audio = false;
        else if (arg("--stats")) cfg.showStats = true;
        else if (arg("--fingerprint")) fingerprintOut = next();
        else if (arg("--fp")) cfg.fpFile = next();
        else if (arg("--fp-cache")) cfg.fpCacheDir = next();
        else if (arg("--corunner")) cfg.corunner = next();
        else if (arg("--frames")) cfg.maxFrames = std::atol(next());
        else if (arg("--max-seconds")) cfg.maxSeconds = parseNum(next(), 0, 1e7, "--max-seconds");
        else if (arg("--scale")) cfg.windowScale = std::atoi(next());
        else if (arg("--seed")) seed = std::strtoull(next(), nullptr, 10);
        else if (arg("--log")) cfg.logPath = next();
        else if (arg("--log-features")) cfg.featuresPath = next();
        else if (arg("--model")) cfg.modelPath = next();
        else if (arg("--ctl-output")) cfg.ctlOutput = next();
        else if (arg("--ctl")) {
            if (!ft::parseCtl(next(), cfg.controller)) { usage(); return 2; }
        } else if (arg("--budget-ms")) cfg.ctl.budgetMs = parseNum(next(), 0.1, 1000, "--budget-ms");
        else if (arg("--ctl-margin")) cfg.ctl.margin = parseNum(next(), -0.5, 0.9, "--ctl-margin");
        else if (arg("--quality")) {
            if (!parseQuality(next(), cfg.quality)) { usage(); return 2; }
        } else if (arg("--loop")) {
            std::string m = next();
            if (m == "fast") cfg.loop = LoopMode::Fast;
            else if (m == "emulate") cfg.loop = LoopMode::Emulate;
            else if (m == "paced") cfg.loop = LoopMode::Paced;
            else { usage(); return 2; }
        } else if (arg("--overshoot-us")) cfg.overshootUs = parseNum(next(), 0, 100000, "--overshoot-us");
        else if (arg("--out-scale")) cfg.outScale = float(parseNum(next(), 0.25, 4, "--out-scale"));
        else if (arg("--burst-mult")) cfg.burstMult = float(parseNum(next(), 0, 20, "--burst-mult"));
        else if (arg("--spawn-mult")) stress().spawnMult = float(parseNum(next(), 0.1, 20, "--spawn-mult"));
        else if (arg("--contend")) {
            double n = 0, busy = cfg.contendBusyMs, idle = cfg.contendIdleMs;
            if (std::sscanf(next(), "%lf:%lf:%lf", &n, &busy, &idle) < 1 || n < 0 || n > 16 || busy <= 0 || idle <= 0) {
                usage();
                return 2;
            }
            cfg.contendThreads = int(n);
            cfg.contendBusyMs = busy;
            cfg.contendIdleMs = idle;
        } else if (arg("--research-seed")) {
            cfg.researchSeed = std::strtoull(next(), nullptr, 10);
            researchSeedSet = true;
        } else if (arg("--tag")) {
            std::string t = next();
            auto eq = t.find('=');
            if (eq == std::string::npos || eq == 0) { usage(); return 2; }
            tags += ",\"" + jsonEscape(t.substr(0, eq)) + "\":\"" + jsonEscape(t.substr(eq + 1)) + "\"";
        }
        else if (arg("--shot")) {
            std::string s = next();
            auto colon = s.find(':');
            if (colon == std::string::npos) { usage(); return 2; }
            shots.emplace_back(std::atol(s.substr(0, colon).c_str()), s.substr(colon + 1));
        } else if (arg("--help") || arg("-h")) {
            usage();
            return 0;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", argv[i]);
            usage();
            return 2;
        }
    }
    if (cfg.headless && !seed) seed = 1; // headless runs are reproducible by default
    setBaseSeed(seed);
    if (!researchSeedSet) cfg.researchSeed = seed ? seed : 1;
    char sess[256];
    std::snprintf(sess, sizeof sess, "\"game\":\"spark_warden\",\"seed\":%llu,\"bot\":%s,\"spawn_mult\":%g",
                  static_cast<unsigned long long>(seed), bot ? "true" : "false", double(stress().spawnMult));
    cfg.sessionJson = std::string(sess) + ",\"args\":\"" + jsonEscape(args) + "\",\"tags\":{" +
                      (tags.empty() ? "" : tags.substr(1)) + "}";

    // Canned game state for the fingerprint's engine kernels (fixed seed, autopilot).
    cfg.fpHooks.makeScene = [](Engine &e) { return makePlay(e, 424242); };
    cfg.fpHooks.botStep = [](Engine &e, long f) {
        static BotReport r;
        botStep(e, f, r);
    };
    cfg.fpHooks.setSpawnMult = [](float m) {
        static float saved = -1;
        if (m < 0) { // restore
            if (saved >= 0) stress().spawnMult = saved;
            saved = -1;
        } else {
            if (saved < 0) saved = stress().spawnMult;
            stress().spawnMult = m;
        }
    };
    if (!fingerprintOut.empty()) {
        CoRunner co;
        std::string err;
        if (!cfg.corunner.empty() && !co.start(cfg.corunner, err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 2;
        }
        std::string j = fp::run(cfg, cfg.fpHooks, cfg.corunner);
        co.stop();
        fp::Fingerprint f;
        if (!fp::parse(j, f, err)) {
            std::fprintf(stderr, "fingerprint failed: %s\n", err.c_str());
            return 1;
        }
        FILE *out = std::fopen(fingerprintOut.c_str(), "w");
        if (!out) return 1;
        std::fprintf(out, "%s\n", j.c_str());
        std::fclose(out);
        if (!cfg.fpCacheDir.empty()) { // also store it in the cache under its state key
            const std::string &dir = cfg.fpCacheDir;
            for (size_t i = 1; i <= dir.size(); ++i)
                if (i == dir.size() || dir[i] == '/') ::mkdir(dir.substr(0, i).c_str(), 0755);
            if (FILE *c = std::fopen(fp::cachePath(dir, f.key).c_str(), "w")) {
                std::fprintf(c, "%s\n", j.c_str());
                std::fclose(c);
            }
        }
        std::printf("fingerprint %s: %.2f s, %.2f GHz, step %.4f ms, frame %.3f ms (s1) / %.3f ms (s3.2), DRAM %.0f ns "
                    "%.1f GB/s, noise p99 +%.1f us\nkey %s\n",
                    fingerprintOut.c_str(), f.get("duration_s"), f.get("ghz_start"), f.get("step_ms_normal"),
                    f.get("frame_ms_s1"), f.get("frame_ms_s32"), f.get("lat_dram_ns"), f.get("bw_dram_gbs"),
                    f.get("noise_excess_p99_us"), f.key.c_str());
        return 0;
    }

    BotReport report;
    Engine *enginePtr = nullptr;
    cfg.onStep = [&](long frame) {
        Engine &e = *enginePtr;
        for (auto &[at, path] : shots)
            if (frame == at && e.canvas().savePPM(path.c_str())) std::printf("saved %s\n", path.c_str());
        if (bot) botStep(e, frame, report);
    };

    Engine engine(cfg);
    enginePtr = &engine;
    bindControls(engine.input());
    int rc = engine.run(makeTitle(engine));

    if (bot) {
        std::printf("bot: %ld frames (%.0f s simulated), %d deaths, best score %ld, best wave %d, peak entities %zu\n",
                    engine.frame(), engine.time(), report.deaths, report.bestScore, report.bestWave,
                    report.peakEntities);
        if (report.bestWave < 2) {
            std::fprintf(stderr, "bot: never reached wave 2 - gameplay loop looks broken\n");
            return 1;
        }
    }
    return rc;
}
