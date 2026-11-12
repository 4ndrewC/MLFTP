// Spark Drift entry point.
//   spark_drift                         play (opens a window: for people, not automated runs)
//   spark_drift --headless --bot --frames 7200 [--seed N] [--shot FRAME:file.ppm]
//       no window; the autopilot plays; prints one summary line and exits 1 if it never reaches wave 2.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "drift.hpp"

using namespace drift;

namespace {

void bindControls(Input &in) {
    in.bind(ActUp, {Key::W, Key::Up});
    in.bind(ActDown, {Key::S, Key::Down});
    in.bind(ActLeft, {Key::A, Key::Left});
    in.bind(ActRight, {Key::D, Key::Right});
    in.bind(ActBoost, {Key::Space, Key::Shift});
    in.bind(ActPause, {Key::P, Key::Escape});
    in.bind(ActConfirm, {Key::Enter, Key::MouseLeft});
    in.bind(ActBack, {Key::Escape});
    in.bind(ActMute, {Key::M});
}

void usage() {
    std::puts("usage: spark_drift [--seed N] [--scale N] [--mute]\n"
              "                  [--headless] [--bot] [--frames N] [--shot FRAME:path.ppm]...\n"
              "  --headless      no window or audio; runs as fast as possible (default 7200 frames)\n"
              "  --bot           autopilot; prints one summary line at the end\n"
              "  --frames N      stop after N updates (0 = no limit when windowed)\n"
              "  --seed N        world seed (headless default 1)\n"
              "  --scale N       initial window scale (windowed only, 1..4)\n"
              "  --mute          no sound\n"
              "  --shot F:PATH   save the rendered frame F as a binary PPM (repeatable)\n"
              "  --stats         start with the F3 overlay on (frame time, quality, controller)\n"
              "quality control (engine levers; in Spark Drift: scale = internal resolution,\n"
              "glow = view distance, particles = ground detail, aa = terrain detail):\n"
              "  --ctl MODE      fixed (default) | reactive. (model: not yet; the trained models are for Spark Warden)\n"
              "  --quality Q     starting quality: S,P,G[,A] levels or ladder level Ln (0 = best, 8 = lowest)\n"
              "  --budget-ms X   frame budget for the controller (default 16.667)\n"
              "  --out-scale X   stress: render at X pixels per logical pixel (0.25..4)\n"
              "  --loop MODE     headless timing: fast (default) | emulate | paced (real 60 Hz with sleeps)\n"
              "  --max-seconds S stop after S seconds of wall time\n"
              "  --log PATH      per-frame binary log (research/frametime/DATA_FORMAT.md). Frame times,\n"
              "                  quality and controller fields are valid; the model-feature fields describe\n"
              "                  Spark Warden and are not meaningful here");
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
    cfg.title = "Spark Drift";
    cfg.width = kWidth;
    cfg.height = kHeight;
    cfg.showCursor = false;
    bool bot = false;
    uint64_t seed = 0;
    std::vector<std::pair<long, std::string>> shots;

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
        else if (arg("--frames")) cfg.maxFrames = long(parseNum(next(), 0, 1e7, "--frames"));
        else if (arg("--seed")) seed = std::strtoull(next(), nullptr, 10);
        else if (arg("--scale")) cfg.windowScale = int(parseNum(next(), 1, 4, "--scale"));
        else if (arg("--stats")) cfg.showStats = true;
        else if (arg("--ctl")) {
            std::string m = next();
            if (m == "model") {
                std::fprintf(stderr, "spark_drift: --ctl model is not supported yet (no model is trained for this game)\n");
                return 2;
            }
            if (m != "fixed" && m != "reactive") {
                usage();
                return 2;
            }
            ft::parseCtl(m.c_str(), cfg.controller);
        } else if (arg("--quality")) {
            if (!parseQuality(next(), cfg.quality)) {
                usage();
                return 2;
            }
        } else if (arg("--budget-ms")) cfg.ctl.budgetMs = parseNum(next(), 0.1, 1000, "--budget-ms");
        else if (arg("--out-scale")) cfg.outScale = float(parseNum(next(), 0.25, 4, "--out-scale"));
        else if (arg("--max-seconds")) cfg.maxSeconds = parseNum(next(), 0, 1e7, "--max-seconds");
        else if (arg("--log")) cfg.logPath = next();
        else if (arg("--loop")) {
            std::string m = next();
            if (m == "fast") cfg.loop = LoopMode::Fast;
            else if (m == "emulate") cfg.loop = LoopMode::Emulate;
            else if (m == "paced") cfg.loop = LoopMode::Paced;
            else {
                usage();
                return 2;
            }
        }
        else if (arg("--shot")) {
            std::string s = next();
            auto colon = s.find(':');
            if (colon == std::string::npos) {
                usage();
                return 2;
            }
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

    if (seed == 0) {
        seed = cfg.headless ? 1 : uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
    }
    setBaseSeed(seed);
    // The hardware fingerprint's benchmark scene is Spark Warden's; this game has none, so never measure one.
    cfg.fpFile = "none";
    cfg.fpAllowRun = false;
    if (cfg.headless && cfg.maxFrames == 0) cfg.maxFrames = 7200;

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
    const int rc = engine.run(std::make_unique<TitleScene>(engine));

    if (bot) {
        const DrawTotals &t = drawTotals();
        const double perFrame = t.frames ? double(t.triangles) / double(t.frames) : 0.0;
        std::printf("bot: %ld frames (%.1f s simulated), score %ld, wave %d, deaths %d, "
                    "triangles drawn per frame %.1f (avg over %ld 3D frames)\n",
                    engine.frame(), engine.time(), report.bestScore, report.bestWave, report.deaths, perFrame,
                    t.frames);
        if (report.bestWave < 2) {
            std::fprintf(stderr, "bot: never reached wave 2 - gameplay loop looks broken\n");
            return 1;
        }
    }
    return rc;
}
