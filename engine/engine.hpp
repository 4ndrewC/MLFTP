// Engine: owns the window, canvas, input and audio, runs the fixed-timestep
// loop and the scene stack.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "audio.hpp"
#include "canvas.hpp"
#include "controller.hpp"
#include "features.hpp"
#include "fingerprint.hpp"
#include "framelog.hpp"
#include "input.hpp"
#include "math.hpp"
#include "predictor.hpp"
#include "profile.hpp"

namespace spark {

class Engine;
class Window;

/**
 * A screen of the game (title, gameplay, pause menu...). Only the top scene
 * is updated; all scenes in the stack are drawn bottom-to-top, so an overlay
 * like a pause menu shows the frozen game beneath it.
 */
class Scene {
public:
    explicit Scene(Engine &e) : engine(e) {}
    virtual ~Scene() = default;
    virtual void enter() {}
    virtual void exit() {}
    virtual void update(float dt) { (void)dt; }
    virtual void draw(Canvas &cv) { (void)cv; }

    // Frame-time research hooks; called only when logging/prediction is on.
    /** Fill the planned-work ("before") fields of the coming frame. Called bottom-to-top. */
    virtual void planWork(FrameRecord &r) const { (void)r; }
    /** Fill outcome ("after") fields once the frame has run. */
    virtual void reportWork(FrameRecord &r) { (void)r; }

protected:
    Engine &engine;
};

/** Headless loop timing (windowed runs always use the real clock). */
enum class LoopMode {
    Fast,    // one step + one render per iteration, as fast as possible (default)
    Emulate, // no sleeping, but the step count comes from a virtual clock that advances by
             // max(measured frame work, step) + sleep overshoot, like the windowed loop
    Paced,   // real clock and real sleeps, like the windowed loop, but no window
};

struct Config {
    std::string title = "Spark";
    int width = 960, height = 540;  // logical resolution
    int windowScale = 1;            // initial window size multiplier
    double step = 1.0 / 60.0;       // fixed update step (seconds)
    bool headless = false;          // no window/audio; run as fast as possible
    long maxFrames = 0;             // stop after N updates (0 = no limit)
    double maxSeconds = 0;          // stop after this much wall time (0 = no limit)
    bool showCursor = true;
    bool audio = true;
    bool showStats = false;         // start with the F3 stats overlay on
    std::function<void(long frame)> onStep; // runs before each update (bots, replays)

    // Frame-time research (research/frametime). Everything here is off by default.
    LoopMode loop = LoopMode::Fast;   // headless timing mode
    double overshootUs = 70;          // Emulate: mean sleep overshoot (floor 0.8x + exponential tail)
    std::string logPath;              // per-frame binary log (DATA_FORMAT.md)
    std::string featuresPath;         // debug: per-frame feature vectors (parity checks)
    std::string modelPath;            // checkpoint folder or .spkm file
    std::string ctlOutput = "t_p95_ms"; // model output the controller compares with the budget
    ft::CtlMode controller = ft::CtlMode::Fixed;
    ft::CtlParams ctl;                // budget, margins, hysteresis
    Quality quality;                  // starting quality levels
    float outScale = 1;               // stress: pixels per logical unit before the quality scale
    float burstMult = 1;              // stress: particle burst multiplier
    int contendThreads = 0;           // stress: artificial CPU contention threads
    double contendBusyMs = 4, contendIdleMs = 12;
    uint64_t researchSeed = 1;        // emulation / contention / random-controller RNG
    std::string corunner;             // stress: "virtual hardware" co-runner threads (CoRunner spec)
    std::string fpFile;               // hardware fingerprint JSON, "none" to disable; empty = cache lookup
    std::string fpCacheDir;           // fingerprint cache dir (empty = fp::defaultCacheDir())
    bool fpAllowRun = true;           // measure a fingerprint (~4 s) if none is cached for this machine state
    fp::Hooks fpHooks;                // game-provided canned scene for the fingerprint kernels
    std::string sessionJson;          // extra "key":value members for the log header
};

class Engine {
public:
    explicit Engine(Config cfg);
    ~Engine();

    /** Run until quit() / window closed / maxFrames. Returns a process exit code. */
    int run(std::unique_ptr<Scene> first);
    void quit() { quit_ = true; }

    // Scene changes are deferred until the end of the current update, so a
    // scene can safely replace itself from inside its own update().
    void switchTo(std::unique_ptr<Scene> s);
    void push(std::unique_ptr<Scene> s);
    void pop();
    Scene *top() const { return scenes_.empty() ? nullptr : scenes_.back().get(); }

    template <class S, class... Args> void switchTo(Args &&...args) {
        switchTo(std::make_unique<S>(*this, std::forward<Args>(args)...));
    }
    template <class S, class... Args> void push(Args &&...args) {
        push(std::make_unique<S>(*this, std::forward<Args>(args)...));
    }

    Canvas &canvas() { return canvas_; }
    Input &input() { return input_; }
    Audio &audio() { return audio_; }
    const Config &config() const { return cfg_; }
    int width() const { return cfg_.width; }
    int height() const { return cfg_.height; }

    long frame() const { return frame_; }
    double time() const { return time_; }
    int fps() const { return fps_; }
    bool headless() const { return cfg_.headless; }

    /** Add camera-shake trauma (0..1). Decays automatically. */
    void shake(float trauma) { trauma_ = std::min(1.0f, trauma_ + trauma); }
    /** Freeze updates for a moment (hit-stop) while still rendering. */
    void hitstop(float seconds) { hitstop_ = std::max(hitstop_, seconds); }

    /** Draw one frame now (for tools/tests that need the current picture). */
    void render();
    /** Tools (fingerprint kernels): make `s` the only scene without running the loop, then advance n steps. */
    void startScene(std::unique_ptr<Scene> s) {
        switchTo(std::move(s));
        applySceneChanges();
    }
    void advance(int steps) {
        for (int i = 0; i < steps && !scenes_.empty(); ++i) step();
    }

    // Frame-time research.
    Profiler &prof() { return prof_; }
    const Quality &quality() const { return quality_; }
    void setQuality(const Quality &q);
    /** Burst particle multiplier scenes should apply: quality density x stress multiplier. */
    float particleScale() const { return quality_.particleDensity() * cfg_.burstMult; }
    const ft::Model *model() const { return model_.get(); }
    const fp::Fingerprint &fingerprint() const { return fp_; }
    const std::string &modelError() const { return modelError_; }
    const FrameRecord &lastRecord() const { return rec_; }

private:
    void step();
    void applySceneChanges();
    void setupResearch();
    void beginFrame(int steps, double acc);
    void endFrame(double totalUs, double stepsUs, double renderUs, double presentUs, double sleepReqUs,
                  double sleepUs);
    void drawStats();

    struct Change {
        enum Kind { Switch, Push, Pop } kind;
        std::unique_ptr<Scene> scene;
    };

    Config cfg_;
    Canvas canvas_;
    Input input_;
    Audio audio_;
    std::unique_ptr<Window> window_;
    std::vector<std::unique_ptr<Scene>> scenes_;
    std::vector<Change> changes_;
    bool quit_ = false;
    long frame_ = 0;
    double time_ = 0;
    int fps_ = 0;
    bool showStats_ = false;
    float trauma_ = 0, hitstop_ = 0;
    Vec2 shakeOffset_;
    Rng rng_{0xC0FFEE};
    double frameMs_ = 0; // CPU time of the last update+render, for the stats overlay

    // Frame-time research state.
    Profiler prof_;
    SysSampler sys_;
    CpuContention contention_;
    CoRunner corunner_;
    fp::Fingerprint fp_;
    long powerPoll_ = 0;
    bool onAC_ = true;
    TableWriter log_, featLog_;
    Quality quality_;
    std::unique_ptr<ft::Model> model_;
    std::string modelError_;
    ft::Controller ctl_;
    ft::FeatureState feat_;
    ft::HistoryRing hist_;
    ft::FrameContext fctx_;
    FrameRecord rec_{};
    float x_[ft::kNumFeatures] = {}, xc_[ft::kNumFeatures] = {}, out_[16] = {};
    int outCtl_ = -1, outP50_ = -1, outP95_ = -1, outP99_ = -1, outPmiss_ = -1;
    bool instrumented_ = false, needFeatures_ = false;
    uint32_t iter_ = 0;
    double preLogUs_ = 0, lastLogUs_ = 0, lastTotalMs_ = 0, presentEma_ = 0, pendingPredUs_ = 0;
    Rng emuRng_{1};
    static constexpr int kGraph = 120; // overlay graph length
    float graphAct_[kGraph] = {}, graphPred_[kGraph] = {};
    int graphPos_ = 0;
};

} // namespace spark
