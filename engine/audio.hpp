// Synthesized sound effects. No asset files: every sound is generated from a
// few parameters (waveform, pitch slide, duration). A background thread mixes
// active voices and streams 16-bit PCM to an external player process
// (pacat or aplay), which avoids linking any audio library. If no player is
// found, audio is silently disabled.
//
//   audio.play({.wave = Wave::Square, .freq = 660, .slide = -300, .dur = 0.1f, .vol = 0.3f});
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace spark {

enum class Wave { Square, Saw, Triangle, Sine, Noise };

struct Sound {
    Wave wave = Wave::Square;
    float freq = 440;   // start pitch, Hz (for Noise: how often the sample changes)
    float slide = 0;    // pitch change over the whole duration, Hz
    float dur = 0.1f;   // seconds
    float vol = 0.3f;   // 0..1
    float delay = 0;    // seconds before it starts
    float duty = 0.5f;  // square wave duty cycle
};

class Audio {
public:
    static constexpr int kRate = 44100;

    Audio() = default;
    ~Audio() { stop(); }
    Audio(const Audio &) = delete;
    Audio &operator=(const Audio &) = delete;

    /** Start the mixer. Returns false (and stays silent) if no player is available. */
    bool start();
    void stop();
    bool available() const { return running_; }

    void play(const Sound &s);
    /** Play notes one after another: {{freq, dur}, ...}; freq 0 = rest. */
    void melody(std::initializer_list<std::pair<float, float>> notes, Wave wave = Wave::Triangle, float vol = 0.2f);

    void setMuted(bool m) { muted_ = m; }
    bool muted() const { return muted_; }
    void setVolume(float v) { volume_ = v; }

    /** Render `n` samples of the current mix (used by the thread; public for tests). */
    void mix(int16_t *out, int n);

private:
    struct Voice {
        Sound s;
        float t = 0;      // seconds since start (negative while delayed)
        float phase = 0;  // 0..1
        float hold = 0;   // current noise sample
        uint32_t seed = 1;
    };
    void run();

    std::mutex mu_;
    std::vector<Voice> incoming_;
    std::vector<Voice> voices_; // mixer thread only
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> muted_{false};
    std::atomic<float> volume_{0.6f};
    FILE *pipe_ = nullptr;
    uint32_t noiseSeed_ = 12345;
};

} // namespace spark
