#include "audio.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <string>

namespace spark {

namespace {

bool onPath(const char *exe) {
    const char *path = std::getenv("PATH");
    if (!path) return false;
    std::string p = path;
    size_t start = 0;
    while (start <= p.size()) {
        size_t end = p.find(':', start);
        if (end == std::string::npos) end = p.size();
        std::string full = p.substr(start, end - start) + "/" + exe;
        if (access(full.c_str(), X_OK) == 0) return true;
        start = end + 1;
    }
    return false;
}

float oscillator(Wave w, float phase, float duty) {
    switch (w) {
    case Wave::Square: return phase < duty ? 1.0f : -1.0f;
    case Wave::Saw: return 2.0f * phase - 1.0f;
    case Wave::Triangle: return phase < 0.5f ? 4.0f * phase - 1.0f : 3.0f - 4.0f * phase;
    case Wave::Sine: return std::sin(phase * 6.2831853f);
    case Wave::Noise: return 0; // handled by the caller
    }
    return 0;
}

} // namespace

bool Audio::start() {
    if (running_) return true;
    const char *cmd = nullptr;
    if (onPath("pacat"))
        cmd = "pacat --raw --format=s16le --rate=44100 --channels=1 --latency-msec=30 2>/dev/null";
    else if (onPath("aplay"))
        cmd = "aplay -q -t raw -f S16_LE -r 44100 -c 1 --buffer-time=50000 2>/dev/null";
    if (!cmd) return false;

    std::signal(SIGPIPE, SIG_IGN); // a dead player must not kill the game
    pipe_ = popen(cmd, "w");
    if (!pipe_) return false;
    // Shrink the pipe buffer so we don't queue up ~0.4s of latency in it.
    fcntl(fileno(pipe_), F_SETPIPE_SZ, 4096);
    running_ = true;
    thread_ = std::thread(&Audio::run, this);
    return true;
}

void Audio::stop() {
    running_ = false; // the thread may also have cleared this itself if the player died
    if (thread_.joinable()) thread_.join();
    if (pipe_) {
        pclose(pipe_);
        pipe_ = nullptr;
    }
}

void Audio::play(const Sound &s) {
    if (!running_ || muted_) return;
    std::lock_guard lock(mu_);
    Voice v;
    v.s = s;
    v.t = -s.delay;
    v.seed = (noiseSeed_ = noiseSeed_ * 1664525u + 1013904223u);
    if (incoming_.size() < 64) incoming_.push_back(v);
}

void Audio::melody(std::initializer_list<std::pair<float, float>> notes, Wave wave, float vol) {
    float delay = 0;
    for (auto [freq, dur] : notes) {
        if (freq > 0) play({.wave = wave, .freq = freq, .dur = dur, .vol = vol, .delay = delay});
        delay += dur;
    }
}

void Audio::mix(int16_t *out, int n) {
    {
        std::lock_guard lock(mu_);
        voices_.insert(voices_.end(), incoming_.begin(), incoming_.end());
        incoming_.clear();
    }
    const float dt = 1.0f / kRate;
    const float master = muted_ ? 0.0f : volume_.load();
    for (int i = 0; i < n; ++i) {
        float acc = 0;
        for (Voice &v : voices_) {
            v.t += dt;
            if (v.t < 0 || v.t >= v.s.dur) continue;
            float p = v.t / v.s.dur; // progress 0..1
            float freq = std::max(10.0f, v.s.freq + v.s.slide * p);
            v.phase += freq * dt;
            bool wrapped = v.phase >= 1.0f;
            v.phase -= std::floor(v.phase);
            float sample;
            if (v.s.wave == Wave::Noise) {
                if (wrapped || v.hold == 0) { // sample-and-hold noise: lower freq = rumblier
                    v.seed = v.seed * 1664525u + 1013904223u;
                    v.hold = float(v.seed >> 8) / 8388608.0f - 1.0f;
                }
                sample = v.hold;
            } else {
                sample = oscillator(v.s.wave, v.phase, v.s.duty);
            }
            // 4ms attack, then a curved decay to silence.
            float env = std::min(1.0f, v.t / 0.004f) * (1 - p) * (1 - p);
            acc += sample * env * v.s.vol;
        }
        acc *= master;
        acc = acc / (1.0f + std::fabs(acc)) * 1.6f; // soft clip so many voices never wrap
        out[i] = int16_t(std::clamp(acc, -1.0f, 1.0f) * 32000.0f);
    }
    std::erase_if(voices_, [](const Voice &v) { return v.t >= v.s.dur; });
}

void Audio::run() {
    int16_t buf[256]; // ~6ms per chunk; the blocking write paces this loop
    int fd = fileno(pipe_);
    while (running_) {
        mix(buf, 256);
        const char *p = reinterpret_cast<const char *>(buf);
        size_t left = sizeof buf;
        while (left > 0 && running_) {
            ssize_t w = write(fd, p, left);
            if (w <= 0) {
                running_ = false; // player died; go silent
                break;
            }
            p += w;
            left -= size_t(w);
        }
    }
}

} // namespace spark
