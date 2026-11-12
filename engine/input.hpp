// Keyboard + mouse state with named actions.
//
//   input.bind(Fire, {Key::Space, Key::MouseLeft});
//   if (input.down(Fire)) ...      // held
//   if (input.pressed(Fire)) ...   // went down since the previous update step
//
// Edge states (pressed/released) are cleared after every fixed update, so a
// tap is seen by exactly one update even when several run per frame.
#pragma once

#include <array>
#include <bitset>
#include <initializer_list>
#include <vector>

#include "math.hpp"

namespace spark {

enum class Key : uint8_t {
    None,
    A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
    Num0, Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9,
    Space, Enter, Escape, Tab, Backspace,
    Up, Down, Left, Right,
    Shift, Ctrl, Alt,
    F1, F2, F3, F4,
    MouseLeft, MouseMiddle, MouseRight,
    Count
};

class Input {
public:
    static constexpr size_t kKeys = size_t(Key::Count);

    bool down(Key k) const { return held_[size_t(k)]; }
    bool pressed(Key k) const { return pressed_[size_t(k)]; }
    bool released(Key k) const { return released_[size_t(k)]; }
    bool anyPressed() const { return pressed_.any(); }

    /** Bind keys to an action. Actions are any integer-like id (e.g. a game enum). */
    template <class A> void bind(A action, std::initializer_list<Key> keys) {
        size_t i = size_t(action);
        if (i >= actions_.size()) actions_.resize(i + 1);
        actions_[i].insert(actions_[i].end(), keys);
    }
    template <class A> bool down(A action) const { return anyOf(action, held_); }
    template <class A> bool pressed(A action) const { return anyOf(action, pressed_); }
    template <class A> bool released(A action) const { return anyOf(action, released_); }
    /** -1, 0 or +1 from two opposing actions. */
    template <class A> float axis(A negative, A positive) const {
        return float(down(positive)) - float(down(negative));
    }

    Vec2 mouse() const { return mouse_; }

    // State changes (from the platform layer, or scripted for bots/tests).
    void setKey(Key k, bool isDown) {
        size_t i = size_t(k);
        if (isDown && !held_[i]) pressed_[i] = true;
        if (!isDown && held_[i]) released_[i] = true;
        held_[i] = isDown;
    }
    void setMouse(Vec2 p) { mouse_ = p; }
    /** Release everything (e.g. window lost focus). */
    void releaseAll() {
        released_ |= held_;
        held_.reset();
    }
    void endStep() {
        pressed_.reset();
        released_.reset();
    }

private:
    template <class A> bool anyOf(A action, const std::bitset<kKeys> &set) const {
        size_t i = size_t(action);
        if (i >= actions_.size()) return false;
        for (Key k : actions_[i])
            if (set[size_t(k)]) return true;
        return false;
    }

    std::bitset<kKeys> held_, pressed_, released_;
    std::vector<std::vector<Key>> actions_;
    Vec2 mouse_;
};

} // namespace spark
