// Tiny self-contained test harness shared by the test files: no framework dependency.
#pragma once

#include <cstdio>
#include <functional>
#include <utility>
#include <vector>

namespace testing {
inline int gFailures = 0;
inline int gChecks = 0;
inline std::vector<std::pair<const char *, std::function<void()>>> &registry() {
    static std::vector<std::pair<const char *, std::function<void()>>> r;
    return r;
}
struct Register {
    Register(const char *name, std::function<void()> fn) { registry().emplace_back(name, std::move(fn)); }
};
} // namespace testing

#define TEST(name)                                         \
    static void name();                                    \
    static testing::Register reg_##name(#name, name);      \
    static void name()

#define CHECK(cond)                                                                   \
    do {                                                                              \
        ++testing::gChecks;                                                           \
        if (!(cond)) {                                                                \
            ++testing::gFailures;                                                     \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);             \
        }                                                                             \
    } while (0)
