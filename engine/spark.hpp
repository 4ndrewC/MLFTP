// Spark Engine — a small 2D game engine in C++20.
//
// Include this one header to get everything:
//   math.hpp       Vec2, Color, Rng, easing
//   ecs.hpp        World / Entity (sparse-set ECS)
//   canvas.hpp     software renderer (AA shapes, glow, bitmap font)
//   input.hpp      keyboard/mouse + named actions
//   collision.hpp  spatial hash + circle collisions with layers/masks
//   particles.hpp  particle pool
//   audio.hpp      synthesized sound effects
//   engine.hpp     Engine (loop, scenes, shake, hit-stop) and Scene
#pragma once

#include "audio.hpp"
#include "canvas.hpp"
#include "collision.hpp"
#include "ecs.hpp"
#include "engine.hpp"
#include "input.hpp"
#include "math.hpp"
#include "particles.hpp"
