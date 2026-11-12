// Entity-Component-System.
//
//   World w;
//   Entity e = w.create();
//   w.add<Pos>(e, {10, 20});
//   w.each<Pos, Vel>([&](Entity e, Pos &p, Vel &v) { p.x += v.x * dt; });
//   w.destroy(e);   // deferred until w.flush()
//
// Storage: one sparse set per component type. Component data lives in a
// std::deque, so adding components (even to new entities) during each() never
// invalidates references the callback is holding. Entities created during
// each() are not visited until the next pass. Destruction is deferred, so
// it's always safe to destroy during iteration; dying entities are skipped.
//
// Entity handles carry a generation counter, so a handle to a destroyed
// entity stays invalid even after its slot is reused.
#pragma once

#include <cassert>
#include <cstdint>
#include <deque>
#include <memory>
#include <tuple>
#include <vector>

namespace spark {

struct Entity {
    uint32_t index = UINT32_MAX;
    uint32_t gen = 0;

    explicit operator bool() const { return index != UINT32_MAX; }
    bool operator==(const Entity &) const = default;
};

class World {
public:
    World() = default;
    World(const World &) = delete;
    World &operator=(const World &) = delete;

    // ------------------------------------------------------------ entities

    Entity create() {
        uint32_t idx;
        if (!free_.empty()) {
            idx = free_.back();
            free_.pop_back();
        } else {
            idx = uint32_t(gens_.size());
            gens_.push_back(0);
            state_.push_back(State::Dead);
        }
        state_[idx] = State::Alive;
        ++alive_;
        return {idx, gens_[idx]};
    }

    /** Mark for destruction; actually removed at flush(). */
    void destroy(Entity e) {
        if (!alive(e)) return;
        state_[e.index] = State::Dying;
        pending_.push_back(e.index);
    }

    /** True if the handle refers to a live entity that hasn't been destroyed. */
    bool alive(Entity e) const {
        return e.index < gens_.size() && gens_[e.index] == e.gen && state_[e.index] == State::Alive;
    }

    void flush() {
        // Swap out first: pools may be touched by code reacting to removal.
        std::vector<uint32_t> dying;
        dying.swap(pending_);
        for (uint32_t idx : dying) {
            for (auto &p : pools_)
                if (p) p->remove(idx);
            ++gens_[idx];
            state_[idx] = State::Dead;
            free_.push_back(idx);
            --alive_;
        }
    }

    /** Destroy everything immediately. */
    void clear() {
        for (uint32_t i = 0; i < gens_.size(); ++i)
            if (state_[i] == State::Alive) destroy({i, gens_[i]});
        flush();
    }

    size_t size() const { return alive_; }

    // ------------------------------------------------------------ components

    template <class T> T &add(Entity e, T value = {}) {
        assert(alive(e) && "add() on dead entity");
        return pool<T>().set(e.index, std::move(value));
    }

    template <class T> void remove(Entity e) {
        if (alive(e)) pool<T>().remove(e.index);
    }

    /** Pointer to the component, or nullptr if absent or entity is dead. */
    template <class T> T *get(Entity e) {
        if (!alive(e)) return nullptr;
        auto *p = findPool<T>();
        return p ? p->find(e.index) : nullptr;
    }

    template <class T> bool has(Entity e) { return get<T>(e) != nullptr; }

    /** Number of entities that have component T (including dying ones). */
    template <class T> size_t count() {
        auto *p = findPool<T>();
        return p ? p->dense.size() : 0;
    }

    // ------------------------------------------------------------ queries

    /** Calls f(Entity, A&, B&, ...) for each live entity having all of Ts. */
    template <class First, class... Rest, class F> void each(F &&f) {
        auto *lead = findPool<First>();
        if (!lead) return;
        if constexpr (sizeof...(Rest) > 0)
            if (((findPool<Rest>() == nullptr) || ...)) return;
        // Snapshot the size: entities that gain First during the loop are
        // appended and skipped. Iterate backwards so swap-removes (from
        // remove<First>() inside f) don't skip anyone.
        for (size_t i = lead->dense.size(); i-- > 0;) {
            if (i >= lead->dense.size()) continue;
            uint32_t idx = lead->dense[i];
            if (state_[idx] != State::Alive) continue;
            Entity e{idx, gens_[idx]};
            if constexpr (sizeof...(Rest) == 0) {
                f(e, lead->data[i]);
            } else {
                std::tuple<Rest *...> rest{findPool<Rest>()->find(idx)...};
                if (((std::get<Rest *>(rest) == nullptr) || ...)) continue;
                f(e, lead->data[i], *std::get<Rest *>(rest)...);
            }
        }
    }

    /** First live entity with all of Ts (or a null Entity). */
    template <class... Ts> Entity first() {
        Entity found;
        each<Ts...>([&](Entity e, Ts &...) { if (!found) found = e; });
        return found;
    }

private:
    enum class State : uint8_t { Dead, Alive, Dying };

    struct PoolBase {
        virtual ~PoolBase() = default;
        virtual void remove(uint32_t idx) = 0;
    };

    template <class T> struct Pool final : PoolBase {
        static constexpr uint32_t NONE = UINT32_MAX;
        std::vector<uint32_t> sparse; // entity index -> position in dense/data
        std::vector<uint32_t> dense;  // position -> entity index
        std::deque<T> data;           // deque: stable references on push_back

        T *find(uint32_t idx) {
            return idx < sparse.size() && sparse[idx] != NONE ? &data[sparse[idx]] : nullptr;
        }
        T &set(uint32_t idx, T &&v) {
            if (T *existing = find(idx)) return *existing = std::move(v);
            if (idx >= sparse.size()) sparse.resize(idx + 1, NONE);
            sparse[idx] = uint32_t(dense.size());
            dense.push_back(idx);
            data.push_back(std::move(v));
            return data.back();
        }
        void remove(uint32_t idx) override {
            if (idx >= sparse.size() || sparse[idx] == NONE) return;
            uint32_t pos = sparse[idx], last = uint32_t(dense.size() - 1);
            if (pos != last) {
                data[pos] = std::move(data[last]);
                dense[pos] = dense[last];
                sparse[dense[pos]] = pos;
            }
            data.pop_back();
            dense.pop_back();
            sparse[idx] = NONE;
        }
    };

    static inline size_t nextTypeId_ = 0;
    template <class T> static size_t typeId() {
        static const size_t id = nextTypeId_++;
        return id;
    }

    template <class T> Pool<T> *findPool() {
        size_t id = typeId<T>();
        return id < pools_.size() ? static_cast<Pool<T> *>(pools_[id].get()) : nullptr;
    }

    template <class T> Pool<T> &pool() {
        size_t id = typeId<T>();
        if (id >= pools_.size()) pools_.resize(id + 1);
        if (!pools_[id]) pools_[id] = std::make_unique<Pool<T>>();
        return *static_cast<Pool<T> *>(pools_[id].get());
    }

    std::vector<std::unique_ptr<PoolBase>> pools_;
    std::vector<uint32_t> gens_;
    std::vector<State> state_;
    std::vector<uint32_t> free_;
    std::vector<uint32_t> pending_;
    size_t alive_ = 0;
};

} // namespace spark
