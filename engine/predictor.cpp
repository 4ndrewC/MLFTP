#include "predictor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <sstream>

namespace spark::ft {

uint32_t crc32(const unsigned char *data, size_t n, uint32_t crc) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        init = true;
    }
    crc = ~crc;
    for (size_t i = 0; i < n; ++i) crc = table[(crc ^ data[i]) & 0xff] ^ (crc >> 8);
    return ~crc;
}

namespace {

constexpr int kMaxOutputs = 16;

inline float sigmoid(float v) { return 1.0f / (1.0f + std::exp(-v)); }

inline float activate(int act, float v) {
    switch (act) {
    case 1: return v > 0 ? v : 0;
    case 2: return std::tanh(v);
    case 3: return sigmoid(v);
    default: return v;
    }
}

struct Dense {
    int nIn = 0, nOut = 0, act = 0;
    std::vector<float> W, b;
    void forward(const float *in, float *out) const {
        for (int o = 0; o < nOut; ++o) {
            const float *w = &W[size_t(o) * nIn];
            float acc = b[size_t(o)];
            for (int i = 0; i < nIn; ++i) acc += w[i] * in[i];
            out[o] = activate(act, acc);
        }
    }
};

/** Runs a stack of dense layers; returns a pointer to the final activations. */
struct DenseStack {
    std::vector<Dense> layers;
    std::vector<float> a, b;
    void prepare() {
        size_t m = 0;
        for (auto &l : layers) m = std::max({m, size_t(l.nIn), size_t(l.nOut)});
        a.assign(m, 0);
        b.assign(m, 0);
    }
    const float *run(const float *in) {
        const float *cur = in;
        float *dst = a.data();
        for (auto &l : layers) {
            l.forward(cur, dst);
            cur = dst;
            dst = dst == a.data() ? b.data() : a.data();
        }
        return cur;
    }
    int inDim() const { return layers.empty() ? 0 : layers.front().nIn; }
    int outDim() const { return layers.empty() ? 0 : layers.back().nOut; }
};

// ---------------------------------------------------------------- families

struct LinearModel : Model {
    Dense lin;
    void raw(const float *xn, const float *, float *out) override { lin.forward(xn, out); }
};

struct MlpModel : Model {
    DenseStack net;
    void raw(const float *xn, const float *, float *out) override {
        const float *y = net.run(xn);
        std::copy(y, y + net.outDim(), out);
    }
};

struct GbtModel : Model {
    struct Node {
        int32_t feat;
        float thr;
        int32_t left, right;
        float value;
        uint32_t flags; // bit0: missing (NaN) goes left
    };
    struct Tree {
        uint32_t out;
        std::vector<Node> nodes;
    };
    std::vector<float> base;
    std::vector<Tree> trees;
    void raw(const float *xn, const float *, float *out) override {
        for (size_t o = 0; o < base.size(); ++o) out[o] = base[o];
        for (const Tree &t : trees) {
            const Node *nodes = t.nodes.data();
            int32_t i = 0;
            while (nodes[i].feat >= 0) {
                const Node &nd = nodes[i];
                float v = xn[nd.feat];
                bool left = std::isnan(v) ? (nd.flags & 1u) : v <= nd.thr;
                i = left ? nd.left : nd.right;
            }
            out[t.out] += nodes[i].value;
        }
    }
};

struct RlsModel : Model {
    std::vector<uint32_t> idx;
    int baseIdx = -1;
    double lambda = 0.999, maxTrace = 1e6, gamma = 0.01, varAlpha = 0.01, var = 1;
    bool online = true;
    std::vector<double> w, P, qLevel, qOff, phi, Pphi;
    std::vector<int> outKind; // -1 mean, -2 p_miss, >=0 quantile index
    size_t dim() const { return idx.size() + 1; }
    void features(const float *xn) {
        for (size_t i = 0; i < idx.size(); ++i) phi[i] = double(xn[idx[i]]);
        phi[idx.size()] = 1.0;
    }
    double mean(const float *x) const {
        double m = 0;
        for (size_t i = 0; i < dim(); ++i) m += w[i] * phi[i];
        if (baseIdx >= 0) m += double(x[baseIdx]);
        return m;
    }
    void raw(const float *xn, const float *x, float *out) override {
        features(xn);
        double m = mean(x);
        for (size_t o = 0; o < outKind.size(); ++o) {
            int k = outKind[o];
            if (k == -1) out[o] = float(m);
            else if (k == -2) out[o] = float(0.5 * std::erfc((budgetMs_ - m) / (std::sqrt(std::max(var, 1e-12)) * std::sqrt(2.0))));
            else out[o] = float(m + qOff[size_t(k)]);
        }
    }
    void observe(const float *xn, const float *x, const float *, float yMs) override {
        if (!online) return;
        features(xn);
        const size_t n = dim();
        double e = double(yMs) - mean(x);
        for (size_t q = 0; q < qOff.size(); ++q) qOff[q] += gamma * ((e > qOff[q] ? 1.0 : 0.0) - (1.0 - qLevel[q]));
        var = (1 - varAlpha) * var + varAlpha * e * e;
        double denom = lambda;
        for (size_t i = 0; i < n; ++i) {
            double s = 0;
            for (size_t j = 0; j < n; ++j) s += P[i * n + j] * phi[j];
            Pphi[i] = s;
            denom += phi[i] * s;
        }
        double trace = 0;
        for (size_t i = 0; i < n; ++i) {
            double k = Pphi[i] / denom;
            w[i] += k * e;
            for (size_t j = 0; j < n; ++j) P[i * n + j] = (P[i * n + j] - k * Pphi[j]) / lambda;
            trace += P[i * n + i];
        }
        if (trace > maxTrace)
            for (double &v : P) v *= maxTrace / trace;
    }
};

struct GruModel : Model {
    int D = 0, H = 0;
    bool stream = false;
    std::vector<float> Wih, Whh, bih, bhh, h, hTmp, gi, gh, headIn;
    DenseStack head;
    void step(const float *xin, float *hs) {
        const int G = 3 * H;
        for (int g = 0; g < G; ++g) {
            float a = bih[size_t(g)], c = bhh[size_t(g)];
            const float *wi = &Wih[size_t(g) * D], *wh = &Whh[size_t(g) * H];
            for (int i = 0; i < D; ++i) a += wi[i] * xin[i];
            for (int i = 0; i < H; ++i) c += wh[i] * hs[i];
            gi[size_t(g)] = a;
            gh[size_t(g)] = c;
        }
        for (int j = 0; j < H; ++j) {
            float r = sigmoid(gi[size_t(j)] + gh[size_t(j)]);
            float z = sigmoid(gi[size_t(H + j)] + gh[size_t(H + j)]);
            float n = std::tanh(gi[size_t(2 * H + j)] + r * gh[size_t(2 * H + j)]);
            hTmp[size_t(j)] = (1 - z) * n + z * hs[j];
        }
        std::copy(hTmp.begin(), hTmp.end(), hs);
    }
    void encode(const HistoryRing &ring, const float *hm, const float *hi) override {
        if (stream) return;
        std::fill(h.begin(), h.end(), 0.0f);
        int n = std::min(historyLen_, ring.size());
        float xin[kHistoryDim];
        for (int t = ring.size() - n; t < ring.size(); ++t) {
            const float *v = ring.at(t);
            for (int i = 0; i < D; ++i) xin[i] = (v[i] - hm[i]) * hi[i];
            step(xin, h.data());
        }
    }
    void raw(const float *xn, const float *, float *out) override {
        std::copy(h.begin(), h.end(), headIn.begin());
        std::copy(xn, xn + kNumFeatures, headIn.begin() + H);
        const float *y = head.run(headIn.data());
        std::copy(y, y + head.outDim(), out);
    }
    void observe(const float *, const float *, const float *hn, float) override {
        if (stream) step(hn, h.data());
    }
};

struct TcnModel : Model {
    struct Conv {
        int cin, cout, k, dil, act, residual;
        std::vector<float> W, b; // W[o][i][j]
    };
    int D = 0;
    std::vector<Conv> convs;
    std::vector<float> seqA, seqB, headIn;
    std::vector<std::vector<int>> need; // per layer: output positions the last output depends on
    DenseStack head;
    void plan() {
        // Walk back from the last position of the last layer; each layer only needs
        // the positions its successor reads. Same result as the full computation.
        const int L = historyLen_;
        need.assign(convs.size(), {});
        std::vector<char> mark(size_t(L), 0);
        mark[size_t(L - 1)] = 1;
        for (size_t l = convs.size(); l-- > 0;) {
            for (int t = 0; t < L; ++t)
                if (mark[size_t(t)]) need[l].push_back(t);
            std::vector<char> prev(size_t(L), 0);
            for (int t : need[l]) {
                for (int j = 0; j < convs[l].k; ++j) {
                    int ts = t - (convs[l].k - 1 - j) * convs[l].dil;
                    if (ts >= 0) prev[size_t(ts)] = 1;
                }
                if (convs[l].residual) prev[size_t(t)] = 1;
            }
            mark = prev;
        }
    }
    void encode(const HistoryRing &ring, const float *hm, const float *hi) override {
        const int L = historyLen_;
        size_t maxC = size_t(D);
        for (auto &c : convs) maxC = std::max(maxC, size_t(c.cout));
        seqA.assign(size_t(L) * maxC, 0.0f);
        seqB.assign(size_t(L) * maxC, 0.0f);
        int n = std::min(L, ring.size());
        for (int t = 0; t < n; ++t) { // left zero padding: the newest vector sits at position L-1
            const float *v = ring.at(ring.size() - n + t);
            float *dst = &seqA[size_t(L - n + t) * maxC];
            for (int i = 0; i < D; ++i) dst[i] = (v[i] - hm[i]) * hi[i];
        }
        float *in = seqA.data(), *out = seqB.data();
        for (size_t l = 0; l < convs.size(); ++l) {
            const Conv &c = convs[l];
            for (int t : need[l]) {
                for (int o = 0; o < c.cout; ++o) {
                    float acc = c.b[size_t(o)];
                    for (int j = 0; j < c.k; ++j) {
                        int ts = t - (c.k - 1 - j) * c.dil;
                        if (ts < 0) continue;
                        const float *x = in + size_t(ts) * maxC;
                        const float *w = &c.W[(size_t(o) * c.cin) * c.k + j];
                        for (int i = 0; i < c.cin; ++i) acc += w[size_t(i) * c.k] * x[i];
                    }
                    acc = activate(c.act, acc);
                    if (c.residual) acc += in[size_t(t) * maxC + o];
                    out[size_t(t) * maxC + o] = acc;
                }
            }
            std::swap(in, out);
        }
        const int C = convs.back().cout;
        std::copy(in + size_t(L - 1) * maxC, in + size_t(L - 1) * maxC + C, headIn.begin());
    }
    void raw(const float *xn, const float *, float *out) override {
        const int C = convs.back().cout;
        std::copy(xn, xn + kNumFeatures, headIn.begin() + C);
        const float *y = head.run(headIn.data());
        std::copy(y, y + head.outDim(), out);
    }
};

// ---------------------------------------------------------------- parsing

struct Cursor {
    const unsigned char *p, *end;
    bool ok = true;
    template <class T> T get() {
        T v{};
        if (size_t(end - p) < sizeof(T)) {
            ok = false;
            return v;
        }
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    template <class T> void vec(std::vector<T> &out, size_t n) {
        if (n > size_t(end - p) / sizeof(T)) {
            ok = false;
            return;
        }
        out.resize(n);
        std::memcpy(out.data(), p, n * sizeof(T));
        p += n * sizeof(T);
    }
    bool done() const { return ok && p == end; }
};

struct Chunk {
    std::string tag;
    const unsigned char *data;
    size_t size;
};

bool parseDense(const Chunk &c, Dense &d, std::string &err) {
    Cursor cur{c.data, c.data + c.size};
    d.nIn = int(cur.get<uint32_t>());
    d.nOut = int(cur.get<uint32_t>());
    d.act = int(cur.get<uint32_t>());
    if (!cur.ok || d.nIn <= 0 || d.nOut <= 0 || d.nIn > 4096 || d.nOut > 4096 || d.act < 0 || d.act > 3) {
        err = "bad DENS header";
        return false;
    }
    cur.vec(d.W, size_t(d.nIn) * d.nOut);
    cur.vec(d.b, size_t(d.nOut));
    if (!cur.done()) {
        err = "DENS chunk size does not match its shape";
        return false;
    }
    return true;
}

bool parseOutputName(const std::string &n) {
    if (n == "t_mean_ms" || n == "p_miss") return true;
    int q = 0;
    char tail[8] = {};
    return std::sscanf(n.c_str(), "t_p%d%7s", &q, tail) == 2 && std::strcmp(tail, "_ms") == 0 && q >= 1 && q <= 99;
}

int transformCode(const std::string &t) {
    if (t == "identity") return 0;
    if (t == "sigmoid") return 1;
    if (t == "exp") return 2;
    if (t == "softplus") return 3;
    if (t == "mulexp") return 4;
    return -1;
}

int featureIndex(const std::string &n) {
    for (int i = 0; i < kNumFeatures; ++i)
        if (n == kFeatureNames[i]) return i;
    return -1;
}

} // namespace

struct SpkmParser {
    static std::unique_ptr<Model> parse(const std::vector<unsigned char> &bytes, std::string &err);
};

std::unique_ptr<Model> SpkmParser::parse(const std::vector<unsigned char> &bytes, std::string &err) {
    if (bytes.size() < 16) {
        err = "file too short (" + std::to_string(bytes.size()) + " bytes)";
        return nullptr;
    }
    if (std::memcmp(bytes.data(), "SPKM", 4) != 0) {
        err = "not a .spkm file (bad magic)";
        return nullptr;
    }
    uint32_t version, nChunks, crc;
    std::memcpy(&version, &bytes[4], 4);
    std::memcpy(&nChunks, &bytes[8], 4);
    std::memcpy(&crc, &bytes[12], 4);
    if (version != kSpkmVersion) {
        err = "unsupported .spkm format version " + std::to_string(version) + " (engine reads version " +
              std::to_string(kSpkmVersion) + ")";
        if (version == 1) err += "; v1 models use feature set spark-ft-v1, re-export them for spark-ft-v2";
        return nullptr;
    }
    if (crc32(bytes.data() + 16, bytes.size() - 16) != crc) {
        err = "CRC mismatch: file is truncated or corrupt";
        return nullptr;
    }
    std::vector<Chunk> chunks;
    {
        Cursor cur{bytes.data() + 16, bytes.data() + bytes.size()};
        for (uint32_t i = 0; i < nChunks; ++i) {
            if (size_t(cur.end - cur.p) < 8) {
                err = "truncated chunk table";
                return nullptr;
            }
            std::string tag(reinterpret_cast<const char *>(cur.p), 4);
            cur.p += 4;
            uint32_t size = cur.get<uint32_t>();
            size_t padded = (size_t(size) + 3) / 4 * 4;
            if (size_t(cur.end - cur.p) < padded) {
                err = "chunk " + tag + " runs past end of file";
                return nullptr;
            }
            chunks.push_back({tag, cur.p, size});
            cur.p += padded;
        }
        if (cur.p != cur.end) {
            err = "trailing bytes after the last chunk";
            return nullptr;
        }
    }
    auto find = [&](const char *tag) -> const Chunk * {
        for (auto &c : chunks)
            if (c.tag == tag) return &c;
        return nullptr;
    };

    // INFO: key=value lines.
    const Chunk *infoC = find("INFO");
    if (!infoC) {
        err = "missing INFO chunk";
        return nullptr;
    }
    std::map<std::string, std::string> info;
    std::vector<std::pair<std::string, std::string>> infoList;
    {
        std::istringstream in(std::string(reinterpret_cast<const char *>(infoC->data), infoC->size));
        std::string line;
        while (std::getline(in, line)) {
            auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            info[line.substr(0, eq)] = line.substr(eq + 1);
            infoList.emplace_back(line.substr(0, eq), line.substr(eq + 1));
        }
    }
    auto infoGet = [&](const char *k) { return info.count(k) ? info[k] : std::string(); };

    // Feature contract.
    if (infoGet("feature_set") != kFeatureSet) {
        err = "feature set '" + infoGet("feature_set") + "' does not match the engine's '" + kFeatureSet + "'";
        return nullptr;
    }
    char hashHex[32];
    std::snprintf(hashHex, sizeof hashHex, "%016llx", static_cast<unsigned long long>(featureHash()));
    if (infoGet("feature_hash") != hashHex) {
        err = "feature hash " + infoGet("feature_hash") + " does not match the engine's " + hashHex +
              " (model trained on a different feature list)";
        return nullptr;
    }
    if (infoGet("n_features") != std::to_string(kNumFeatures)) {
        err = "n_features " + infoGet("n_features") + " != engine's " + std::to_string(kNumFeatures);
        return nullptr;
    }
    const Chunk *featC = find("FEAT");
    if (!featC) {
        err = "missing FEAT chunk";
        return nullptr;
    }
    {
        std::istringstream in(std::string(reinterpret_cast<const char *>(featC->data), featC->size));
        std::string name;
        int i = 0;
        while (std::getline(in, name)) {
            if (name.empty()) continue;
            const char *want = i < kNumFeatures ? kFeatureNames[i]
                               : i < kHistoryDim ? kHistoryExtraNames[i - kNumFeatures] : "";
            if (name != want) {
                err = "feature " + std::to_string(i) + " is '" + name + "', engine expects '" + want + "'";
                return nullptr;
            }
            ++i;
        }
        if (i != kHistoryDim) {
            err = "FEAT lists " + std::to_string(i) + " names, expected " + std::to_string(kHistoryDim);
            return nullptr;
        }
    }

    std::string family = infoGet("family");
    int historyLen = std::atoi(infoGet("history_len").c_str());
    if (historyLen < 0 || historyLen > kMaxHistory) {
        err = "history_len " + std::to_string(historyLen) + " outside 0.." + std::to_string(kMaxHistory);
        return nullptr;
    }
    std::unique_ptr<Model> m;
    size_t params = 0;
    auto denseChunks = [&](DenseStack &st) -> bool {
        for (auto &c : chunks)
            if (c.tag == "DENS") {
                Dense d;
                if (!parseDense(c, d, err)) return false;
                if (!st.layers.empty() && st.layers.back().nOut != d.nIn) {
                    err = "DENS layer input " + std::to_string(d.nIn) + " != previous output " +
                          std::to_string(st.layers.back().nOut);
                    return false;
                }
                params += d.W.size() + d.b.size();
                st.layers.push_back(std::move(d));
            }
        if (st.layers.empty()) {
            err = "no DENS layers";
            return false;
        }
        st.prepare();
        return true;
    };

    if (family == "linear") {
        auto lm = std::make_unique<LinearModel>();
        const Chunk *c = find("LINR");
        if (!c) {
            err = "missing LINR chunk";
            return nullptr;
        }
        Cursor cur{c->data, c->data + c->size};
        lm->lin.nOut = int(cur.get<uint32_t>());
        lm->lin.nIn = int(cur.get<uint32_t>());
        if (!cur.ok || lm->lin.nIn != kNumFeatures || lm->lin.nOut < 1 || lm->lin.nOut > kMaxOutputs) {
            err = "bad LINR shape";
            return nullptr;
        }
        cur.vec(lm->lin.W, size_t(lm->lin.nIn) * lm->lin.nOut);
        cur.vec(lm->lin.b, size_t(lm->lin.nOut));
        if (!cur.done()) {
            err = "LINR chunk size does not match its shape";
            return nullptr;
        }
        params = lm->lin.W.size() + lm->lin.b.size();
        m = std::move(lm);
    } else if (family == "mlp") {
        auto mm = std::make_unique<MlpModel>();
        if (!denseChunks(mm->net)) return nullptr;
        if (mm->net.inDim() != kNumFeatures) {
            err = "MLP input " + std::to_string(mm->net.inDim()) + " != n_features";
            return nullptr;
        }
        m = std::move(mm);
    } else if (family == "gbt") {
        auto gm = std::make_unique<GbtModel>();
        const Chunk *c = find("GBTE");
        if (!c) {
            err = "missing GBTE chunk";
            return nullptr;
        }
        Cursor cur{c->data, c->data + c->size};
        uint32_t nOut = cur.get<uint32_t>();
        if (!cur.ok || nOut < 1 || nOut > kMaxOutputs) {
            err = "bad GBTE output count";
            return nullptr;
        }
        cur.vec(gm->base, nOut);
        uint32_t nTrees = cur.get<uint32_t>();
        if (!cur.ok || nTrees > 100000) {
            err = "bad GBTE tree count";
            return nullptr;
        }
        gm->trees.resize(nTrees);
        for (auto &t : gm->trees) {
            t.out = cur.get<uint32_t>();
            uint32_t nNodes = cur.get<uint32_t>();
            if (!cur.ok || t.out >= nOut || nNodes == 0 || nNodes > 1000000) {
                err = "bad GBTE tree header";
                return nullptr;
            }
            cur.vec(t.nodes, nNodes);
            if (!cur.ok) {
                err = "GBTE truncated";
                return nullptr;
            }
            for (uint32_t i = 0; i < nNodes; ++i) { // children must point forward: no cycles, no out-of-range
                const auto &nd = t.nodes[i];
                if (nd.feat >= kNumFeatures ||
                    (nd.feat >= 0 && (nd.left <= int32_t(i) || nd.right <= int32_t(i) || nd.left >= int32_t(nNodes) ||
                                      nd.right >= int32_t(nNodes)))) {
                    err = "GBTE node " + std::to_string(i) + " has an invalid feature or child index";
                    return nullptr;
                }
            }
            params += size_t(nNodes) * 2;
        }
        if (!cur.done()) {
            err = "GBTE chunk size mismatch";
            return nullptr;
        }
        params += nOut;
        m = std::move(gm);
    } else if (family == "rls") {
        auto rm = std::make_unique<RlsModel>();
        const Chunk *c = find("RLS_");
        if (!c) {
            err = "missing RLS_ chunk";
            return nullptr;
        }
        Cursor cur{c->data, c->data + c->size};
        uint32_t nIdx = cur.get<uint32_t>();
        if (!cur.ok || nIdx < 1 || nIdx > uint32_t(kNumFeatures)) {
            err = "bad RLS feature count";
            return nullptr;
        }
        cur.vec(rm->idx, nIdx);
        for (uint32_t i : rm->idx)
            if (i >= uint32_t(kNumFeatures)) cur.ok = false;
        rm->baseIdx = cur.get<int32_t>();
        rm->lambda = cur.get<double>();
        double delta = cur.get<double>();
        rm->maxTrace = cur.get<double>();
        rm->online = cur.get<uint32_t>() != 0;
        uint32_t hasP = cur.get<uint32_t>();
        size_t n = nIdx + 1;
        cur.vec(rm->w, n);
        if (hasP) cur.vec(rm->P, n * n);
        else if (cur.ok) {
            rm->P.assign(n * n, 0.0);
            for (size_t i = 0; i < n; ++i) rm->P[i * n + i] = delta;
        }
        uint32_t nq = cur.get<uint32_t>();
        if (nq > 8) cur.ok = false;
        cur.vec(rm->qLevel, nq);
        cur.vec(rm->qOff, nq);
        rm->gamma = cur.get<double>();
        rm->var = cur.get<double>();
        rm->varAlpha = cur.get<double>();
        if (!cur.done() || rm->baseIdx >= kNumFeatures || rm->lambda <= 0 || rm->lambda > 1) {
            err = "bad RLS_ chunk";
            return nullptr;
        }
        rm->phi.assign(n, 0);
        rm->Pphi.assign(n, 0);
        params = n + n * n + 2 * nq;
        m = std::move(rm);
    } else if (family == "gru") {
        auto gm = std::make_unique<GruModel>();
        const Chunk *c = find("GRU_");
        if (!c) {
            err = "missing GRU_ chunk";
            return nullptr;
        }
        Cursor cur{c->data, c->data + c->size};
        gm->D = int(cur.get<uint32_t>());
        gm->H = int(cur.get<uint32_t>());
        gm->stream = cur.get<uint32_t>() != 0;
        if (!cur.ok || gm->D != kHistoryDim || gm->H < 1 || gm->H > 1024) {
            err = "bad GRU_ shape";
            return nullptr;
        }
        size_t G = size_t(3 * gm->H);
        cur.vec(gm->Wih, G * gm->D);
        cur.vec(gm->Whh, G * gm->H);
        cur.vec(gm->bih, G);
        cur.vec(gm->bhh, G);
        if (!cur.done()) {
            err = "GRU_ chunk size mismatch";
            return nullptr;
        }
        params = gm->Wih.size() + gm->Whh.size() + 2 * G;
        if (!denseChunks(gm->head)) return nullptr;
        if (gm->head.inDim() != gm->H + kNumFeatures) {
            err = "GRU head input must be hidden + n_features";
            return nullptr;
        }
        if (!gm->stream && historyLen < 1) {
            err = "windowed GRU needs history_len >= 1";
            return nullptr;
        }
        gm->h.assign(size_t(gm->H), 0);
        gm->hTmp.assign(size_t(gm->H), 0);
        gm->gi.assign(G, 0);
        gm->gh.assign(G, 0);
        gm->headIn.assign(size_t(gm->H + kNumFeatures), 0);
        m = std::move(gm);
    } else if (family == "tcn") {
        auto tm = std::make_unique<TcnModel>();
        tm->D = kHistoryDim;
        int prev = kHistoryDim;
        for (auto &c : chunks) {
            if (c.tag != "TCNL") continue;
            Cursor cur{c.data, c.data + c.size};
            TcnModel::Conv cv{};
            cv.cin = int(cur.get<uint32_t>());
            cv.cout = int(cur.get<uint32_t>());
            cv.k = int(cur.get<uint32_t>());
            cv.dil = int(cur.get<uint32_t>());
            cv.act = int(cur.get<uint32_t>());
            cv.residual = int(cur.get<uint32_t>());
            if (!cur.ok || cv.cin != prev || cv.cout < 1 || cv.cout > 1024 || cv.k < 1 || cv.k > 16 || cv.dil < 1 ||
                cv.act < 0 || cv.act > 3 || (cv.residual && cv.cin != cv.cout)) {
                err = "bad TCNL layer";
                return nullptr;
            }
            cur.vec(cv.W, size_t(cv.cout) * cv.cin * cv.k);
            cur.vec(cv.b, size_t(cv.cout));
            if (!cur.done()) {
                err = "TCNL chunk size mismatch";
                return nullptr;
            }
            params += cv.W.size() + cv.b.size();
            prev = cv.cout;
            tm->convs.push_back(std::move(cv));
        }
        if (tm->convs.empty() || historyLen < 1) {
            err = "TCN needs TCNL layers and history_len >= 1";
            return nullptr;
        }
        if (!denseChunks(tm->head)) return nullptr;
        if (tm->head.inDim() != prev + kNumFeatures) {
            err = "TCN head input must be last channels + n_features";
            return nullptr;
        }
        tm->headIn.assign(size_t(prev + kNumFeatures), 0);
        tm->historyLen_ = historyLen;
        tm->plan();
        m = std::move(tm);
    } else {
        err = "unknown model family '" + family + "'";
        return nullptr;
    }

    m->family_ = family;
    m->historyLen_ = historyLen;
    m->info_ = infoList;
    m->params_ = params;
    if (!infoGet("budget_ms").empty()) m->budgetMs_ = std::atof(infoGet("budget_ms").c_str());

    // Normalisation (identity when absent).
    m->mean_.assign(kNumFeatures, 0.0f);
    m->invStd_.assign(kNumFeatures, 1.0f);
    m->hMean_.assign(kHistoryDim, 0.0f);
    m->hInvStd_.assign(kHistoryDim, 1.0f);
    auto readNorm = [&](const char *tag, int n, std::vector<float> &mean, std::vector<float> &inv) -> bool {
        const Chunk *c = find(tag);
        if (!c) return true;
        Cursor cur{c->data, c->data + c->size};
        std::vector<float> sd;
        cur.vec(mean, size_t(n));
        cur.vec(sd, size_t(n));
        if (!cur.done()) {
            err = std::string(tag) + " chunk size mismatch";
            return false;
        }
        for (int i = 0; i < n; ++i) {
            if (!(sd[size_t(i)] > 0) || !std::isfinite(sd[size_t(i)]) || !std::isfinite(mean[size_t(i)])) {
                err = std::string(tag) + ": std must be finite and > 0";
                return false;
            }
            inv[size_t(i)] = 1.0f / sd[size_t(i)];
        }
        return true;
    };
    if (!readNorm("NORM", kNumFeatures, m->mean_, m->invStd_)) return nullptr;
    if (!readNorm("HNRM", kHistoryDim, m->hMean_, m->hInvStd_)) return nullptr;

    // Outputs: "name transform base" lines.
    const Chunk *outC = find("OUTS");
    if (!outC) {
        err = "missing OUTS chunk";
        return nullptr;
    }
    {
        std::istringstream in(std::string(reinterpret_cast<const char *>(outC->data), outC->size));
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            std::istringstream ls(line);
            std::string name, tr, base, add = "-";
            ls >> name >> tr >> base;
            if (!(ls >> add)) add = "-";
            OutputSpec o;
            o.name = name;
            o.transform = transformCode(tr);
            o.baseFeature = base == "-" ? -1 : featureIndex(base);
            o.addFeature = add == "-" ? -1 : featureIndex(add);
            if (!parseOutputName(name) || o.transform < 0 || (base != "-" && o.baseFeature < 0) ||
                (add != "-" && o.addFeature < 0) || (o.transform == 4 && o.baseFeature < 0)) {
                err = "bad output line '" + line + "'";
                return nullptr;
            }
            m->outputs_.push_back(o);
        }
    }
    if (m->outputs_.empty() || m->outputs_.size() > size_t(kMaxOutputs)) {
        err = "model must declare 1.." + std::to_string(kMaxOutputs) + " outputs";
        return nullptr;
    }
    int nRaw = -1;
    if (auto *lm = dynamic_cast<LinearModel *>(m.get())) nRaw = lm->lin.nOut;
    else if (auto *mm = dynamic_cast<MlpModel *>(m.get())) nRaw = mm->net.outDim();
    else if (auto *gm = dynamic_cast<GbtModel *>(m.get())) nRaw = int(gm->base.size());
    else if (auto *ru = dynamic_cast<GruModel *>(m.get())) nRaw = ru->head.outDim();
    else if (auto *tm = dynamic_cast<TcnModel *>(m.get())) nRaw = tm->head.outDim();
    else if (auto *rm = dynamic_cast<RlsModel *>(m.get())) {
        nRaw = int(m->outputs_.size());
        for (auto &o : m->outputs_) {
            if (o.transform != 0 || o.baseFeature >= 0 || o.addFeature >= 0) {
                err = "rls outputs must use 'identity -' (the base feature is set in RLS_)";
                return nullptr;
            }
            if (o.name == "t_mean_ms") rm->outKind.push_back(-1);
            else if (o.name == "p_miss") rm->outKind.push_back(-2);
            else {
                int q = std::atoi(o.name.c_str() + 3);
                int k = -1;
                for (size_t i = 0; i < rm->qLevel.size(); ++i)
                    if (std::fabs(rm->qLevel[i] * 100 - q) < 1e-6) k = int(i);
                if (k < 0) {
                    err = "rls output " + o.name + " has no matching quantile level in RLS_";
                    return nullptr;
                }
                rm->outKind.push_back(k);
            }
        }
    }
    if (nRaw != int(m->outputs_.size())) {
        err = "model produces " + std::to_string(nRaw) + " values but OUTS lists " +
              std::to_string(m->outputs_.size());
        return nullptr;
    }
    if (m->historyLen_ > 0 && family != "gru" && family != "tcn") {
        err = "history_len > 0 is only meaningful for gru/tcn";
        return nullptr;
    }
    return m;
}

std::unique_ptr<Model> Model::loadBytes(const std::vector<unsigned char> &bytes, std::string &err) {
    return SpkmParser::parse(bytes, err);
}

std::unique_ptr<Model> Model::load(const std::string &path, std::string &err) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) {
        err = "cannot open " + path;
        return nullptr;
    }
    std::vector<unsigned char> bytes;
    unsigned char chunk[65536];
    size_t n;
    while ((n = std::fread(chunk, 1, sizeof chunk, f)) > 0) bytes.insert(bytes.end(), chunk, chunk + n);
    std::fclose(f);
    return loadBytes(bytes, err);
}

int Model::outputIndex(const std::string &name) const {
    for (size_t i = 0; i < outputs_.size(); ++i)
        if (outputs_[i].name == name) return int(i);
    return -1;
}

const std::string &Model::info(const std::string &key) const {
    static const std::string empty;
    for (auto &[k, v] : info_)
        if (k == key) return v;
    return empty;
}

void Model::beginFrame(const HistoryRing &h) { encode(h, hMean_.data(), hInvStd_.data()); }

void Model::predict(const float *x, float *out) {
    for (int i = 0; i < kNumFeatures; ++i) xn_[i] = (x[i] - mean_[size_t(i)]) * invStd_[size_t(i)];
    raw(xn_, x, raw_);
    for (size_t o = 0; o < outputs_.size(); ++o) {
        const OutputSpec &s = outputs_[o];
        float v;
        if (s.transform == 4) { // mulexp: reference time x exp(log-ratio), LIT_REVIEW_HW option A
            v = x[s.baseFeature] * std::exp(raw_[o]);
        } else {
            v = raw_[o] + (s.baseFeature >= 0 ? x[s.baseFeature] : 0.0f);
            switch (s.transform) {
            case 1: v = sigmoid(v); break;
            case 2: v = std::exp(v); break;
            case 3: v = v > 20 ? v : std::log1p(std::exp(v)); break;
            default: break;
            }
        }
        if (s.addFeature >= 0) v += x[s.addFeature]; // option B: additive tail term
        out[o] = v;
    }
}

void Model::endFrame(const float *x, const float *h, float yMs) {
    for (int i = 0; i < kNumFeatures; ++i) xn_[i] = (x[i] - mean_[size_t(i)]) * invStd_[size_t(i)];
    for (int i = 0; i < kHistoryDim; ++i) hn_[i] = (h[i] - hMean_[size_t(i)]) * hInvStd_[size_t(i)];
    observe(xn_, x, hn_, yMs);
}

} // namespace spark::ft
