#include "permutations_cxx.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace {

struct signature_t {
    std::array<std::uint16_t, 26> cnt;

    signature_t() : cnt() {}

    explicit signature_t(const std::string& s) : cnt() {
        for (char ch : s) {
            ++cnt[static_cast<std::size_t>(ch - 'a')];
        }
    }

    bool operator==(const signature_t& other) const {
        return cnt == other.cnt;
    }
};

struct signature_hash_t {
    std::size_t operator()(const signature_t& sig) const {
        std::size_t h = 0;
        for (std::size_t i = 0; i < sig.cnt.size(); ++i) {
            h = h * 131u + sig.cnt[i];
        }
        return h;
    }
};

}  // namespace

void Permutations(dictionary_t& dictionary) {
    using dict_it_t = dictionary_t::iterator;
    using groups_t = std::unordered_map<signature_t, std::vector<dict_it_t>, signature_hash_t>;

    groups_t groups;
    groups.reserve(dictionary.size());

    for (dict_it_t it = dictionary.begin(); it != dictionary.end(); ++it) {
        groups[signature_t(it->first)].push_back(it);
    }

    for (groups_t::iterator g = groups.begin(); g != groups.end(); ++g) {
        std::vector<dict_it_t>& group = g->second;
        const std::size_t n = group.size();

        if (n <= 1) {
            continue;
        }

        std::sort(group.begin(), group.end(),
                  [](const dict_it_t& a, const dict_it_t& b) {
                      return a->first > b->first;
                  });

        for (std::size_t i = 0; i < n; ++i) {
            std::vector<std::string>& out = group[i]->second;
            out.clear();
            out.reserve(n - 1);

            for (std::size_t j = 0; j < n; ++j) {
                if (i != j) {
                    out.push_back(group[j]->first);
                }
            }
        }
    }
}
