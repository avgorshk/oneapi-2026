#include "permutations_cxx.h"

#include <algorithm>
#include <unordered_map>

void Permutations(dictionary_t& dictionary) {
    std::unordered_map<std::string, std::vector<std::string>> groups;

    for (const auto& [word, _] : dictionary) {
        std::string key = word;
        std::sort(key.begin(), key.end());
        groups[key].push_back(word);
    }

    for (auto& [word, permutations] : dictionary) {
        std::string key = word;
        std::sort(key.begin(), key.end());

        const auto& group = groups[key];

        std::vector<std::string> result;
        result.reserve(group.size());

        for (const auto& candidate : group) {
            if (candidate != word) {
                result.push_back(candidate);
            }
        }

        std::sort(result.begin(), result.end(), std::greater<std::string>());

        permutations = std::move(result);
    }
}
