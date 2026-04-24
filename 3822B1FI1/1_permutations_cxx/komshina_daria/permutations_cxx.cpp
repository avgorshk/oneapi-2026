#include "permutations_cxx.h"

#include <algorithm>
#include <unordered_map>

static std::string MakePermutationKey(const std::string& word) {
    std::string key = word;
    std::sort(key.begin(), key.end());
    return key;
}

void Permutations(dictionary_t& dictionary) {
    std::unordered_map<std::string, std::vector<std::string>> groups;
    groups.reserve(dictionary.size() * 2u + 1u);

    for (const auto& item : dictionary) {
        const std::string& word = item.first;
        groups[MakePermutationKey(word)].push_back(word);
    }

    for (auto& item : dictionary) {
        item.second.clear();
    }

    for (const auto& group : groups) {
        const std::vector<std::string>& words = group.second;

        if (words.size() < 2u) {
            continue;
        }

        for (const std::string& current : words) {
            std::vector<std::string>& result = dictionary[current];
            result.reserve(words.size() - 1u);

            for (auto it = words.rbegin(); it != words.rend(); ++it) {
                if (*it != current) {
                    result.push_back(*it);
                }
            }
        }
    }
}