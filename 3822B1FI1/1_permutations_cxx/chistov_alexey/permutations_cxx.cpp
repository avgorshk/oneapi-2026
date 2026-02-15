#include "permutations_cxx.h"

#include <unordered_map>
#include <algorithm>

void Permutations(dictionary_t& dictionary) {
    using iterator = dictionary_t::iterator;

    std::unordered_map<std::string, std::vector<iterator>> groups;
    groups.reserve(dictionary.size());

    for (auto it = dictionary.begin(); it != dictionary.end(); ++it) {
        std::string key = it->first;
        std::sort(key.begin(), key.end());
        groups[key].push_back(it);
    }

    for (auto& pair : groups) {
        auto& group = pair.second;

        if (group.size() <= 1)
            continue;

        std::sort(group.begin(), group.end(),
                  [](const iterator& a, const iterator& b) {
                      return a->first > b->first;
                  });

        for (size_t i = 0; i < group.size(); ++i) {
            auto& perms = group[i]->second;
            perms.reserve(group.size() - 1);

            for (size_t j = 0; j < group.size(); ++j) {
                if (i != j)
                    perms.push_back(group[j]->first);
            }
        }
    }
}
