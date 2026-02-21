#include "permutations_cxx.h"
#include <unordered_map>
#include <algorithm>
#include <string>

void Permutations(dictionary_t& dictionary) {
    std::unordered_map<std::string, std::vector<dictionary_t::iterator>> signature_groups;
    signature_groups.reserve(dictionary.size());

    for (auto it = dictionary.begin(); it != dictionary.end(); it++) {
        std::string signature = it->first;
        std::sort(signature.begin(), signature.end());
        signature_groups[signature].emplace_back(it);
    }

    for (auto& [signature, group] : signature_groups) {
        if (group.size() < 2) continue;

        std::vector<std::string> sorted_keys;
        sorted_keys.reserve(group.size());
        for (auto& it : group) {
            sorted_keys.push_back(it->first);
        }
        std::sort(sorted_keys.begin(), sorted_keys.end(), std::greater<>());

        for (auto& it : group) {
            auto& perms = it->second;
            perms = sorted_keys;
            perms.erase(std::remove(perms.begin(), perms.end(), it->first), perms.end());
        }
    }
}
