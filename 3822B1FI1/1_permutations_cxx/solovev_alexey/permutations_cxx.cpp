#include "permutations_cxx.h"

#include <algorithm>
#include <unordered_map>

void Permutations(dictionary_t& dictionary) {
    
    std::unordered_map<std::string, std::vector<std::string>> groups;

    for (const auto& pair : dictionary) {
        const std::string& word = pair.first;

        std::string signature = word;
        std::sort(signature.begin(), signature.end());

        groups[signature].push_back(word);
    }

    
    for (auto& pair : dictionary) {
        const std::string& word = pair.first;

        std::string signature = word;
        std::sort(signature.begin(), signature.end());

        const auto& group = groups[signature];

        std::vector<std::string> permutations;

        
        for (const auto& candidate : group) {
            if (candidate != word) {
                permutations.push_back(candidate);
            }
        }

        
        std::sort(permutations.rbegin(), permutations.rend());

        pair.second = std::move(permutations);
    }
}