#include "permutations_cxx.h"
#include <algorithm> 
#include <functional> 

void Permutations(dictionary_t& dictionary) {
    std::map<std::string, std::vector<std::string>> groups;

    for (auto const& item : dictionary) {
        std::string word = item.first;
        std::string sorted_word = word;
        std::sort(sorted_word.begin(), sorted_word.end());
        groups[sorted_word].push_back(word);
    }

    for (auto& item : dictionary) {
        const std::string& word = item.first;
        std::string sorted_word = word;
        std::sort(sorted_word.begin(), sorted_word.end());

        std::vector<std::string> permutations = groups[sorted_word];

        auto it = std::find(permutations.begin(), permutations.end(), word);
        if (it != permutations.end()) {
            permutations.erase(it);
        }

        std::sort(permutations.begin(), permutations.end(), std::greater<std::string>());

        item.second = permutations;
    }
}