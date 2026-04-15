#include "permutations_cxx.h"

void Permutations(dictionary_t& dictionary) {
    std::map<std::string, std::vector<std::string>> groups;

    for (const auto& item : dictionary) {
        const std::string& word = item.first;
        std::string sorted_word = word;
        std::sort(sorted_word.begin(), sorted_word.end());
        groups[sorted_word].push_back(word);
    }

    for (auto& item : dictionary) {
        const std::string& word = item.first;
        std::string sorted_word = word;
        std::sort(sorted_word.begin(), sorted_word.end());

        item.second.clear();

        const std::vector<std::string>& anagrams = groups[sorted_word];
        for (const auto& candidate : anagrams) {
            if (candidate != word) {
                item.second.push_back(candidate);
            }
        }

        std::sort(item.second.rbegin(), item.second.rend());
    }
}