#include "permutations_cxx.h"
#include <algorithm>
#include <functional>
#include <iterator>
#include <unordered_map>

void Permutations(dictionary_t& dictionary) {
    std::unordered_map<std::string, std::vector<std::reference_wrapper<const std::string>>> groups;
    groups.reserve(dictionary.size());

    for (const auto& entry : dictionary) {
        const std::string& word = entry.first;
        std::string sorted = word;
        std::sort(sorted.begin(), sorted.end());
        
        groups[sorted].push_back(std::cref(word));
    }

    for (auto& entry : dictionary) {
        const std::string& word = entry.first;
        std::vector<std::string>& permutations = entry.second;

        std::string sorted = word;
        std::sort(sorted.begin(), sorted.end());

        const auto& group = groups[sorted];
        
        permutations.clear();
        permutations.reserve(group.size() - 1);

        std::transform(group.begin(), group.end(),
                      std::back_inserter(permutations),
                      [](std::reference_wrapper<const std::string> ref) {
                          return ref.get();
                      });
        
        permutations.erase(
            std::remove(permutations.begin(), permutations.end(), word),
            permutations.end()
        );

        std::sort(permutations.begin(), permutations.end(),
                  [](const std::string& a, const std::string& b) {
                      return a > b; 
                  });
    }
}