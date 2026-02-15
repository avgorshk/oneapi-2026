#include "permutations_cxx.h"
#include <algorithm>
#include <unordered_map>

void Permutations(dictionary_t &dictionary) {
  std::unordered_map<std::string, std::vector<std::string>> groups;

  for (const auto &pair : dictionary) {
    std::string sorted = pair.first;
    std::sort(sorted.begin(), sorted.end());
    groups[sorted].push_back(pair.first);
  }

  for (auto &pair : dictionary) {
    std::string sorted = pair.first;
    std::sort(sorted.begin(), sorted.end());

    auto &group = groups[sorted];

    const std::string &current_key = pair.first;
    std::copy_if(
        group.begin(), group.end(), std::back_inserter(pair.second),
        [&current_key](const std::string &s) { return s != current_key; });

    std::sort(pair.second.begin(), pair.second.end(),
              std::greater<std::string>());
  }
}
