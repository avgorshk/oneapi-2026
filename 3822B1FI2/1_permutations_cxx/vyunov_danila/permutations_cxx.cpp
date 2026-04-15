#include "permutations_cxx.h"

#include <algorithm>
#include <unordered_map>

void Permutations(dictionary_t& dictionary) {
  std::unordered_map<std::string, std::vector<std::string>> groups;
  
  for (const auto& [key, _] : dictionary) {
    std::string sig = key;
    std::sort(sig.begin(), sig.end());
    groups[sig].push_back(key);
  }

  for (auto& [key, perms] : dictionary) {
    std::string sig = key;
    std::sort(sig.begin(), sig.end());

    perms.clear();
    for (const auto& candidate : groups[sig]) {
      if (candidate != key) {
        perms.push_back(candidate);
      }
    }

    std::sort(perms.begin(), perms.end(), std::greater<std::string>());
  }
}
