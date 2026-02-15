#include "permutations_cxx.h"
#include <algorithm>
#include <set>

void Permutations(dictionary_t &dictionary) {
  std::set<std::string> keys;
  for (const auto &pair : dictionary) {
    keys.insert(pair.first);
  }

  for (auto &pair : dictionary) {
    std::string key = pair.first;
    std::vector<std::string> &permutations = pair.second;

    std::sort(key.begin(), key.end(), std::greater<char>());
    do {
      if (pair.first != key && keys.count(key))
        permutations.push_back(key);
    } while (std::prev_permutation(key.begin(), key.end()));
  }
}
