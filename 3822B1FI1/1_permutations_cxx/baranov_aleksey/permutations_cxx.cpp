#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>

using dictionary_t = std::unordered_map<std::string, std::vector<std::string>>;

namespace {

struct CharCounts {
    std::array<std::uint16_t, 26> counts;

    CharCounts() : counts() {}

    explicit CharCounts(const std::string& word) : counts() {
        for (char letter : word) {
            ++counts[static_cast<std::size_t>(letter - 'a')];
        }
    }

    bool operator==(const CharCounts& other) const {
        return counts == other.counts;
    }
};

struct CharCountsHasher {
    std::size_t operator()(const CharCounts& obj) const {
        std::size_t hashValue = 0;
        for (std::size_t i = 0; i < obj.counts.size(); ++i) {
            hashValue = hashValue * 131u + obj.counts[i];
        }
        return hashValue;
    }
};

}  // namespace

void Permutations(dictionary_t& dictionary) {
    using EntryIt = dictionary_t::iterator;
    using AnagramGroups = std::unordered_map<CharCounts, std::vector<EntryIt>, CharCountsHasher>;

    AnagramGroups anagramGroups;
    anagramGroups.reserve(dictionary.size());

    for (EntryIt entryIt = dictionary.begin(); entryIt != dictionary.end(); ++entryIt) {
        anagramGroups[CharCounts(entryIt->first)].push_back(entryIt);
    }

    for (auto& groupPair : anagramGroups) {
        std::vector<EntryIt>& groupEntries = groupPair.second;
        const std::size_t groupSize = groupEntries.size();

        if (groupSize <= 1) {
            continue;
        }

        // Сортировка по убыванию исходного слова (ключа)
        std::stable_sort(groupEntries.begin(), groupEntries.end(),
                         [](const EntryIt& a, const EntryIt& b) {
                             return a->first > b->first;
                         });

        for (std::size_t idx = 0; idx < groupSize; ++idx) {
            std::vector<std::string>& permutationsList = groupEntries[idx]->second;
            permutationsList.clear();
            permutationsList.reserve(groupSize - 1);

            for (std::size_t otherIdx = 0; otherIdx < groupSize; ++otherIdx) {
                if (idx != otherIdx) {
                    permutationsList.push_back(groupEntries[otherIdx]->first);
                }
            }
        }
    }
}