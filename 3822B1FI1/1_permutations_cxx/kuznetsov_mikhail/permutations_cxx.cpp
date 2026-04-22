#include "permutations_cxx.h"

std::string MakeSignature(const std::string& word)
{
    int freq[26] = {0};

    for (char c : word) {
        freq[c - 'a']++;
    }

    std::string signature = std::to_string(word.size()) + "#";

    for (int i = 0; i < 26; ++i) {
        if (freq[i] > 0) {
            signature += char('a' + i);
            signature += std::to_string(freq[i]);
        }
    }

    return signature;
}

void Permutations(dictionary_t& dictionary)
{
    std::map<std::string, std::vector<std::string>> groups;

    for (auto it = dictionary.begin(); it != dictionary.end(); ++it) {
        std::string sig = MakeSignature(it->first);
        groups[sig].push_back(it->first);
    }

    for (auto& [sig, words] : groups) {

        std::sort(words.begin(), words.end(),
                  [](const std::string& a, const std::string& b) {
                      return a > b;
                  });

        for (const auto& word : words) {
            auto& target = dictionary[word];

            for (const auto& other : words) {
                if (other != word) {
                    target.push_back(other);
                }
            }
        }
    }
}
