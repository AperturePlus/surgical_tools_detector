#include "label/DictLabelProvider.h"

#include <fstream>
#include <iostream>

namespace sgt {

DictLabelProvider::DictLabelProvider(const std::string&              dictPath,
                                     const std::vector<std::string>& classNames)
    : classNames_(classNames)
{
    if (!dictPath.empty()) {
        loadDict(dictPath);
    }
}

void DictLabelProvider::loadDict(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[LabelProvider] Cannot open: " << path
                  << " — using raw class names.\n";
        return;
    }

    auto trim = [](std::string& s) {
        auto b = s.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) { s.clear(); return; }
        s = s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
    };

    std::string line;
    int loaded = 0;
    while (std::getline(f, line)) {
        trim(line);
        if (line.empty() || line[0] == '#') continue;

        auto sep = line.find('=');
        if (sep == std::string::npos) continue;

        std::string key = line.substr(0, sep);
        std::string val = line.substr(sep + 1);
        trim(key);
        trim(val);

        if (!key.empty() && !val.empty()) {
            dict_[key] = val;
            ++loaded;
        }
    }
    std::cout << "[LabelProvider] Loaded " << loaded
              << " label(s) from " << path << "\n";
}

std::string DictLabelProvider::getLabel(int classId) const
{
    if (classId < 0 || classId >= static_cast<int>(classNames_.size())) {
        return "cls?" + std::to_string(classId);
    }
    const std::string& raw = classNames_[classId];
    auto it = dict_.find(raw);
    if (it != dict_.end()) return it->second;
    return raw; // graceful fallback to pinyin
}

} // namespace sgt
