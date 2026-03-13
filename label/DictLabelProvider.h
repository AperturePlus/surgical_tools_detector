#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include "core/LabelProvider.h"

namespace sgt {

/// Loads a plain-text dictionary file (pinyin=display_label, # comments)
/// and maps class IDs → display labels via the class-name index.
/// Gracefully falls back to the raw pinyin name when:
///   - the dict file is missing or unreadable
///   - a key has no value (empty right-hand side)
///
/// To add a new label source (e.g. database):
///   derive a new class from LabelProvider and swap in main.cpp.
class DictLabelProvider : public LabelProvider {
public:
    /// @param dictPath    Path to labels.dict (empty string = skip loading).
    /// @param classNames  Ordered list of raw class names from the YAML.
    DictLabelProvider(const std::string&              dictPath,
                      const std::vector<std::string>& classNames);

    std::string getLabel(int classId) const override;
    int numClasses() const override {
        return static_cast<int>(classNames_.size());
    }

private:
    std::vector<std::string>              classNames_;  ///< index → raw pinyin
    std::unordered_map<std::string,
                       std::string>       dict_;        ///< pinyin → display label

    void loadDict(const std::string& path);
};

} // namespace sgt
