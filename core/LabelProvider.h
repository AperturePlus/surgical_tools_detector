#pragma once

#include <string>

namespace sgt {

/// Abstract interface for mapping class IDs to human-readable display labels.
/// Implementations may load from files, databases, or remote services.
class LabelProvider {
public:
    virtual ~LabelProvider() = default;

    /// Return the display label for the given class ID.
    /// Must never throw; return a reasonable fallback on unknown IDs.
    virtual std::string getLabel(int classId) const = 0;

    /// Total number of known classes.
    virtual int numClasses() const = 0;
};

} // namespace sgt
