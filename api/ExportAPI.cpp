#include "api/ExportAPI.h"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <opencv2/imgcodecs.hpp>

namespace sgt {

static std::string timestamp() {
    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream o;
    o << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return o.str();
}

static std::string escapeJson(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else out += c;
    }
    return out;
}

static void appendDetections(std::ostringstream& o, const char* key,
                             const std::vector<Detection>& dets) {
    o << "\"" << key << "\":[";
    for (size_t i = 0; i < dets.size(); ++i) {
        auto& d = dets[i];
        if (i) o << ",";
        o << "{\"label\":\"" << escapeJson(d.label)
          << "\",\"classId\":" << d.classId
          << ",\"score\":" << std::fixed << std::setprecision(3) << d.score
          << ",\"bbox\":[" << d.bbox.x << "," << d.bbox.y
          << "," << d.bbox.w << "," << d.bbox.h << "]}";
    }
    o << "]";
}

ExportData exportFrame(const cv::Mat& frame,
                       const std::vector<Detection>& toolDets,
                       const std::vector<Detection>& graspDets,
                       const std::vector<DefectResult>& defects) {
    ExportData data;
    data.frame = frame.clone();

    std::ostringstream o;
    o << std::fixed << std::setprecision(3);
    o << "{\"timestamp\":\"" << timestamp() << "\",";
    appendDetections(o, "toolDetections", toolDets);
    o << ",";
    appendDetections(o, "graspDetections", graspDets);
    o << ",\"defects\":[";
    for (size_t i = 0; i < defects.size(); ++i) {
        auto& d = defects[i];
        if (i) o << ",";
        o << "{\"defective\":" << (d.defective ? "true" : "false")
          << ",\"defectScore\":" << d.defectScore
          << ",\"bbox\":[" << d.bbox.x << "," << d.bbox.y
          << "," << d.bbox.w << "," << d.bbox.h << "]}";
    }
    o << "]}";
    data.statsJson = o.str();
    return data;
}

bool saveExport(const ExportData& data, const std::string& outputDir) {
    namespace fs = std::filesystem;
    try {
        fs::create_directories(outputDir);
        auto ts = timestamp();
        // Replace colons for filename safety
        for (char& c : ts) if (c == ':') c = '-';

        std::string imgPath  = (fs::path(outputDir) / ("export_" + ts + ".jpg")).string();
        std::string jsonPath = (fs::path(outputDir) / ("export_" + ts + ".json")).string();

        if (!cv::imwrite(imgPath, data.frame)) return false;

        std::ofstream f(jsonPath);
        if (!f) return false;
        f << data.statsJson;
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace sgt
