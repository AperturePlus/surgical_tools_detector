#include "api/ExportAPI.h"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include <opencv2/imgcodecs.hpp>

namespace xcwj {

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

std::string escapeJson(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: out += c; break;
        }
    }
    return out;
}

static std::string basenameOf(const std::string& path) {
    namespace fs = std::filesystem;
    if (path.empty()) return {};
    return fs::path(path).filename().string();
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

static void appendThresholds(std::ostringstream& o, const DetectionThresholds& t) {
    o << "\"thresholds\":{"
      << "\"tool\":" << t.tool << ","
      << "\"grasp\":" << t.grasp << ","
      << "\"defect\":" << t.defect << "}";
}

static void appendModels(std::ostringstream& o, const ModelInfo& models) {
    o << "\"models\":{"
      << "\"tool\":\"" << escapeJson(basenameOf(models.toolModel)) << "\","
      << "\"grasp\":\"" << escapeJson(basenameOf(models.graspModel)) << "\","
      << "\"defect\":\"" << escapeJson(basenameOf(models.defectModel)) << "\""
      << "}";
}

static void appendFiles(std::ostringstream& o, const ExportMetadata& meta) {
    o << "\"files\":{"
      << "\"rawImagePath\":\"" << escapeJson(meta.rawImagePath) << "\","
      << "\"annotatedImagePath\":\"" << escapeJson(meta.annotatedImagePath) << "\","
      << "\"jsonPath\":\"" << escapeJson(meta.jsonPath) << "\""
      << "}";
}

static int countDefects(const std::vector<DefectResult>& defects) {
    int count = 0;
    for (const auto& defect : defects) {
        if (defect.defective) ++count;
    }
    return count;
}

static std::string statsJson(const std::vector<Detection>& toolDets,
                             const std::vector<Detection>& graspDets,
                             const std::vector<DefectResult>& defects,
                             const ExportMetadata& meta) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(3);
    o << "{";
    o << "\"schemaVersion\":" << meta.schemaVersion << ",";
    if (!meta.id.empty()) {
        o << "\"id\":\"" << escapeJson(meta.id) << "\",";
    }
    o << "\"timestamp\":\"" << escapeJson(meta.timestamp.empty() ? timestamp() : meta.timestamp) << "\",";
    o << "\"modeMask\":" << static_cast<int>(meta.modeMask) << ",";
    appendThresholds(o, meta.thresholds);
    o << ",";
    appendModels(o, meta.models);
    o << ",";
    appendFiles(o, meta);
    o << ",\"summary\":{"
      << "\"toolCount\":" << toolDets.size() << ","
      << "\"graspCount\":" << graspDets.size() << ","
      << "\"defectCount\":" << countDefects(defects)
      << "},";
    appendDetections(o, "toolDetections", toolDets);
    o << ",";
    appendDetections(o, "graspDetections", graspDets);
    o << ",\"defects\":[";
    for (size_t i = 0; i < defects.size(); ++i) {
        auto& d = defects[i];
        if (i) o << ",";
        o << "{\"toolIndex\":" << d.toolIndex
          << ",\"defective\":" << (d.defective ? "true" : "false")
          << ",\"normalScore\":" << d.normalScore
          << ",\"defectScore\":" << d.defectScore
          << ",\"bbox\":[" << d.bbox.x << "," << d.bbox.y
          << "," << d.bbox.w << "," << d.bbox.h << "]}";
    }
    o << "]}";
    return o.str();
}

ExportData exportFrame(const cv::Mat& frame,
                       const std::vector<Detection>& toolDets,
                       const std::vector<Detection>& graspDets,
                       const std::vector<DefectResult>& defects) {
    ExportMetadata meta;
    meta.timestamp = timestamp();
    return exportFrame(frame, frame, toolDets, graspDets, defects, meta);
}

ExportData exportFrame(const cv::Mat& rawFrame,
                       const cv::Mat& annotatedFrame,
                       const std::vector<Detection>& toolDets,
                       const std::vector<Detection>& graspDets,
                       const std::vector<DefectResult>& defects,
                       const ExportMetadata& meta) {
    ExportData data;
    data.rawFrame = rawFrame.clone();
    data.annotatedFrame = annotatedFrame.empty() ? rawFrame.clone() : annotatedFrame.clone();
    data.statsJson = statsJson(toolDets, graspDets, defects, meta);
    return data;
}

bool saveExport(const ExportData& data, const std::string& outputDir) {
    namespace fs = std::filesystem;
    try {
        fs::create_directories(outputDir);
        auto ts = timestamp();
        // Replace colons for filename safety
        for (char& c : ts) if (c == ':') c = '-';

        std::string rawPath  = (fs::path(outputDir) / ("export_" + ts + "_raw.jpg")).string();
        std::string imgPath  = (fs::path(outputDir) / ("export_" + ts + "_annotated.jpg")).string();
        std::string jsonPath = (fs::path(outputDir) / ("export_" + ts + ".json")).string();

        if (!data.rawFrame.empty() && !cv::imwrite(rawPath, data.rawFrame)) return false;
        if (!cv::imwrite(imgPath, data.annotatedFrame)) return false;

        std::ofstream f(jsonPath);
        if (!f) return false;
        f << data.statsJson;
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace xcwj
