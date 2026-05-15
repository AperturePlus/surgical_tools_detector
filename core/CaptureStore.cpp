#include "core/CaptureStore.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>

#include <opencv2/imgcodecs.hpp>

namespace sgt {

namespace {

std::string isoTimestamp()
{
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

std::string dayPart(const std::string& timestamp)
{
    return timestamp.size() >= 10 ? timestamp.substr(0, 10) : "unknown-date";
}

std::string compactTimestamp(const std::string& timestamp)
{
    std::string out;
    for (char c : timestamp) {
        if ((c >= '0' && c <= '9')) out += c;
    }
    return out.empty() ? "capture" : out;
}

std::string captureId(const std::string& timestamp)
{
    auto now = std::chrono::system_clock::now();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count() % 1000;
    std::ostringstream id;
    id << "capture_" << compactTimestamp(timestamp)
       << "_" << std::setw(3) << std::setfill('0') << millis;
    return id.str();
}

int countDefects(const std::vector<DefectResult>& defects)
{
    return static_cast<int>(std::count_if(defects.begin(), defects.end(),
        [](const DefectResult& d) { return d.defective; }));
}

std::string jsonEscapeLocal(const std::string& value)
{
    std::string out;
    for (char c : value) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: out += c; break;
        }
    }
    return out;
}

std::string readText(const std::filesystem::path& path)
{
    std::ifstream f(path);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string matchString(const std::string& text, const std::string& key)
{
    std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch match;
    if (std::regex_search(text, match, pattern) && match.size() > 1) {
        return match[1].str();
    }
    return {};
}

int matchInt(const std::string& text, const std::string& key)
{
    std::regex pattern("\"" + key + "\"\\s*:\\s*([0-9]+)");
    std::smatch match;
    if (std::regex_search(text, match, pattern) && match.size() > 1) {
        return std::stoi(match[1].str());
    }
    return 0;
}

float matchFloat(const std::string& text, const std::string& key, float fallback)
{
    std::regex pattern("\"" + key + "\"\\s*:\\s*([0-9]+(?:\\.[0-9]+)?)");
    std::smatch match;
    if (std::regex_search(text, match, pattern) && match.size() > 1) {
        return std::stof(match[1].str());
    }
    return fallback;
}

} // namespace

CaptureStore::CaptureStore(std::filesystem::path rootDir)
    : rootDir_(std::move(rootDir))
{
    loadExisting();
}

CaptureRecord CaptureStore::saveCapture(const DetectionFrameResult& result,
                                         const ModelInfo& models)
{
    namespace fs = std::filesystem;
    fs::create_directories(rootDir_);

    CaptureRecord record;
    record.timestamp = isoTimestamp();
    record.id = captureId(record.timestamp);
    record.modeMask = result.activeModes;
    record.thresholds = result.thresholds;
    record.toolCount = static_cast<int>(result.toolDetections.size());
    record.graspCount = static_cast<int>(result.graspDetections.size());
    record.defectCount = countDefects(result.defectResults);

    fs::path dayDir = rootDir_ / dayPart(record.timestamp);
    fs::create_directories(dayDir);
    int suffix = 1;
    while (fs::exists(dayDir / (record.id + ".json"))) {
        record.id = captureId(record.timestamp) + "_" + std::to_string(suffix++);
    }
    record.rawImagePath = (dayDir / (record.id + "_raw.jpg")).string();
    record.annotatedImagePath = (dayDir / (record.id + "_annotated.jpg")).string();
    record.jsonPath = (dayDir / (record.id + ".json")).string();

    ExportMetadata meta;
    meta.schemaVersion = 1;
    meta.id = record.id;
    meta.timestamp = record.timestamp;
    meta.modeMask = record.modeMask;
    meta.thresholds = record.thresholds;
    meta.models = models;
    meta.rawImagePath = fs::relative(record.rawImagePath, rootDir_).generic_string();
    meta.annotatedImagePath = fs::relative(record.annotatedImagePath, rootDir_).generic_string();
    meta.jsonPath = fs::relative(record.jsonPath, rootDir_).generic_string();

    ExportData exportData = exportFrame(result.rawFrame,
                                        result.annotatedFrame,
                                        result.toolDetections,
                                        result.graspDetections,
                                        result.defectResults,
                                        meta);

    if (!cv::imwrite(record.rawImagePath, exportData.rawFrame)) {
        throw std::runtime_error("failed to write raw capture image");
    }
    if (!cv::imwrite(record.annotatedImagePath, exportData.annotatedFrame)) {
        throw std::runtime_error("failed to write annotated capture image");
    }
    std::ofstream json(record.jsonPath);
    if (!json) {
        throw std::runtime_error("failed to write capture json");
    }
    json << exportData.statsJson;

    records_.push_back(record);
    std::sort(records_.begin(), records_.end(),
              [](const CaptureRecord& a, const CaptureRecord& b) {
                  return a.timestamp > b.timestamp;
              });
    writeIndex();
    return record;
}

std::vector<CaptureRecord> CaptureStore::records() const
{
    return records_;
}

void CaptureStore::loadExisting()
{
    namespace fs = std::filesystem;
    records_.clear();
    if (!fs::exists(rootDir_)) return;

    for (const auto& entry : fs::recursive_directory_iterator(rootDir_)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".json") continue;
        if (entry.path().filename() == "index.json") continue;

        std::string text = readText(entry.path());
        if (text.empty()) continue;

        CaptureRecord record;
        record.id = matchString(text, "id");
        record.timestamp = matchString(text, "timestamp");
        record.modeMask = static_cast<uint8_t>(matchInt(text, "modeMask"));
        record.toolCount = matchInt(text, "toolCount");
        record.graspCount = matchInt(text, "graspCount");
        record.defectCount = matchInt(text, "defectCount");
        const DetectionThresholds fallbackThresholds;
        record.thresholds.tool = matchFloat(text, "tool", fallbackThresholds.tool);
        record.thresholds.grasp = matchFloat(text, "grasp", fallbackThresholds.grasp);
        record.thresholds.defect = matchFloat(text, "defect", fallbackThresholds.defect);
        record.jsonPath = entry.path().string();

        fs::path dir = entry.path().parent_path();
        std::string base = entry.path().stem().string();
        record.rawImagePath = (dir / (base + "_raw.jpg")).string();
        record.annotatedImagePath = (dir / (base + "_annotated.jpg")).string();
        if (record.id.empty()) record.id = base;
        records_.push_back(record);
    }

    std::sort(records_.begin(), records_.end(),
              [](const CaptureRecord& a, const CaptureRecord& b) {
                  return a.timestamp > b.timestamp;
              });
    writeIndex();
}

void CaptureStore::writeIndex() const
{
    namespace fs = std::filesystem;
    fs::create_directories(rootDir_);
    std::ofstream out(rootDir_ / "index.json");
    if (!out) return;

    out << "{\n  \"schemaVersion\": 1,\n  \"records\": [\n";
    for (size_t i = 0; i < records_.size(); ++i) {
        const auto& r = records_[i];
        out << "    {"
            << "\"id\":\"" << jsonEscapeLocal(r.id) << "\","
            << "\"timestamp\":\"" << jsonEscapeLocal(r.timestamp) << "\","
            << "\"rawImagePath\":\"" << jsonEscapeLocal(fs::relative(r.rawImagePath, rootDir_).generic_string()) << "\","
            << "\"annotatedImagePath\":\"" << jsonEscapeLocal(fs::relative(r.annotatedImagePath, rootDir_).generic_string()) << "\","
            << "\"jsonPath\":\"" << jsonEscapeLocal(fs::relative(r.jsonPath, rootDir_).generic_string()) << "\","
            << "\"modeMask\":" << static_cast<int>(r.modeMask) << ","
            << "\"toolCount\":" << r.toolCount << ","
            << "\"graspCount\":" << r.graspCount << ","
            << "\"defectCount\":" << r.defectCount
            << "}";
        if (i + 1 < records_.size()) out << ",";
        out << "\n";
    }
    out << "  ]\n}\n";
}

} // namespace sgt
