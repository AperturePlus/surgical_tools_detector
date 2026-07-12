#include "core/DetectionPipeline.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>

#include <opencv2/imgproc.hpp>

#include "classifier/OnnxDefectClassifier.h"
#include "core/DetectorBackend.h"
#include "core/LabelProvider.h"
#include "core/Renderer.h"
#include "detector/YoloOnnxDetector.h"
#include "label/DictLabelProvider.h"

namespace xcwj {

namespace {

using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point start, Clock::time_point end = Clock::now())
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

const std::vector<std::string> TOOL_CLASSES = {
    "cefangkaikouqi","guqian","gujian","gudao","xianjian","yating",
    "bayaqian","guchui","shaozi","7haodaobing","3haodaobing","dagumo",
    "gucuo","zhenchi","zhizhixueqian","wanzhixueqian","paqian","kekeqian",
    "xiaoduqian","xichiqian","zuzhinie","dangou","pingnie","zuzhijian",
    "zhijiaoqian","jiazhuangxianlagou","huanqian","xiaogumo","sichilagou",
    "guachi","gangsijian","xiaolagou","eliekaikouqi","yasheban","eliejian",
    "xueguanshenjingboliqi1","xueguanshenjingboliqi2","yingeboliqi",
};

const std::vector<std::string> GRASP_CLASSES = {"work", "grasp"};

int countDefects(const std::vector<DefectResult>& defects) {
    return static_cast<int>(std::count_if(defects.begin(), defects.end(),
        [](const DefectResult& result) { return result.defective; }));
}

} // namespace

int DetectionFrameResult::defectCount() const
{
    return countDefects(defectResults);
}

cv::Scalar FrameAnnotator::classColor(int classId)
{
    float hue = std::fmod(static_cast<float>(classId) * 137.508f, 360.0f);
    cv::Mat hsv(1, 1, CV_8UC3,
                cv::Scalar(static_cast<uchar>(hue / 2.0f), 200, 220));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    auto* p = bgr.ptr<uchar>(0);
    return cv::Scalar(p[0], p[1], p[2]);
}

void FrameAnnotator::drawLabelBadge(cv::Mat& frame,
                                    const std::string& text,
                                    cv::Point anchor,
                                    cv::Scalar bgColor)
{
    constexpr double fontScale = 0.58;
    constexpr int thickness = 1;
    int baseline = 0;
    cv::Size sz = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontScale,
                                  thickness, &baseline);

    int padX = 6;
    int padY = 4;
    int bx = anchor.x;
    int by = anchor.y - sz.height - baseline - padY * 2;
    if (by < 0) by = anchor.y + padY;
    if (bx + sz.width + padX * 2 > frame.cols) {
        bx = std::max(0, frame.cols - sz.width - padX * 2);
    }

    cv::Rect bg(bx, by, sz.width + padX * 2, sz.height + baseline + padY * 2);
    bg &= cv::Rect(0, 0, frame.cols, frame.rows);
    cv::rectangle(frame, bg, bgColor, cv::FILLED);
    cv::putText(frame, text, {bx + padX, by + sz.height + padY},
                cv::FONT_HERSHEY_SIMPLEX, fontScale, {255,255,255},
                thickness, cv::LINE_AA);
}

cv::Mat FrameAnnotator::annotate(const cv::Mat& frame,
                                 const std::vector<Detection>& toolDets,
                                 const std::vector<Detection>& graspDets,
                                 const std::vector<DefectResult>& defects) const
{
    cv::Mat out = frame.clone();
    auto drawDets = [&](const std::vector<Detection>& dets) {
        for (const auto& d : dets) {
            cv::Scalar color = classColor(d.classId);
            cv::Rect rect = d.bbox.toRect() & cv::Rect(0, 0, out.cols, out.rows);
            if (rect.empty()) continue;
            cv::rectangle(out, rect, color, 2);
            std::ostringstream label;
            label << (d.label.empty() ? "cls" + std::to_string(d.classId) : d.label)
                  << " " << std::fixed << std::setprecision(0) << d.score * 100.0f << "%";
            drawLabelBadge(out, label.str(), {rect.x, rect.y}, color);
        }
    };

    drawDets(toolDets);
    drawDets(graspDets);

    const cv::Scalar defectColor(20, 20, 230);
    for (const auto& d : defects) {
        if (!d.defective) continue;
        cv::Rect rect = d.bbox.toRect() & cv::Rect(0, 0, out.cols, out.rows);
        if (rect.empty()) continue;
        cv::rectangle(out, rect, defectColor, 4);
        std::ostringstream label;
        label << "Defect " << std::fixed << std::setprecision(0) << d.defectScore * 100.0f << "%";
        drawLabelBadge(out, label.str(), {rect.x, rect.y}, defectColor);
    }

    return out;
}

DetectionEngine::DetectionEngine(const std::string& toolModelPath,
                                 const std::string& graspModelPath,
                                 const std::string& defectModelPath,
                                 const std::string& dictPath)
    : toolPath_(toolModelPath)
    , graspPath_(graspModelPath)
    , defectPath_(defectModelPath)
    , dictPath_(dictPath)
{
}

DetectionEngine::~DetectionEngine() = default;

bool DetectionEngine::ensureLoaded(uint8_t bit, std::string* error)
{
    try {
        if (bit == MODE_TOOL && !toolDet_) {
            if (!toolLabels_) {
                toolLabels_ = std::make_unique<DictLabelProvider>(dictPath_, TOOL_CLASSES);
            }
            toolDet_ = std::make_unique<YoloOnnxDetector>(toolPath_, 640,
                                                          pendingThresholds_.tool, 0.45f,
                                                          toolLabels_.get());
        }
        if (bit == MODE_GRASP && !graspDet_) {
            if (!graspLabels_) {
                graspLabels_ = std::make_unique<DictLabelProvider>("", GRASP_CLASSES);
            }
            graspDet_ = std::make_unique<YoloOnnxDetector>(graspPath_, 640,
                                                           pendingThresholds_.grasp, 0.45f,
                                                           graspLabels_.get());
        }
        if (bit == MODE_DEFECT && !defectCls_) {
            defectCls_ = std::make_unique<OnnxDefectClassifier>(defectPath_, 512, 4,
                                                                pendingThresholds_.defect);
        }
        setThresholds(pendingThresholds_);
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}

DetectionFrameResult DetectionEngine::process(const cv::Mat& frame, uint8_t modeMask)
{
    DetectionFrameResult result;
    if (frame.empty()) return result;

    result.rawFrame = frame.clone();
    result.activeModes = modeMask;

    if ((modeMask & MODE_TOOL) && ensureLoaded(MODE_TOOL)) {
        result.toolDetections = toolDet_->detect(frame);
        result.perf += toolDet_->lastPerfStats();
    }
    if ((modeMask & MODE_GRASP) && ensureLoaded(MODE_GRASP)) {
        result.graspDetections = graspDet_->detect(frame);
        result.perf += graspDet_->lastPerfStats();
    }
    if ((modeMask & MODE_DEFECT) && ensureLoaded(MODE_DEFECT)) {
        Detection full;
        full.bbox = {0, 0, float(frame.cols), float(frame.rows)};
        result.defectResults = defectCls_->classify(frame, {full});
    }

    result.thresholds = thresholds();
    auto annotateStart = Clock::now();
    result.annotatedFrame = annotator_.annotate(frame,
                                                result.toolDetections,
                                                result.graspDetections,
                                                result.defectResults);
    result.perf.annotateMs = elapsedMs(annotateStart);
    return result;
}

void DetectionEngine::setThresholds(const DetectionThresholds& thresholds)
{
    pendingThresholds_.tool = std::clamp(thresholds.tool, 0.05f, 0.95f);
    pendingThresholds_.grasp = std::clamp(thresholds.grasp, 0.05f, 0.95f);
    pendingThresholds_.defect = std::clamp(thresholds.defect, 0.05f, 0.95f);
    if (toolDet_) toolDet_->setConfThresh(pendingThresholds_.tool);
    if (graspDet_) graspDet_->setConfThresh(pendingThresholds_.grasp);
    if (defectCls_) defectCls_->setDefectThresh(pendingThresholds_.defect);
}

DetectionThresholds DetectionEngine::thresholds() const
{
    DetectionThresholds t = pendingThresholds_;
    if (toolDet_) t.tool = toolDet_->getConfThresh();
    if (graspDet_) t.grasp = graspDet_->getConfThresh();
    if (defectCls_) t.defect = defectCls_->getDefectThresh();
    return t;
}

ModelInfo DetectionEngine::models() const
{
    return {toolPath_, graspPath_, defectPath_};
}

} // namespace xcwj
