#include "api/ImageInputAPI.h"

#include "classifier/OnnxDefectClassifier.h"
#include "core/DetectorBackend.h"
#include "core/LabelProvider.h"
#include "detector/YoloOnnxDetector.h"
#include "label/DictLabelProvider.h"

namespace sgt {

static const std::vector<std::string> TOOL_CLASSES = {
    "cefangkaikouqi","guqian","gujian","gudao","xianjian","yating",
    "bayaqian","guchui","shaozi","7haodaobing","3haodaobing","dagumo",
    "gucuo","zhenchi","zhizhixueqian","wanzhixueqian","paqian","kekeqian",
    "xiaoduqian","xichiqian","zuzhinie","dangou","pingnie","zuzhijian",
    "zhijiaoqian","jiazhuangxianlagou","huanqian","xiaogumo","sichilagou",
    "guachi","gangsijian","xiaolagou","eliekaikouqi","yasheban","eliejian",
    "xueguanshenjingboliqi1","xueguanshenjingboliqi2","yingeboliqi",
};
static const std::vector<std::string> GRASP_CLASSES = {"work", "grasp"};

ImageInputAPI::ImageInputAPI(const std::string& toolModelPath,
                             const std::string& graspModelPath,
                             const std::string& defectModelPath,
                             const std::string& dictPath)
    : toolPath_(toolModelPath), graspPath_(graspModelPath)
    , defectPath_(defectModelPath), dictPath_(dictPath) {}

ImageInputAPI::~ImageInputAPI() = default;

void ImageInputAPI::ensureLoaded(uint8_t bit) {
    if (bit == MODE_TOOL && !toolDet_) {
        if (!toolLabels_)
            toolLabels_ = std::make_unique<DictLabelProvider>(dictPath_, TOOL_CLASSES);
        toolDet_ = std::make_unique<YoloOnnxDetector>(toolPath_, 640, 0.65f, 0.45f, toolLabels_.get());
    }
    if (bit == MODE_GRASP && !graspDet_) {
        if (!graspLabels_)
            graspLabels_ = std::make_unique<DictLabelProvider>("", GRASP_CLASSES);
        graspDet_ = std::make_unique<YoloOnnxDetector>(graspPath_, 640, 0.25f, 0.45f, graspLabels_.get());
    }
    if (bit == MODE_DEFECT && !defectCls_)
        defectCls_ = std::make_unique<OnnxDefectClassifier>(defectPath_, 512, 4, 0.50f);
}

ImageDetectionResult ImageInputAPI::detectImage(const cv::Mat& image, uint8_t modeMask) {
    ImageDetectionResult result;
    if (modeMask & MODE_TOOL) {
        ensureLoaded(MODE_TOOL);
        result.toolDetections = toolDet_->detect(image);
    }
    if (modeMask & MODE_GRASP) {
        ensureLoaded(MODE_GRASP);
        result.graspDetections = graspDet_->detect(image);
    }
    if (modeMask & MODE_DEFECT) {
        ensureLoaded(MODE_DEFECT);
        Detection full;
        full.bbox = {0, 0, float(image.cols), float(image.rows)};
        result.defectResults = defectCls_->classify(image, {full});
    }
    return result;
}

} // namespace sgt
