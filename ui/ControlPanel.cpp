#include "ui/ControlPanel.h"

#include <filesystem>

#include <QAbstractItemView>
#include <QColor>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSlider>
#include <QStringList>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QVBoxLayout>

#include "core/Renderer.h"
#include "ui/Card.h"
#include "ui/ThemeManager.h"

namespace xcwj::ui {

namespace {

QString basename(const std::string& path)
{
    if (path.empty()) return "Not configured";
    return QString::fromStdString(std::filesystem::path(path).filename().string());
}

QTableWidgetItem* makeItem(const QString& text)
{
    auto* item = new QTableWidgetItem(text);
    item->setFlags(item->flags() & ~Qt::ItemIsEditable);
    return item;
}

void colorRow(QTableWidget* table, int row, const QColor& color)
{
    for (int col = 0; col < table->columnCount(); ++col) {
        if (auto* item = table->item(row, col)) {
            item->setForeground(color);
        }
    }
}

} // namespace

ControlPanel::ControlPanel(uint8_t initialModeMask,
                           const DetectionThresholds& initialThresholds,
                           QWidget* parent)
    : QFrame(parent)
    , currentModeMask_(initialModeMask ? initialModeMask : MODE_TOOL)
    , thresholds_(initialThresholds)
{
    setObjectName("ControlRail");
    setMinimumWidth(340);

    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(12);

    auto* thresholdsCard = new Card("Thresholds");
    toolSlider_ = makeSlider(thresholds_.tool, toolThresholdLabel_, "Tool");
    graspSlider_ = makeSlider(thresholds_.grasp, graspThresholdLabel_, "Grasp");
    defectSlider_ = makeSlider(thresholds_.defect, defectThresholdLabel_, "Defect");
    thresholdsCard->bodyLayout()->addWidget(makeSliderRow(toolSlider_, toolThresholdLabel_));
    thresholdsCard->bodyLayout()->addWidget(makeSliderRow(graspSlider_, graspThresholdLabel_));
    thresholdsCard->bodyLayout()->addWidget(makeSliderRow(defectSlider_, defectThresholdLabel_));
    auto* resetButton = new QPushButton("Reset thresholds");
    resetButton->setCursor(Qt::PointingHandCursor);
    thresholdsCard->bodyLayout()->addWidget(resetButton);
    layout->addWidget(thresholdsCard);

    auto* dataCard = new Card("Live Data");
    resultTable_ = new QTableWidget(0, 3);
    resultTable_->setHorizontalHeaderLabels(QStringList{"Mode", "Label", "Score"});
    resultTable_->verticalHeader()->setVisible(false);
    resultTable_->horizontalHeader()->setStretchLastSection(true);
    resultTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    resultTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    resultTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    resultTable_->setSelectionMode(QAbstractItemView::NoSelection);
    resultTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    resultTable_->setAlternatingRowColors(false);
    resultTable_->setMinimumHeight(220);
    dataCard->bodyLayout()->addWidget(resultTable_);
    layout->addWidget(dataCard, 1);

    auto* modelsCard = new Card("Models");
    toolModelLabel_ = new QLabel("Tool: Not configured");
    graspModelLabel_ = new QLabel("Grasp: Not configured");
    defectModelLabel_ = new QLabel("Defect: Not configured");
    toolModelLabel_->setObjectName("SubtleText");
    graspModelLabel_->setObjectName("SubtleText");
    defectModelLabel_->setObjectName("SubtleText");
    modelsCard->bodyLayout()->addWidget(toolModelLabel_);
    modelsCard->bodyLayout()->addWidget(graspModelLabel_);
    modelsCard->bodyLayout()->addWidget(defectModelLabel_);
    layout->addWidget(modelsCard);

    connect(resetButton, &QPushButton::clicked, this, [this]() {
        thresholds_ = DetectionThresholds{};
        {
            const QSignalBlocker b(toolSlider_);
            toolSlider_->setValue(static_cast<int>(thresholds_.tool * 100.0f));
        }
        {
            const QSignalBlocker b(graspSlider_);
            graspSlider_->setValue(static_cast<int>(thresholds_.grasp * 100.0f));
        }
        {
            const QSignalBlocker b(defectSlider_);
            defectSlider_->setValue(static_cast<int>(thresholds_.defect * 100.0f));
        }
        updateThresholdLabel(toolThresholdLabel_, "Tool", toolSlider_->value());
        updateThresholdLabel(graspThresholdLabel_, "Grasp", graspSlider_->value());
        updateThresholdLabel(defectThresholdLabel_, "Defect", defectSlider_->value());
        emit thresholdsChanged(thresholds_);
    });

    connect(toolSlider_, &QSlider::valueChanged, this, [this](int value) {
        thresholds_.tool = value / 100.0f;
        updateThresholdLabel(toolThresholdLabel_, "Tool", value);
        emit thresholdsChanged(thresholds_);
    });
    connect(graspSlider_, &QSlider::valueChanged, this, [this](int value) {
        thresholds_.grasp = value / 100.0f;
        updateThresholdLabel(graspThresholdLabel_, "Grasp", value);
        emit thresholdsChanged(thresholds_);
    });
    connect(defectSlider_, &QSlider::valueChanged, this, [this](int value) {
        thresholds_.defect = value / 100.0f;
        updateThresholdLabel(defectThresholdLabel_, "Defect", value);
        emit thresholdsChanged(thresholds_);
    });
}

uint8_t ControlPanel::modeMask() const
{
    return currentModeMask_;
}

void ControlPanel::setFrameResult(const DetectionFrameResult& result)
{
    if (tableUpdateTimer_.isValid() && tableUpdateTimer_.elapsed() < 200) {
        return;
    }
    if (tableUpdateTimer_.isValid()) {
        tableUpdateTimer_.restart();
    } else {
        tableUpdateTimer_.start();
    }

    const int rows = static_cast<int>(result.toolDetections.size()
        + result.graspDetections.size()
        + result.defectResults.size());
    resultTable_->setRowCount(rows);

    int row = 0;
    auto addDetection = [this, &row](const QString& mode, const QString& label, float score) {
        resultTable_->setItem(row, 0, makeItem(mode));
        resultTable_->setItem(row, 1, makeItem(label));
        resultTable_->setItem(row, 2, makeItem(QString("%1%").arg(score * 100.0f, 0, 'f', 1)));
        ++row;
    };

    for (const auto& d : result.toolDetections) {
        addDetection("Tool", QString::fromStdString(d.label), d.score);
    }
    for (const auto& d : result.graspDetections) {
        addDetection("Grasp", QString::fromStdString(d.label), d.score);
    }
    for (const auto& d : result.defectResults) {
        const int defectRow = row;
        addDetection("Defect", d.defective ? "Defective" : "Normal", d.defectScore);
        if (d.defective) {
            colorRow(resultTable_, defectRow, QColor(ThemeManager::instance().tokens().warn));
        }
    }
}

void ControlPanel::setModels(const ModelInfo& models)
{
    toolModelLabel_->setText("Tool: " + basename(models.toolModel));
    graspModelLabel_->setText("Grasp: " + basename(models.graspModel));
    defectModelLabel_->setText("Defect: " + basename(models.defectModel));
}

void ControlPanel::setModeMask(uint8_t modeMask)
{
    emitModeIfChanged(modeMask ? modeMask : MODE_TOOL);
}

void ControlPanel::toggleMode(uint8_t bit)
{
    uint8_t next = currentModeMask_ ^ bit;
    if (!next) next = bit;
    emitModeIfChanged(next);
}

void ControlPanel::toggleToolMode()
{
    toggleMode(MODE_TOOL);
}

void ControlPanel::toggleGraspMode()
{
    toggleMode(MODE_GRASP);
}

void ControlPanel::toggleDefectMode()
{
    toggleMode(MODE_DEFECT);
}

QSlider* ControlPanel::makeSlider(float value, QLabel*& label, const QString& name)
{
    label = new QLabel();
    label->setObjectName("SubtleText");
    auto* slider = new QSlider(Qt::Horizontal);
    slider->setRange(5, 95);
    slider->setValue(static_cast<int>(value * 100.0f));
    slider->setMinimumHeight(30);
    updateThresholdLabel(label, name, slider->value());
    return slider;
}

QWidget* ControlPanel::makeSliderRow(QSlider* slider, QLabel* label)
{
    auto* widget = new QWidget();
    auto* layout = new QVBoxLayout(widget);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(4);
    layout->addWidget(label);
    layout->addWidget(slider);
    return widget;
}

void ControlPanel::updateThresholdLabel(QLabel* label, const QString& name, int value)
{
    label->setText(QString("%1 threshold %2%").arg(name).arg(value));
}

void ControlPanel::emitModeIfChanged(uint8_t next)
{
    if (next == currentModeMask_) return;
    currentModeMask_ = next;
    emit modeMaskChanged(currentModeMask_);
}

} // namespace xcwj::ui
