#include "ui/SettingsPage.h"

#include <QCheckBox>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSlider>
#include <QSpinBox>
#include <QVBoxLayout>

#include "core/Renderer.h"
#include "ui/Card.h"

namespace xcwj::ui {

namespace {

constexpr int kSliderMin = 5;
constexpr int kSliderMax = 95;

QLabel* makeHintLabel()
{
    auto* hint = new QLabel("Applies on next launch.");
    hint->setObjectName("SubtleText");
    return hint;
}

} // namespace

SettingsPage::SettingsPage(QWidget* parent)
    : QWidget(parent)
{
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(22, 18, 22, 18);
    root->setSpacing(14);

    auto* title = new QLabel("Settings");
    title->setObjectName("AppTitle");
    auto* subtitle = new QLabel("Preferences are saved automatically. Some changes apply on next launch.");
    subtitle->setObjectName("SubtleText");
    root->addWidget(title);
    root->addWidget(subtitle);

    root->addWidget(buildSourceCard());
    root->addWidget(buildDefaultsCard());
    root->addWidget(buildStorageCard());
    root->addStretch();

    populateFromSettings();
}

Card* SettingsPage::buildSourceCard()
{
    auto* card = new Card("Capture Source");
    auto* row = new QHBoxLayout();
    row->setSpacing(10);
    auto* label = new QLabel("Camera index");
    cameraSpin_ = new QSpinBox();
    cameraSpin_->setRange(0, 15);
    cameraSpin_->setFixedWidth(96);
    row->addWidget(label);
    row->addStretch();
    row->addWidget(cameraSpin_);
    card->bodyLayout()->addLayout(row);
    card->bodyLayout()->addWidget(makeHintLabel());

    connect(cameraSpin_, qOverload<int>(&QSpinBox::valueChanged), this,
            [this](int value) { settings_.setCameraId(value); });
    return card;
}

Card* SettingsPage::buildDefaultsCard()
{
    auto* card = new Card("Detection Defaults");
    auto* body = card->bodyLayout();

    auto addSliderRow = [&](const QString& name,
                            QSlider*& slider, QLabel*& valueLabel) {
        auto* row = new QHBoxLayout();
        auto* nameLabel = new QLabel(name);
        valueLabel = new QLabel("--");
        valueLabel->setObjectName("SubtleText");
        valueLabel->setMinimumWidth(48);
        valueLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        row->addWidget(nameLabel);
        row->addStretch();
        row->addWidget(valueLabel);
        body->addLayout(row);
        slider = makeThresholdSlider(0.5f, valueLabel);
        body->addWidget(slider);
    };

    addSliderRow("Tool threshold",   toolSlider_,   toolValueLabel_);
    addSliderRow("Grasp threshold",  graspSlider_,  graspValueLabel_);
    addSliderRow("Defect threshold", defectSlider_, defectValueLabel_);

    auto* modesRow = new QHBoxLayout();
    auto* modesLabel = new QLabel("Active modes");
    toolModeBox_   = new QCheckBox("Tool");
    graspModeBox_  = new QCheckBox("Grasp");
    defectModeBox_ = new QCheckBox("Defect");
    modesRow->addWidget(modesLabel);
    modesRow->addStretch();
    modesRow->addWidget(toolModeBox_);
    modesRow->addWidget(graspModeBox_);
    modesRow->addWidget(defectModeBox_);
    body->addLayout(modesRow);

    auto* footer = new QHBoxLayout();
    footer->addWidget(makeHintLabel());
    footer->addStretch();
    resetButton_ = new QPushButton("Reset to factory");
    resetButton_->setCursor(Qt::PointingHandCursor);
    footer->addWidget(resetButton_);
    body->addLayout(footer);

    auto wireSlider = [this](QSlider* slider, QLabel* label, float DetectionThresholds::* member) {
        connect(slider, &QSlider::valueChanged, this,
                [this, label, member](int value) {
                    label->setText(thresholdLabelText(value));
                    DetectionThresholds t = settings_.defaultThresholds();
                    t.*member = value / 100.0f;
                    settings_.setDefaultThresholds(t);
                });
    };
    wireSlider(toolSlider_,   toolValueLabel_,   &DetectionThresholds::tool);
    wireSlider(graspSlider_,  graspValueLabel_,  &DetectionThresholds::grasp);
    wireSlider(defectSlider_, defectValueLabel_, &DetectionThresholds::defect);

    auto wireModeBox = [this](QCheckBox* box) {
        connect(box, &QCheckBox::toggled, this, [this]() { writeModeMask(); });
    };
    wireModeBox(toolModeBox_);
    wireModeBox(graspModeBox_);
    wireModeBox(defectModeBox_);

    connect(resetButton_, &QPushButton::clicked, this, [this]() {
        settings_.resetDefaults();
        populateFromSettings();
    });

    return card;
}

Card* SettingsPage::buildStorageCard()
{
    auto* card = new Card("Capture Storage");
    auto* row = new QHBoxLayout();
    row->setSpacing(8);
    auto* label = new QLabel("Output folder");
    captureDirEdit_ = new QLineEdit();
    captureDirEdit_->setReadOnly(true);
    captureDirEdit_->setPlaceholderText("Default: <exe-dir>/captures");
    browseButton_ = new QPushButton("Browse...");
    browseButton_->setCursor(Qt::PointingHandCursor);
    row->addWidget(label);
    row->addWidget(captureDirEdit_, 1);
    row->addWidget(browseButton_);
    card->bodyLayout()->addLayout(row);
    card->bodyLayout()->addWidget(makeHintLabel());

    connect(browseButton_, &QPushButton::clicked, this, [this]() {
        const QString chosen = QFileDialog::getExistingDirectory(
            this, "Choose capture output folder", captureDirEdit_->text());
        if (chosen.isEmpty()) return;
        captureDirEdit_->setText(chosen);
        settings_.setCaptureDir(chosen);
    });
    return card;
}

QSlider* SettingsPage::makeThresholdSlider(float initial, QLabel* /*valueLabel*/)
{
    auto* slider = new QSlider(Qt::Horizontal);
    slider->setRange(kSliderMin, kSliderMax);
    slider->setValue(static_cast<int>(initial * 100.0f));
    slider->setMinimumHeight(30);
    return slider;
}

void SettingsPage::writeModeMask()
{
    uint8_t mask = 0;
    if (toolModeBox_->isChecked())   mask |= MODE_TOOL;
    if (graspModeBox_->isChecked())  mask |= MODE_GRASP;
    if (defectModeBox_->isChecked()) mask |= MODE_DEFECT;
    if (!mask) {
        QSignalBlocker block(toolModeBox_);
        toolModeBox_->setChecked(true);
        mask = MODE_TOOL;
    }
    settings_.setModeMask(mask);
}

void SettingsPage::populateFromSettings()
{
    {
        QSignalBlocker b(cameraSpin_);
        cameraSpin_->setValue(settings_.cameraId());
    }

    const auto t = settings_.defaultThresholds();
    auto setSlider = [](QSlider* s, QLabel* l, float v) {
        QSignalBlocker b(s);
        const int percent = static_cast<int>(v * 100.0f);
        s->setValue(percent);
        l->setText(thresholdLabelText(percent));
    };
    setSlider(toolSlider_,   toolValueLabel_,   t.tool);
    setSlider(graspSlider_,  graspValueLabel_,  t.grasp);
    setSlider(defectSlider_, defectValueLabel_, t.defect);

    const uint8_t mask = settings_.modeMask();
    auto setBox = [](QCheckBox* box, bool checked) {
        QSignalBlocker b(box);
        box->setChecked(checked);
    };
    setBox(toolModeBox_,   mask & MODE_TOOL);
    setBox(graspModeBox_,  mask & MODE_GRASP);
    setBox(defectModeBox_, mask & MODE_DEFECT);

    captureDirEdit_->setText(settings_.captureDir());
}

QString SettingsPage::thresholdLabelText(int percent)
{
    return QStringLiteral("%1%").arg(percent);
}

} // namespace xcwj::ui
