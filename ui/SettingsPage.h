#pragma once

#include <QWidget>

#include "core/AppSettings.h"

class QCheckBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QSlider;
class QSpinBox;

namespace xcwj::ui {

class Card;

class SettingsPage final : public QWidget {
    Q_OBJECT

public:
    explicit SettingsPage(QWidget* parent = nullptr);

private:
    AppSettings settings_;

    QSpinBox* cameraSpin_ = nullptr;

    QSlider* toolSlider_ = nullptr;
    QSlider* graspSlider_ = nullptr;
    QSlider* defectSlider_ = nullptr;
    QLabel* toolValueLabel_ = nullptr;
    QLabel* graspValueLabel_ = nullptr;
    QLabel* defectValueLabel_ = nullptr;
    QCheckBox* toolModeBox_ = nullptr;
    QCheckBox* graspModeBox_ = nullptr;
    QCheckBox* defectModeBox_ = nullptr;
    QPushButton* resetButton_ = nullptr;

    QLineEdit* captureDirEdit_ = nullptr;
    QPushButton* browseButton_ = nullptr;

    Card* buildSourceCard();
    Card* buildDefaultsCard();
    Card* buildStorageCard();

    QSlider* makeThresholdSlider(float initial, QLabel* valueLabel);
    void writeModeMask();
    void populateFromSettings();
    static QString thresholdLabelText(int percent);
};

} // namespace xcwj::ui
