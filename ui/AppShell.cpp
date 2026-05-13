#include "ui/AppShell.h"

#include <exception>

#include <QHBoxLayout>
#include <QMessageBox>
#include <QShortcut>
#include <QStackedWidget>
#include <QStatusBar>
#include <QWidget>

#include "core/CaptureStore.h"
#include "ui/GalleryPage.h"
#include "ui/LivePage.h"
#include "ui/SettingsPage.h"
#include "ui/Sidebar.h"

namespace sgt::ui {

AppShell::AppShell(AppOptions opts,
                   std::unique_ptr<DetectionEngine> engine,
                   std::unique_ptr<CaptureStore> store,
                   QWidget* parent)
    : QMainWindow(parent)
    , opts_(std::move(opts))
    , engine_(std::move(engine))
    , store_(std::move(store))
    , activeMask_(opts_.modeMask ? opts_.modeMask : MODE_TOOL)
    , thresholds_(opts_.thresholds)
{
    engine_->setThresholds(thresholds_);
    buildUi();
    wireEvents();
    startCamera();
}

AppShell::~AppShell()
{
    timer_.stop();
    cap_.release();
}

void AppShell::buildUi()
{
    setWindowTitle("SGTDetector");
    resize(1440, 900);

    auto* root = new QWidget(this);
    root->setObjectName("AppRoot");
    auto* layout = new QHBoxLayout(root);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    sidebar_ = new Sidebar();
    stack_ = new QStackedWidget();
    livePage_ = new LivePage(activeMask_, thresholds_);
    galleryPage_ = new GalleryPage();
    settingsPage_ = new SettingsPage();
    galleryPage_->setRecords(store_->records());
    livePage_->setModels(engine_->models());

    stack_->addWidget(livePage_);
    stack_->addWidget(galleryPage_);
    stack_->addWidget(settingsPage_);
    layout->addWidget(sidebar_);
    layout->addWidget(stack_, 1);
    setCentralWidget(root);

    statusBar()->showMessage("Ready");
}

void AppShell::wireEvents()
{
    connect(&timer_, &QTimer::timeout, this, [this]() { processFrame(); });
    connect(sidebar_, &Sidebar::pageRequested, this, [this](int index) {
        if (index < 0 || index >= stack_->count()) return;
        stack_->setCurrentIndex(index);
        sidebar_->setCurrentPage(index);
    });
    connect(livePage_, &LivePage::captureRequested, this, [this]() { captureCurrent(); });
    connect(livePage_, &LivePage::modeMaskChanged, this, [this](uint8_t mask) {
        activeMask_ = mask ? mask : MODE_TOOL;
    });
    connect(livePage_, &LivePage::thresholdsChanged, this, [this](const DetectionThresholds& thresholds) {
        thresholds_ = thresholds;
        engine_->setThresholds(thresholds_);
    });

    new QShortcut(QKeySequence(Qt::Key_1), this, [this]() { livePage_->toggleMode(MODE_TOOL); });
    new QShortcut(QKeySequence(Qt::Key_2), this, [this]() { livePage_->toggleMode(MODE_GRASP); });
    new QShortcut(QKeySequence(Qt::Key_3), this, [this]() { livePage_->toggleMode(MODE_DEFECT); });
    new QShortcut(QKeySequence(Qt::Key_C), this, [this]() { captureCurrent(); });
    new QShortcut(QKeySequence(Qt::Key_Q), this, [this]() { close(); });
    new QShortcut(QKeySequence(Qt::Key_Escape), this, [this]() { close(); });
}

void AppShell::startCamera()
{
    cap_.open(opts_.cameraId);
    if (!cap_.isOpened()) {
        livePage_->setCameraStatus("Camera unavailable");
        statusBar()->showMessage("Cannot open camera " + QString::number(opts_.cameraId));
        return;
    }

    cap_.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    livePage_->setCameraStatus("Camera " + QString::number(opts_.cameraId) + " online");
    timer_.start(1);
}

void AppShell::processFrame()
{
    if (!cap_.isOpened()) return;

    cv::Mat frame;
    if (!cap_.read(frame) || frame.empty()) return;

    try {
        lastResult_ = engine_->process(frame, activeMask_);
        updateFps();
        lastResult_.fps = fps_;
        livePage_->setFrameResult(lastResult_, activeMask_);
    } catch (const std::exception& e) {
        statusBar()->showMessage("Inference error: " + QString::fromLocal8Bit(e.what()), 3000);
    }
}

void AppShell::updateFps()
{
    QDateTime now = QDateTime::currentDateTimeUtc();
    if (lastFrameTime_.isValid()) {
        qint64 ms = lastFrameTime_.msecsTo(now);
        if (ms > 0) {
            float instant = 1000.0f / static_cast<float>(ms);
            fps_ = fps_ * 0.9f + instant * 0.1f;
        }
    }
    lastFrameTime_ = now;
    livePage_->setFps(fps_);
}

void AppShell::captureCurrent()
{
    if (lastResult_.rawFrame.empty()) {
        statusBar()->showMessage("No frame to capture", 2500);
        return;
    }
    try {
        auto record = store_->saveCapture(lastResult_, engine_->models());
        statusBar()->showMessage("Capture saved: " + QString::fromStdString(record.id), 3500);
        galleryPage_->addRecord(record);
    } catch (const std::exception& e) {
        QMessageBox::warning(this, "Capture failed", QString::fromLocal8Bit(e.what()));
    }
}

} // namespace sgt::ui
