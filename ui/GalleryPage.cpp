#include "ui/GalleryPage.h"

#include <algorithm>
#include <functional>
#include <map>

#include <QDate>
#include <QDateTime>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QScrollArea>
#include <QVBoxLayout>

#include "ui/CaptureDetailDialog.h"
#include "ui/FlowLayout.h"
#include "ui/GalleryFilterBar.h"
#include "ui/IconLoader.h"
#include "ui/StatusChip.h"
#include "ui/ThemeManager.h"
#include "ui/ThumbCard.h"

namespace xcwj::ui {

GalleryPage::GalleryPage(QWidget* parent)
    : QWidget(parent)
{
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(20, 18, 20, 18);
    root->setSpacing(14);

    auto* header = new QHBoxLayout();
    header->setSpacing(10);
    titleLabel_ = new QLabel("Captures");
    titleLabel_->setObjectName("AppTitle");
    header->addWidget(titleLabel_);

    countChip_ = new StatusChip("0");
    countChip_->setDotColor(ThemeManager::instance().tokens().accent);
    header->addWidget(countChip_);
    header->addSpacing(6);

    connect(&ThemeManager::instance(), &ThemeManager::themeChanged, this, [this](const ThemeTokens& t) {
        countChip_->setDotColor(t.accent);
    });

    searchEdit_ = new QLineEdit();
    searchEdit_->setPlaceholderText("Search id or date");
    searchEdit_->setMinimumWidth(260);
    header->addWidget(searchEdit_);
    header->addStretch();

    auto* exportButton = new QPushButton("Export all...");
    exportButton->setObjectName("Secondary");
    exportButton->setEnabled(false);
    exportButton->setToolTip("Export is planned for a later phase.");
    header->addWidget(exportButton);
    root->addLayout(header);

    filterBar_ = new GalleryFilterBar();
    root->addWidget(filterBar_);
    connect(filterBar_, &GalleryFilterBar::rangeChanged, this,
            [this](QDate from, QDate to) {
                filterFrom_ = from;
                filterTo_ = to;
                rebuild();
            });

    auto* scroll = new QScrollArea();
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    content_ = new QWidget();
    contentLayout_ = new QVBoxLayout(content_);
    contentLayout_->setContentsMargins(0, 0, 0, 0);
    contentLayout_->setSpacing(10);
    scroll->setWidget(content_);
    root->addWidget(scroll, 1);

    connect(searchEdit_, &QLineEdit::textChanged, this, [this]() { rebuild(); });
}

void GalleryPage::setRecords(const std::vector<CaptureRecord>& records)
{
    records_ = records;
    std::sort(records_.begin(), records_.end(), [](const CaptureRecord& a, const CaptureRecord& b) {
        return a.timestamp > b.timestamp;
    });
    rebuild();
}

void GalleryPage::addRecord(const CaptureRecord& record)
{
    const auto existing = std::find_if(records_.begin(), records_.end(), [&](const CaptureRecord& item) {
        return item.id == record.id;
    });
    if (existing == records_.end()) {
        records_.push_back(record);
    } else {
        *existing = record;
    }
    setRecords(records_);
}

void GalleryPage::rebuild()
{
    titleLabel_->setText("Captures");
    countChip_->setText(QString::number(records_.size()));

    QLayoutItem* item = nullptr;
    while ((item = contentLayout_->takeAt(0))) {
        if (auto* widget = item->widget()) {
            delete widget;
        }
        delete item;
    }

    std::map<QString, std::vector<CaptureRecord>, std::greater<QString>> groups;
    for (const auto& record : records_) {
        if (!matchesFilter(record)) continue;
        const QString ts = QString::fromStdString(record.timestamp);
        groups[ts.left(10)].push_back(record);
    }

    if (groups.empty()) {
        auto* empty = new QWidget();
        auto* emptyLayout = new QVBoxLayout(empty);
        emptyLayout->setAlignment(Qt::AlignCenter);
        emptyLayout->setSpacing(8);
        const bool noCaptures = records_.empty();
        const auto& tokens = ThemeManager::instance().tokens();
        auto* icon = new QLabel();
        icon->setPixmap(IconLoader::load("nav-gallery", tokens.textSecondary, QSize(32, 32)).pixmap(QSize(32, 32)));
        icon->setAlignment(Qt::AlignCenter);
        auto* primary = new QLabel(noCaptures ? "No captures yet" : "No matches");
        primary->setObjectName("PanelTitle");
        primary->setAlignment(Qt::AlignCenter);
        auto* secondary = new QLabel(noCaptures
            ? "Press Capture or 'C' in Live."
            : "Try a different search or date range.");
        secondary->setObjectName("SubtleText");
        secondary->setAlignment(Qt::AlignCenter);
        emptyLayout->addWidget(icon);
        emptyLayout->addWidget(primary);
        emptyLayout->addWidget(secondary);
        empty->setMinimumHeight(320);
        contentLayout_->addWidget(empty, 1);
        return;
    }

    for (const auto& [day, dayRecords] : groups) {
        auto* heading = new QLabel(dateHeading(day));
        heading->setObjectName("DateHeading");
        contentLayout_->addWidget(heading);

        auto* groupWidget = new QWidget();
        auto* flow = new FlowLayout(groupWidget, 0, 12, 12);
        for (const auto& record : dayRecords) {
            auto* card = new ThumbCard(record);
            connect(card, &ThumbCard::activated, this, &GalleryPage::openDetail);
            flow->addWidget(card);
        }
        contentLayout_->addWidget(groupWidget);
    }
    contentLayout_->addStretch();
}

void GalleryPage::openDetail(const QString& id)
{
    const auto it = std::find_if(records_.begin(), records_.end(), [&](const CaptureRecord& record) {
        return QString::fromStdString(record.id) == id;
    });
    if (it == records_.end()) return;

    const int index = static_cast<int>(std::distance(records_.begin(), it));
    CaptureDetailDialog dialog(records_, index, this);
    dialog.exec();
}

bool GalleryPage::matchesFilter(const CaptureRecord& record) const
{
    return matchesSearch(record) && matchesDateRange(record);
}

bool GalleryPage::matchesSearch(const CaptureRecord& record) const
{
    const QString needle = searchEdit_->text().trimmed();
    if (needle.isEmpty()) return true;
    const QString id = QString::fromStdString(record.id);
    const QString timestamp = QString::fromStdString(record.timestamp);
    return id.contains(needle, Qt::CaseInsensitive)
        || timestamp.contains(needle, Qt::CaseInsensitive);
}

bool GalleryPage::matchesDateRange(const CaptureRecord& record) const
{
    if (!filterFrom_.isValid() && !filterTo_.isValid()) return true;
    const QDate captureDay = QDate::fromString(
        QString::fromStdString(record.timestamp).left(10), Qt::ISODate);
    if (!captureDay.isValid()) return false;
    if (filterFrom_.isValid() && captureDay < filterFrom_) return false;
    if (filterTo_.isValid()   && captureDay > filterTo_)   return false;
    return true;
}

QString GalleryPage::dateHeading(const QString& timestamp) const
{
    const QDate date = QDate::fromString(timestamp.left(10), Qt::ISODate);
    if (date.isValid() && date == QDate::currentDate()) {
        return "Today";
    }
    return timestamp.isEmpty() ? "Unknown date" : timestamp.left(10);
}

} // namespace xcwj::ui
