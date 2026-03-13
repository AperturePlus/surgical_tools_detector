#include "render/OpenCVFontRenderer.h"

#include <algorithm>
#include <cstring>

#ifdef _WIN32
#include <windows.h>

namespace {

std::wstring utf8ToWide(const std::string& text)
{
    if (text.empty()) return {};
    const int len = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, nullptr, 0);
    if (len <= 1) return {};
    std::wstring out(static_cast<size_t>(len - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, out.data(), len);
    return out;
}

} // namespace
#endif

namespace sgt {

void OpenCVFontRenderer::putText(cv::Mat&           img,
                                 const std::string& text,
                                 cv::Point          origin,
                                 double             fontScale,
                                 cv::Scalar         color,
                                 int                thickness)
{
#ifdef _WIN32
    if (img.empty() || text.empty()) return;

    std::wstring wtext = utf8ToWide(text);
    if (wtext.empty()) return;

    const int pixelHeight = std::max(1, static_cast<int>(basePixelHeight_ * fontScale));
    const int fontWeight = (thickness >= 2) ? FW_BOLD : FW_NORMAL;

    HDC hdc = CreateCompatibleDC(nullptr);
    if (!hdc) return;

    const int fontNameLen = MultiByteToWideChar(CP_UTF8, 0, fontName_.c_str(), -1, nullptr, 0);
    std::wstring wfontName;
    if (fontNameLen > 1) {
        wfontName.resize(static_cast<size_t>(fontNameLen - 1));
        MultiByteToWideChar(CP_UTF8, 0, fontName_.c_str(), -1, wfontName.data(), fontNameLen);
    } else {
        wfontName = L"Microsoft YaHei";
    }

    HFONT hfont = CreateFontW(-pixelHeight, 0, 0, 0, fontWeight,
                              FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                              OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                              CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE,
                              wfontName.c_str());
    if (!hfont) {
        DeleteDC(hdc);
        return;
    }

    HGDIOBJ oldFont = SelectObject(hdc, hfont);
    SetBkMode(hdc, TRANSPARENT);
    SetTextColor(hdc, RGB(static_cast<int>(color[2]),
                          static_cast<int>(color[1]),
                          static_cast<int>(color[0])));

    SIZE sz{};
    if (!GetTextExtentPoint32W(hdc, wtext.c_str(), static_cast<int>(wtext.size()), &sz)) {
        SelectObject(hdc, oldFont);
        DeleteObject(hfont);
        DeleteDC(hdc);
        return;
    }

    TEXTMETRICW tm{};
    GetTextMetricsW(hdc, &tm);

    const int textX = origin.x;
    const int textTop = origin.y - tm.tmAscent;
    const int textW = std::max(1, static_cast<int>(sz.cx));
    const int textH = std::max(1, static_cast<int>(tm.tmAscent + tm.tmDescent));

    const int pad = 2 + std::max(0, thickness);
    int x0 = std::max(0, textX - pad);
    int y0 = std::max(0, textTop - pad);
    int x1 = std::min(img.cols, textX + textW + pad);
    int y1 = std::min(img.rows, textTop + textH + pad);
    if (x0 >= x1 || y0 >= y1) {
        SelectObject(hdc, oldFont);
        DeleteObject(hfont);
        DeleteDC(hdc);
        return;
    }

    cv::Rect roi(x0, y0, x1 - x0, y1 - y0);
    cv::Mat patchBgr = img(roi);
    cv::Mat patchBgra;
    cv::cvtColor(patchBgr, patchBgra, cv::COLOR_BGR2BGRA);

    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = patchBgra.cols;
    bmi.bmiHeader.biHeight = -patchBgra.rows;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    void* bits = nullptr;
    HBITMAP dib = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, &bits, nullptr, 0);
    if (!dib || !bits) {
        SelectObject(hdc, oldFont);
        DeleteObject(hfont);
        DeleteDC(hdc);
        return;
    }

    HGDIOBJ oldBmp = SelectObject(hdc, dib);
    std::memcpy(bits, patchBgra.data, static_cast<size_t>(patchBgra.total() * patchBgra.elemSize()));

    const int localX = textX - roi.x;
    const int localY = textTop - roi.y;
    TextOutW(hdc, localX, localY, wtext.c_str(), static_cast<int>(wtext.size()));

    std::memcpy(patchBgra.data, bits, static_cast<size_t>(patchBgra.total() * patchBgra.elemSize()));
    cv::cvtColor(patchBgra, patchBgr, cv::COLOR_BGRA2BGR);

    SelectObject(hdc, oldBmp);
    DeleteObject(dib);
    SelectObject(hdc, oldFont);
    DeleteObject(hfont);
    DeleteDC(hdc);
#else
    cv::putText(img, text, origin, fontFace_,
                fontScale, color, thickness, cv::LINE_AA);
#endif
}

cv::Size OpenCVFontRenderer::getTextSize(const std::string& text,
                                         double             fontScale,
                                         int                thickness,
                                         int*               baseline)
{
#ifdef _WIN32
    if (text.empty()) {
        if (baseline) *baseline = 0;
        return {0, 0};
    }

    std::wstring wtext = utf8ToWide(text);
    if (wtext.empty()) {
        if (baseline) *baseline = 0;
        return {0, 0};
    }

    const int pixelHeight = std::max(1, static_cast<int>(basePixelHeight_ * fontScale));
    const int fontWeight = (thickness >= 2) ? FW_BOLD : FW_NORMAL;

    HDC hdc = CreateCompatibleDC(nullptr);
    if (!hdc) {
        if (baseline) *baseline = 0;
        return {0, 0};
    }

    const int fontNameLen = MultiByteToWideChar(CP_UTF8, 0, fontName_.c_str(), -1, nullptr, 0);
    std::wstring wfontName;
    if (fontNameLen > 1) {
        wfontName.resize(static_cast<size_t>(fontNameLen - 1));
        MultiByteToWideChar(CP_UTF8, 0, fontName_.c_str(), -1, wfontName.data(), fontNameLen);
    } else {
        wfontName = L"Microsoft YaHei";
    }

    HFONT hfont = CreateFontW(-pixelHeight, 0, 0, 0, fontWeight,
                              FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                              OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                              CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE,
                              wfontName.c_str());
    if (!hfont) {
        DeleteDC(hdc);
        if (baseline) *baseline = 0;
        return {0, 0};
    }

    HGDIOBJ oldFont = SelectObject(hdc, hfont);

    SIZE sz{};
    GetTextExtentPoint32W(hdc, wtext.c_str(), static_cast<int>(wtext.size()), &sz);

    TEXTMETRICW tm{};
    GetTextMetricsW(hdc, &tm);

    SelectObject(hdc, oldFont);
    DeleteObject(hfont);
    DeleteDC(hdc);

    if (baseline) *baseline = tm.tmDescent;
    return {std::max(0L, sz.cx), std::max(0L, static_cast<LONG>(tm.tmAscent))};
#else
    return cv::getTextSize(text, fontFace_, fontScale, thickness, baseline);
#endif
}

} // namespace sgt
