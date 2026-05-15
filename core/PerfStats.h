#pragma once

namespace sgt {

struct PerfStats {
    double cameraReadMs = 0.0;
    double preprocessMs = 0.0;
    double inputCastMs = 0.0;
    double ortRunMs = 0.0;
    double outputCastMs = 0.0;
    double postprocessMs = 0.0;
    double annotateMs = 0.0;
    double qtImageMs = 0.0;
    double qtScaleDisplayMs = 0.0;
    double totalFrameMs = 0.0;

    PerfStats& operator+=(const PerfStats& rhs)
    {
        cameraReadMs += rhs.cameraReadMs;
        preprocessMs += rhs.preprocessMs;
        inputCastMs += rhs.inputCastMs;
        ortRunMs += rhs.ortRunMs;
        outputCastMs += rhs.outputCastMs;
        postprocessMs += rhs.postprocessMs;
        annotateMs += rhs.annotateMs;
        qtImageMs += rhs.qtImageMs;
        qtScaleDisplayMs += rhs.qtScaleDisplayMs;
        totalFrameMs += rhs.totalFrameMs;
        return *this;
    }
};

inline PerfStats averagePerfStats(const PerfStats& totals, int frames)
{
    if (frames <= 0) return {};
    const double inv = 1.0 / static_cast<double>(frames);
    PerfStats out;
    out.cameraReadMs = totals.cameraReadMs * inv;
    out.preprocessMs = totals.preprocessMs * inv;
    out.inputCastMs = totals.inputCastMs * inv;
    out.ortRunMs = totals.ortRunMs * inv;
    out.outputCastMs = totals.outputCastMs * inv;
    out.postprocessMs = totals.postprocessMs * inv;
    out.annotateMs = totals.annotateMs * inv;
    out.qtImageMs = totals.qtImageMs * inv;
    out.qtScaleDisplayMs = totals.qtScaleDisplayMs * inv;
    out.totalFrameMs = totals.totalFrameMs * inv;
    return out;
}

} // namespace sgt
