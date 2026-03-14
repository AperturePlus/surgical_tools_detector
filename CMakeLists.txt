cmake_minimum_required(VERSION 3.16)
project(SGTDetector VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─────────────────────────────────────────────────────────────────────────────
# Optional features — enable at configure time:
#   cmake -B build -G Ninja                         (CPU only, OpenCV HighGUI)
#   cmake -B build -G Ninja -DWITH_CUDA=ON          (CUDA EP for GPU inference)
#   cmake -B build -G Ninja -DWITH_FREETYPE=ON      (FreeType for CJK text)
#   cmake -B build -G Ninja -DWITH_IMGUI=ON         (Dear ImGui renderer)
# ─────────────────────────────────────────────────────────────────────────────
option(WITH_FREETYPE "Enable FreeType font renderer for CJK/Unicode text" OFF)
option(WITH_CUDA     "Enable CUDA Execution Provider for ONNX Runtime"    OFF)
option(WITH_IMGUI    "Enable Dear ImGui renderer"                          OFF)

# ─────────────────────────────────────────────────────────────────────────────
# OpenCV
# ─────────────────────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio dnn)
message(STATUS "OpenCV ${OpenCV_VERSION} found: ${OpenCV_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# ONNX Runtime — bundled prebuilt in deps/, with MinGW import library
#
# The official Windows x64 release ships an MSVC .lib; we use the MinGW
# import library (libonnxruntime.a) generated via dlltool from the DLL.
# To upgrade ORT, download a new release zip to deps/, run:
#   gendef.exe onnxruntime.dll
#   dlltool.exe -d onnxruntime.def -D onnxruntime.dll -l libonnxruntime.a
# ─────────────────────────────────────────────────────────────────────────────
set(ORT_ROOT "${CMAKE_SOURCE_DIR}/deps/onnxruntime-win-x64-1.24.3")
set(ORT_INCLUDE_DIR "${ORT_ROOT}/include")
set(ORT_LIB_DIR     "${ORT_ROOT}/lib")

if(NOT EXISTS "${ORT_INCLUDE_DIR}/onnxruntime_cxx_api.h")
    message(FATAL_ERROR
        "ONNX Runtime headers not found at ${ORT_INCLUDE_DIR}\n"
        "Check that deps/onnxruntime-win-x64-1.24.3 is present.")
endif()
if(NOT EXISTS "${ORT_LIB_DIR}/libonnxruntime.a")
    message(FATAL_ERROR
        "MinGW import library not found: ${ORT_LIB_DIR}/libonnxruntime.a\n"
        "Generate it with:\n"
        "  gendef.exe ${ORT_LIB_DIR}/onnxruntime.dll\n"
        "  dlltool.exe -d onnxruntime.def -D onnxruntime.dll -l ${ORT_LIB_DIR}/libonnxruntime.a")
endif()

message(STATUS "ONNX Runtime 1.24.3 (bundled): ${ORT_LIB_DIR}")
add_library(onnxruntime STATIC IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION             "${ORT_LIB_DIR}/libonnxruntime.a"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}"
    INTERFACE_COMPILE_DEFINITIONS "ORT_NO_EXCEPTIONS=0"
)

# ─────────────────────────────────────────────────────────────────────────────
# Sources
# ─────────────────────────────────────────────────────────────────────────────
set(SGT_SOURCES
    main.cpp
    detector/YoloOnnxDetector.cpp
    label/DictLabelProvider.cpp
    render/OpenCVFontRenderer.cpp
    render/OpenCVRenderer.cpp
)
set(SGT_HEADERS
    core/Detection.h
    core/DetectorBackend.h
    core/LabelProvider.h
    core/Renderer.h
    core/FontRenderer.h
    detector/YoloOnnxDetector.h
    label/DictLabelProvider.h
    render/OpenCVFontRenderer.h
    render/OpenCVRenderer.h
)

# ── Optional: FreeType ────────────────────────────────────────────────────────
if(WITH_FREETYPE)
    find_package(Freetype REQUIRED)
    list(APPEND SGT_SOURCES render/FreeTypeFontRenderer.cpp)
    list(APPEND SGT_HEADERS render/FreeTypeFontRenderer.h)
    message(STATUS "FreeType support enabled")
endif()

# ── Optional: Dear ImGui ──────────────────────────────────────────────────────
if(WITH_IMGUI)
    find_package(imgui CONFIG REQUIRED)
    list(APPEND SGT_SOURCES render/ImGuiRenderer.cpp)
    list(APPEND SGT_HEADERS render/ImGuiRenderer.h)
    message(STATUS "Dear ImGui support enabled")
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Executable
# ─────────────────────────────────────────────────────────────────────────────
add_executable(SGTDetector ${SGT_SOURCES} ${SGT_HEADERS})

target_include_directories(SGTDetector PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(SGTDetector PRIVATE
    ${OpenCV_LIBS}
    onnxruntime
)

# ── Feature compile definitions ───────────────────────────────────────────────
if(WITH_CUDA)
    target_compile_definitions(SGTDetector PRIVATE SGT_WITH_CUDA)
    message(STATUS "CUDA EP support enabled (runtime GPU selection)")
endif()
if(WITH_FREETYPE)
    target_compile_definitions(SGTDetector PRIVATE SGT_WITH_FREETYPE)
    target_link_libraries(SGTDetector PRIVATE Freetype::Freetype)
endif()
if(WITH_IMGUI)
    target_compile_definitions(SGTDetector PRIVATE SGT_WITH_IMGUI)
    target_link_libraries(SGTDetector PRIVATE imgui::imgui)
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Build type defaults and flags
# ─────────────────────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

target_compile_options(SGTDetector PRIVATE
    $<$<CONFIG:Release>:-O2>
    $<$<CONFIG:Debug>:-g -O0>
)

# ─────────────────────────────────────────────────────────────────────────────
# Copy runtime assets next to the executable after build
# ─────────────────────────────────────────────────────────────────────────────
foreach(_asset best.onnx labels.dict)
    add_custom_command(TARGET SGTDetector POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/assets/${_asset}"
            "$<TARGET_FILE_DIR:SGTDetector>/${_asset}"
        COMMENT "Copying ${_asset}...")
endforeach()

# Copy ONNX Runtime DLLs (required at runtime on Windows)
foreach(_dll onnxruntime.dll onnxruntime_providers_shared.dll)
    add_custom_command(TARGET SGTDetector POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ORT_LIB_DIR}/${_dll}"
            "$<TARGET_FILE_DIR:SGTDetector>/${_dll}"
        COMMENT "Copying ${_dll}...")
endforeach()

# ─────────────────────────────────────────────────────────────────────────────
# DLL deployment on Windows (MinGW / MSYS2 builds)
#
# Strategy: after each build, use ldd to discover EVERY DLL the executable
# needs and copy all non-system DLLs to the output directory.  This replaces
# the old hard-coded DLL list and automatically picks up new dependencies
# without requiring manual CMakeLists.txt edits.
#
# Qt6 plugins (platform plugin, image formats, styles, …) are loaded at
# runtime and are therefore invisible to ldd.  They are deployed with
# windeployqt6 when available; otherwise qwindows.dll is copied manually.
# ─────────────────────────────────────────────────────────────────────────────
get_filename_component(MINGW_BIN "${CMAKE_CXX_COMPILER}" DIRECTORY)

# --- ldd-based auto-deploy script -------------------------------------------
# Written to the build tree at configure time; invoked as a POST_BUILD step.
file(WRITE "${CMAKE_BINARY_DIR}/deploy_runtime_dlls.sh" [=[
#!/usr/bin/env bash
# Copies every non-system DLL that <exe> depends on into <out_dir>.
# Usage: bash deploy_runtime_dlls.sh <exe> <out_dir>
set -euo pipefail

EXE="$1"
OUT="$2"

if ! command -v ldd >/dev/null 2>&1; then
    echo "WARNING: ldd not found; skipping auto DLL deployment." >&2
    exit 0
fi

ldd "$EXE" | while IFS= read -r line; do
    # ldd output (MSYS2): "  libfoo.dll => /path/to/libfoo.dll (0xADDR)"
    # Capture the resolved path after "=>" — always a Unix-style absolute path
    # in MSYS2 (e.g. /c/Windows/System32/… or /mingw64/bin/…).
    if [[ "$line" =~ '=>'[[:space:]]+(/[^[:space:]]+\.[Dd][Ll][Ll]) ]]; then
        src="${BASH_REMATCH[1]}"
        [[ -f "$src" ]] || continue
        # Skip Windows system DLLs (/c/Windows, /C/Windows, /c/windows, etc.)
        lower="$(echo "$src" | tr '[:upper:]' '[:lower:]')"
        [[ "$lower" == */windows/* ]] && continue
        [[ "$lower" == */system32/* ]] && continue
        name="$(basename "$src")"
        if cp -f "$src" "$OUT/$name" 2>/dev/null; then
            echo "Deployed: $name"
        else
            echo "WARNING: failed to copy $src" >&2
        fi
    fi
done
]=])

add_custom_command(TARGET SGTDetector POST_BUILD
    COMMAND bash "${CMAKE_BINARY_DIR}/deploy_runtime_dlls.sh"
        "$<TARGET_FILE:SGTDetector>"
        "$<TARGET_FILE_DIR:SGTDetector>"
    COMMENT "Auto-deploying runtime DLLs via ldd...")

# --- Qt6 plugins (runtime-loaded; invisible to ldd) -------------------------
# windeployqt6 is the canonical tool for this; fall back to copying qwindows.dll
# manually when it is not available.
find_program(WINDEPLOYQT6_EXECUTABLE
    NAMES windeployqt6 windeployqt6.exe
    HINTS "${MINGW_BIN}"
    NO_DEFAULT_PATH
)
if(NOT WINDEPLOYQT6_EXECUTABLE)
    find_program(WINDEPLOYQT6_EXECUTABLE NAMES windeployqt6 windeployqt6.exe)
endif()

if(WINDEPLOYQT6_EXECUTABLE)
    message(STATUS "windeployqt6 found: ${WINDEPLOYQT6_EXECUTABLE}")
    add_custom_command(TARGET SGTDetector POST_BUILD
        COMMAND "${WINDEPLOYQT6_EXECUTABLE}"
            --no-translations
            --no-compiler-runtime
            $<$<CONFIG:Debug>:--debug>
            $<$<CONFIG:Release>:--release>
            "$<TARGET_FILE:SGTDetector>"
        COMMENT "Deploying Qt6 plugins via windeployqt6...")
else()
    message(STATUS "windeployqt6 not found; falling back to manual Qt6 platform plugin copy")
    # Helper: copy a file only when the source path exists.
    file(WRITE "${CMAKE_BINARY_DIR}/copy_if_exists.cmake" [=[
if(EXISTS "${SRC}")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${SRC}" "${DST}"
        RESULT_VARIABLE _rc
    )
    if(_rc)
        message(WARNING "Failed to copy ${SRC} -> ${DST} (exit code ${_rc})")
    endif()
else()
    message(STATUS "Optional file not found, skipping: ${SRC}")
endif()
]=])
    get_filename_component(_MINGW_PREFIX "${MINGW_BIN}" DIRECTORY)
    set(QT6_PLATFORMS "${_MINGW_PREFIX}/share/qt6/plugins/platforms")
    add_custom_command(TARGET SGTDetector POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:SGTDetector>/platforms"
        COMMAND ${CMAKE_COMMAND}
            "-DSRC=${QT6_PLATFORMS}/qwindows.dll"
            "-DDST=$<TARGET_FILE_DIR:SGTDetector>/platforms/qwindows.dll"
            -P "${CMAKE_BINARY_DIR}/copy_if_exists.cmake"
        COMMENT "Copying Qt6 platform plugin (if available)...")
endif()
