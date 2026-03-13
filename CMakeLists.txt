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
            "${CMAKE_SOURCE_DIR}/${_asset}"
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
# Copy MinGW runtime + Qt6 DLLs
# OpenCV 4.13 HighGUI is compiled against Qt6; these DLLs must ship with the exe.
# ─────────────────────────────────────────────────────────────────────────────
set(MINGW_BIN "C:/msys64/mingw64/bin")

# MinGW C++ runtime
foreach(_dll libgcc_s_seh-1.dll libstdc++-6.dll libwinpthread-1.dll)
    add_custom_command(TARGET SGTDetector POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MINGW_BIN}/${_dll}"
            "$<TARGET_FILE_DIR:SGTDetector>/${_dll}"
        COMMENT "Copying ${_dll}...")
endforeach()

# Qt6 DLLs pulled in by libopencv_highgui
foreach(_dll Qt6Core.dll Qt6Gui.dll Qt6Widgets.dll Qt6Test.dll Qt6OpenGLWidgets.dll)
    add_custom_command(TARGET SGTDetector POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MINGW_BIN}/${_dll}"
            "$<TARGET_FILE_DIR:SGTDetector>/${_dll}"
        COMMENT "Copying ${_dll}...")
endforeach()

# Qt6 Windows platform plugin (needed for any GUI window)
set(QT6_PLATFORMS "C:/msys64/mingw64/share/qt6/plugins/platforms")
add_custom_command(TARGET SGTDetector POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_FILE_DIR:SGTDetector>/platforms"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${QT6_PLATFORMS}/qwindows.dll"
        "$<TARGET_FILE_DIR:SGTDetector>/platforms/qwindows.dll"
    COMMENT "Copying Qt6 platform plugin...")
