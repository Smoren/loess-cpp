cmake_minimum_required(VERSION 3.22)
project(loess_cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")

set(EIGEN3_INCLUDE_DIR "libs/eigen-3.4.0")
include_directories(${EIGEN3_INCLUDE_DIR})

set(JSON_INCLUDE_DIR "libs/json-3.11.3")
include_directories(${JSON_INCLUDE_DIR})

set(CATCH2_INCLUDE_DIR "libs/catch2-3.11.0")
include_directories(${CATCH2_INCLUDE_DIR})

set(BIUS_BASE_SOURCES
        src/printer.h
        src/tools.h
        src/tools.cpp
        src/loess.h
        src/loess.cpp
        src/holt_winters.h
        src/holt_winters.cpp
        src/generator.h
        src/generator.cpp
 )

set(BIUS_APP_SOURCES
        src/app.h
        src/app.cpp
)

set(BIUS_TEST_SOURCES
        tests/sample_test.cpp
        tests/tools_test.cpp
        tests/holt_winters_test.cpp
)

add_executable(loess_cpp
        main.cpp

        ${BIUS_BASE_SOURCES}
        ${BIUS_APP_SOURCES}
)

add_executable(loess_tests
        libs/catch2-3.11.0/catch_amalgamated.hpp
        libs/catch2-3.11.0/catch_amalgamated.cpp

        ${BIUS_BASE_SOURCES}
        ${BIUS_APP_SOURCES}
        ${BIUS_TEST_SOURCES}
)

set(CURL_INCLUDE ${CMAKE_SOURCE_DIR}/libs/curl-8.5.0/install/include)
set(CURL_LINK
        ${CMAKE_SOURCE_DIR}/libs/curl-8.5.0/install/lib/libcurl.a
        -lz
)

set(OPENCV_INCLUDE ${CMAKE_SOURCE_DIR}/libs/opencv-4.10.0/install/include/opencv4)
set(OPENCV_LINK
        ${CMAKE_SOURCE_DIR}/libs/opencv-4.10.0/install/lib/libopencv_imgproc.a
        ${CMAKE_SOURCE_DIR}/libs/opencv-4.10.0/install/lib/libopencv_imgcodecs.a
        ${CMAKE_SOURCE_DIR}/libs/opencv-4.10.0/install/lib/libopencv_features2d.a
        ${CMAKE_SOURCE_DIR}/libs/opencv-4.10.0/install/lib/libopencv_calib3d.a
        ${CMAKE_SOURCE_DIR}/libs/opencv-4.10.0/install/lib/libopencv_core.a -lz
)
