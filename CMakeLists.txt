cmake_minimum_required(VERSION 3.5)
project(yolo_ort)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Set default ONNXRUNTIME_DIR if not provided
if(NOT DEFINED ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/onnxruntime")
endif()
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

# Set OpenCV path
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/opencv/build")
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV found: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

# Debug: Print all OpenCV libraries found
message(STATUS "All OpenCV libraries: ${OpenCV_LIBS}")

# Set C++ standard globally
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Executables
add_executable(camera_inference src/camera_inference.cpp)
add_executable(video_inference_queue src/video_inference_queue.cpp)
add_executable(video_inference_seq src/video_inference_seq.cpp)

# Common include directories for all targets
target_include_directories(camera_inference PRIVATE 
    "${ONNXRUNTIME_DIR}/include"
    ${OpenCV_INCLUDE_DIRS}
    "include/"
)
target_include_directories(video_inference_queue PRIVATE 
    "${ONNXRUNTIME_DIR}/include"
    ${OpenCV_INCLUDE_DIRS}
    "include/"
)
target_include_directories(video_inference_seq PRIVATE 
    "${ONNXRUNTIME_DIR}/include"
    ${OpenCV_INCLUDE_DIRS}
    "include/"
)

# Set C++ standard for each target
target_compile_features(camera_inference PRIVATE cxx_std_17)
target_compile_features(video_inference_queue PRIVATE cxx_std_17)
target_compile_features(video_inference_seq PRIVATE cxx_std_17)

# Common link libraries
target_link_libraries(camera_inference ${OpenCV_LIBS})
target_link_libraries(video_inference_queue ${OpenCV_LIBS})
target_link_libraries(video_inference_seq ${OpenCV_LIBS})

# Platform-specific ONNX Runtime linking
if(UNIX AND NOT APPLE)  # Linux
    target_link_libraries(camera_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
    target_link_libraries(video_inference_queue "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
    target_link_libraries(video_inference_seq "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
elseif(APPLE)  # macOS
    target_link_libraries(camera_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
    target_link_libraries(video_inference_queue "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
    target_link_libraries(video_inference_seq "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
elseif(WIN32)  # Windows
    target_link_libraries(camera_inference 
        "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib" 
        Psapi.lib
    )
    target_link_libraries(video_inference_queue 
        "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib" 
        Psapi.lib
    )
    target_link_libraries(video_inference_seq 
        "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib" 
        Psapi.lib
    )

    # Copy appropriate OpenCV DLLs based on build type
    get_filename_component(OPENCV_DLL_DIR "${OpenCV_DIR}/x64/vc16/bin" ABSOLUTE)
    message(STATUS "OpenCV DLL directory: ${OPENCV_DLL_DIR}")
    
    foreach(target camera_inference video_inference_queue video_inference_seq)
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            # Copy Release DLLs
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${OPENCV_DLL_DIR}/opencv_world4120.dll"
                "${OPENCV_DLL_DIR}/opencv_videoio_ffmpeg4120_64.dll"
                $<TARGET_FILE_DIR:${target}>
                COMMENT "Copying OpenCV Release DLLs for ${target}"
            )
        else()
            # Copy Debug DLLs  
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${OPENCV_DLL_DIR}/opencv_world4120d.dll"
                "${OPENCV_DLL_DIR}/opencv_videoio_ffmpeg4120_64.dll"
                $<TARGET_FILE_DIR:${target}>
                COMMENT "Copying OpenCV Debug DLLs for ${target}"
            )
        endif()
        
        # Also copy ONNX Runtime DLL
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
            $<TARGET_FILE_DIR:${target}>
            COMMENT "Copying ONNX Runtime DLL for ${target}"
        )
    endforeach()
endif()

# Add compiler definitions for Windows
if(WIN32)
    target_compile_definitions(camera_inference PRIVATE _WIN32_WINNT=0x0601)
    target_compile_definitions(video_inference_queue PRIVATE _WIN32_WINNT=0x0601)
    target_compile_definitions(video_inference_seq PRIVATE _WIN32_WINNT=0x0601)
endif()