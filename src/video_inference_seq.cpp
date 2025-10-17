/**
 * @file video_inference_seq.cpp
 * @brief Sequential video segmentation using YOLOv11 with live stats and total frame count
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>

#include "../include/seg/YOLO11Seg.hpp"

#ifdef _WIN32
extern "C" {
    __declspec(dllimport) int __stdcall GetProcessMemoryInfo(void* process, void* counters, unsigned int size);
    __declspec(dllimport) void* __stdcall GetCurrentProcess();
}
#else
#include <unistd.h>
#include <stdio.h>
#endif

size_t getMemoryUsageMB() {
#ifdef _WIN32
    struct PROCESS_MEMORY_COUNTERS {
        unsigned long cb;
        unsigned long PageFaultCount;
        size_t PeakWorkingSetSize;
        size_t WorkingSetSize;
        size_t QuotaPeakPagedPoolUsage;
        size_t QuotaPagedPoolUsage;
        size_t QuotaPeakNonPagedPoolUsage;
        size_t QuotaNonPagedPoolUsage;
        size_t PagefileUsage;
        size_t PeakPagefileUsage;
    };
    PROCESS_MEMORY_COUNTERS pmc;
    pmc.cb = sizeof(PROCESS_MEMORY_COUNTERS);
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize / (1024 * 1024);
    }
#else
    FILE* file = fopen("/proc/self/statm", "r");
    if (file) {
        size_t size, resident, share, text, lib, data, dt;
        if (fscanf(file, "%zu %zu %zu %zu %zu %zu %zu",
                   &size, &resident, &share, &text, &lib, &data, &dt) == 7) {
            fclose(file);
            return resident * sysconf(_SC_PAGESIZE) / (1024 * 1024);
        }
        fclose(file);
    }
#endif
    return 0;
}

struct Stats {
    int framesProcessed = 0;
    float totalInferenceTimeMS = 0.0f;
    float totalFPS = 0.0f;
    std::vector<size_t> memorySnapshotStorage;

    void update(float inferenceTimeMS, float currentFPS, size_t memorySnapshotMB) {
        framesProcessed++;
        totalInferenceTimeMS += inferenceTimeMS;
        totalFPS += currentFPS;
        memorySnapshotStorage.push_back(memorySnapshotMB);
    }

    float aveInference() const {
        return framesProcessed > 0 ? totalInferenceTimeMS / framesProcessed : 0.0f;
    }

    float aveFPS() const {
        float aveInf = aveInference();
        if (aveInf > 0.0f) return 1000.0f / aveInf;
        else return 0.0f;
    }

    size_t aveMemory() const {
        if (memorySnapshotStorage.empty()) return 0;
        return std::accumulate(memorySnapshotStorage.begin(), memorySnapshotStorage.end(), size_t(0)) / memorySnapshotStorage.size();
    }

    void log(int interval) const {
        if (framesProcessed % interval == 0) {
            std::cout << "Processed " << framesProcessed
                      << " | Average Inference: " << aveInference() << " ms"
                      << " | Average FPS: " << aveFPS()
                      << " | Memory: " << aveMemory() << " MB" << std::endl;
        }
    }

    void finalLog() const {
        std::cout << "\n<<<FINAL STATISTICS>>>" << std::endl;
        std::cout << "Avg Inference: " << aveInference() << " ms" << std::endl;
        std::cout << "Avg FPS: " << aveFPS() << std::endl;
        std::cout << "Avg Memory: " << aveMemory() << " MB" << std::endl;
    }
};

int main() {
    const float CONFIDENCE = 0.5f;
    const float IOU = 0.45f;
    const int LOG_INTERVAL = 50;
    const std::string modelPath = "../../models/Model_6n.onnx";
    const std::string labelsPath = "../../models/classes.names";
    const std::string inputVideoPath = "../../data/fastasfuqboi.mp4";
    const std::string outputVideoPath = "../../data/fastasfuqboi_seq.mp4";
    bool useGPU = false;

    std::cout << "Checking YOLOs-CPP required files..." << std::endl;

    YOLOv11SegDetector segmentor(modelPath, labelsPath, useGPU);
    std::cout << "Files found!" << std::endl;

    cv::VideoCapture cap(inputVideoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video!" << std::endl;
        return -1;
    }

    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    std::cout << "Total frames in video: " << totalFrames << std::endl;

    cv::VideoWriter writer(outputVideoPath, cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(frameWidth, frameHeight));

    cv::Mat frame;
    Stats stats;

    while (cap.read(frame)) {
        if (frame.empty()) continue;

        auto startTime = std::chrono::high_resolution_clock::now();
        std::vector<Segmentation> results = segmentor.segment(frame, CONFIDENCE, IOU);
        auto endTime = std::chrono::high_resolution_clock::now();

        float inferenceTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000.0f;
        float currentFPS = 1000.0f / inferenceTime;
        size_t currentMemory = getMemoryUsageMB();

        stats.update(inferenceTime, currentFPS, currentMemory);
        segmentor.drawSegmentationsAndBoxes(frame, results);

        // Overlay stats
        char inferenceText[50], fpsText[50], memoryText[50], objectsText[50];
        snprintf(inferenceText, sizeof(inferenceText), "Inference: %.1f ms", inferenceTime);
        snprintf(fpsText, sizeof(fpsText), "FPS: %.1f", currentFPS);
        snprintf(memoryText, sizeof(memoryText), "Memory: %zu MB", currentMemory);
        snprintf(objectsText, sizeof(objectsText), "Objects: %zu", results.size());

        int yOffset = 30;
        int lineHeight = 30;

        cv::putText(frame, inferenceText, cv::Point(10, yOffset), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,0), 2);
        cv::putText(frame, fpsText, cv::Point(10, yOffset + lineHeight), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,0), 2);
        cv::putText(frame, objectsText, cv::Point(10, yOffset + 2*lineHeight), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,0), 2);
        cv::putText(frame, memoryText, cv::Point(10, yOffset + 3*lineHeight), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,0), 2);

        cv::imshow("REFLEX Vision - Seq", frame);
        if (cv::waitKey(1) == 'q') break;

        writer.write(frame);
        stats.log(LOG_INTERVAL);
    }

    stats.finalLog();
    cap.release();
    writer.release();
    std::cout << "\nVideo saved to: " << outputVideoPath << std::endl;
    cv::destroyAllWindows();
    return 0;
}
