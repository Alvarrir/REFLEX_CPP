/**
 * @file video_inference_queue.cpp
 * @brief Queue-based video segmentation using YOLOv11 with live overlay
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include "../include/seg/YOLO11Seg.hpp"

// ------------------ Memory Usage Helper ------------------
#ifdef _WIN32
extern "C" {
    __declspec(dllimport) int __stdcall GetProcessMemoryInfo(void* process, void* counters, unsigned int size);
    __declspec(dllimport) void* __stdcall GetCurrentProcess();
}
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

// ------------------ SafeQueue ------------------
template <typename T>
class SafeQueue {
public:
    SafeQueue() : q(), m(), c() {}

    // Add an element to the queue.
    void enqueue(T t) {
        std::lock_guard<std::mutex> lock(m);
        q.push(std::move(t));
        c.notify_one();
    }

    // Get the first element from the queue.
    bool dequeue(T& t) {
        std::unique_lock<std::mutex> lock(m);
        while (q.empty()) {
            if (finished) return false;
            c.wait(lock);
        }
        t = std::move(q.front());
        q.pop();
        return true;
    }

    void setFinished() {
        std::lock_guard<std::mutex> lock(m);
        finished = true;
        c.notify_all();
    }

private:
    std::queue<T> q;
    mutable std::mutex m;
    std::condition_variable c;
    bool finished = false;
};

// ------------------ Stats ------------------
struct Stats {
    int framesProcessed = 0;
    float totalInferenceTimeMS = 0.0f;
    float totalFPS = 0.0f;
    std::vector<size_t> memorySnapshotStorage;

    void update(float inferenceTimeMS, float currentFPS, size_t memoryMB) {
        framesProcessed++;
        totalInferenceTimeMS += inferenceTimeMS;
        totalFPS += currentFPS;
        memorySnapshotStorage.push_back(memoryMB);
    }

    float aveInference() const {
        return framesProcessed > 0 ? totalInferenceTimeMS / framesProcessed : 0.0f;
    }

    float aveFPS() const {
        float aveInf = aveInference();
        return aveInf > 0.0f ? 1000.0f / aveInf : 0.0f;
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

// ------------------ ProcessedFrame ------------------
struct ProcessedFrame {
    int index;
    cv::Mat frame;
    float inferenceTime;
    float fps;
    size_t memoryMB;
    int objectsDetected;
};

// ------------------ Main ------------------
int main() {
    const float CONFIDENCE = 0.5f;
    const float IOU = 0.45f;
    const int LOG_INTERVAL = 50;
    const std::string modelPath = "../../models/Model_6n.onnx";
    const std::string labelsPath = "../../models/classes.names";
    const std::string inputVideoPath = "../../data/fastasfuqboi.mp4";
    const std::string outputVideoPath = "../../data/fastasfuqboi_queue.mp4";
    bool useGPU = false;

    std::cout << "Initializing YOLOv11SegDetector..." << std::endl;
    YOLOv11SegDetector segmentor(modelPath, labelsPath, useGPU);

    cv::VideoCapture cap(inputVideoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video!" << std::endl;
        return -1;
    }

    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));

    cv::VideoWriter writer(outputVideoPath, cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(frameWidth, frameHeight));

    SafeQueue<cv::Mat> frameQueue;
    SafeQueue<ProcessedFrame> processedQueue;
    Stats stats;

    std::atomic<bool> done(false);

    // ---------------- Capture Thread ----------------
    std::thread captureThread([&]() {
        cv::Mat frame;
        int frameCount = 0;
        while (cap.read(frame)) {
            if (frame.empty()) continue;
            frameQueue.enqueue(frame.clone());
            frameCount++;
        }
        frameQueue.setFinished();
        std::cout << "Total frames in video: " << frameCount << std::endl;
    });

    // ---------------- Processing Thread ----------------
    std::thread processingThread([&]() {
        cv::Mat frame;
        int index = 0;
        while (frameQueue.dequeue(frame)) {
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<Segmentation> results = segmentor.segment(frame, CONFIDENCE, IOU);
            auto end = std::chrono::high_resolution_clock::now();

            float inferenceTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0f;
            float currentFPS = 1000.0f / inferenceTime;
            size_t memoryMB = getMemoryUsageMB();
            int objectsDetected = static_cast<int>(results.size());

            stats.update(inferenceTime, currentFPS, memoryMB);
            segmentor.drawSegmentationsAndBoxes(frame, results);

            ProcessedFrame pf{index++, frame, inferenceTime, currentFPS, memoryMB, objectsDetected};
            processedQueue.enqueue(pf);
        }
        processedQueue.setFinished();
    });

    // ---------------- Writing / Display Thread ----------------
    std::thread writingThread([&]() {
        ProcessedFrame pf;
        while (processedQueue.dequeue(pf)) {
            int yOffset = 30;
            int lineHeight = 30;
            char buffer[100];

            snprintf(buffer, sizeof(buffer), "Inference: %.1f ms", pf.inferenceTime);
            cv::putText(pf.frame, buffer, cv::Point(10, yOffset), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,0), 2);
            snprintf(buffer, sizeof(buffer), "FPS: %.1f", pf.fps);
            cv::putText(pf.frame, buffer, cv::Point(10, yOffset+lineHeight), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,0), 2);
            snprintf(buffer, sizeof(buffer), "Objects: %d", pf.objectsDetected);
            cv::putText(pf.frame, buffer, cv::Point(10, yOffset+2*lineHeight), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,0), 2);
            snprintf(buffer, sizeof(buffer), "Memory: %zu MB", pf.memoryMB);
            cv::putText(pf.frame, buffer, cv::Point(10, yOffset+3*lineHeight), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,0), 2);

            cv::imshow("REFLEX Vision - Queue", pf.frame);
            if (cv::waitKey(1) == 'q') break;

            writer.write(pf.frame);
            stats.log(LOG_INTERVAL);
        }
    });

    captureThread.join();
    processingThread.join();
    writingThread.join();

    stats.finalLog();
    cap.release();
    writer.release();
    std::cout << "\nVideo saved to: " << outputVideoPath << std::endl;
    cv::destroyAllWindows();

    return 0;
}
