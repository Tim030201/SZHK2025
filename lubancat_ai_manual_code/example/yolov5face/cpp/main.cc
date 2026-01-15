// Licensed under the Apache License, Version 2.0 (the "License");
// ... (保留原有版权声明)

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h> // 用于精确计时

#include "yolov5face.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include "easy_timer.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <queue>
#include <deque> // 【关键】引入双端队列用于 FPS 计算
#include <mutex>
#include <thread>
#include <condition_variable>

// -------------------------------------------
// [RGA 区域] 
// 如果您的环境配置好了 librga，请取消注释并使用 imcrop
// #include <rga/im2d.h>
// #include <rga/rga.h>
// -------------------------------------------

using namespace cv;
using namespace std;


// [问题3] 线程安全的队列
// 加了 mutex 锁，保证 push 和 pop 不会同时发生。
// 不会出现两个 NPU 取走同一张图的情况。
template <typename T>
class SafeQueue {
public:
    void push(const T& item) {
        unique_lock<mutex> lock(mtx);
        q.push(item);
        cond.notify_one();
    }

    bool try_pop(T& item) {
        unique_lock<mutex> lock(mtx);
        if (q.empty()) return false;
        item = q.front();
        q.pop();
        return true;
    }
    
    void wait_and_pop(T& item) {
        unique_lock<mutex> lock(mtx);
        cond.wait(lock, [this]{ return !q.empty(); });
        item = q.front();
        q.pop();
    }

    int size() {
        unique_lock<mutex> lock(mtx);
        return q.size();
    }

private:
    queue<T> q;
    mutex mtx;
    condition_variable cond;
};

// 数据结构
struct InputData {
    int id;
    Mat img;
};

struct OutputData {
    int id;
    yolov5face_result_list results;
};

// 全局队列
SafeQueue<InputData> input_queue;
SafeQueue<OutputData> output_queue;

// [问题 4] 全局 NPU 互斥锁
std::mutex g_npu_mutex; 
// [问题 3] NPU 工作线程的线程安全
void npu_worker(const char* model_path, int thread_id) {
    int ret;
    rknn_app_context_t rknn_ctx;
    memset(&rknn_ctx, 0, sizeof(rknn_app_context_t));

    {
        std::lock_guard<std::mutex> lock(g_npu_mutex);
        printf("[Thread %d] Initializing model...\n", thread_id);
        ret = init_yolov5face_model(model_path, &rknn_ctx);
    }
    
    if (ret != 0) {
        printf("[Thread %d] Init model failed!\n", thread_id);
        return;
    }

    InputData in_data;
    image_buffer_t src_image;
    OutputData out_data;

    while (true) {
        // [问题 3] 线程安全获取任务
        input_queue.wait_and_pop(in_data);

        // 退出信号
        if (in_data.img.empty()) break;

        // 准备数据
        memset(&src_image, 0, sizeof(image_buffer_t));
        src_image.width = in_data.img.cols;
        src_image.height = in_data.img.rows;
        src_image.format = IMAGE_FORMAT_RGB888;
        src_image.virt_addr = in_data.img.data;

        // 推理
        yolov5face_result_list od_results;
        {
            std::lock_guard<std::mutex> lock(g_npu_mutex); 
            memset(&od_results, 0, sizeof(yolov5face_result_list));
            ret = inference_yolov5face_model(&rknn_ctx, &src_image, &od_results);
        }
        
        
        // 发送结果
        if (ret == 0) {
            out_data.id = in_data.id;
            out_data.results = od_results;
            output_queue.push(out_data);
        }
    }
    release_yolov5face_model(&rknn_ctx);
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    // [问题 4] 如果程序崩溃 (Core Dump)，在终端运行：
    // ulimit -c unlimited
    // ./程序名
    // gdb ./程序名 core
    // 然后输入 bt 查看错误位置

    //方法2(直接在gdb中运行调试)
    //gdb ./demo
    //set args ./模型
    //run
    
    if (argc < 2) {
        printf("Usage: %s <model_path> [video_device]\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *video_dev = (argc >= 3) ? argv[2] : "/dev/video11"; 

    // ---------------------------------------------
    // 1. 启动两个 NPU 线程 (双核并行)
    // ---------------------------------------------

    std::thread t1(npu_worker, model_path, 1);
    std::thread t2(npu_worker, model_path, 2);

    // ---------------------------------------------
    // 2. 主摄像头循环
    // ---------------------------------------------

    VideoCapture cap;
    if (isdigit(video_dev[0]) && strlen(video_dev) < 3) cap.open(atoi(video_dev)); 
    else cap.open(video_dev);       

    if (!cap.isOpened()) {
        cout << "Camera open failed!" << endl;
        // 退出，防止死锁
        input_queue.push({0, Mat()}); input_queue.push({0, Mat()});
        t1.join(); t2.join();
        return -1;
    }
    
    // [问题 1] 设置分辨率
    // 优先使用硬件缩放 (ISP)cap.set，比软件 resize 快，比 RGA imcrop 方便
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);

    namedWindow("RK3576 Dual NPU", WINDOW_NORMAL);

    Mat frame, rgb_frame;
    int frame_id = 0;
    yolov5face_result_list last_results = {0};

    // [问题 5] FPS 滑动平均计算专用变量
    std::deque<double> frame_times;
    int fps_window_size = 30; // 统计窗口：过去 30 帧
    double avg_fps = 0.0;

    printf("--> Main Loop Start.\n");

    while (true)
    {
        // 1. 抓图 (这行代码的耗时波动最大，是 FPS 计算的关键)
        cap >> frame;
        if (frame.empty()) break;

        // [问题 5] 记录这一帧到达的时间点 (秒)
        double now = (double)getTickCount() / getTickFrequency();
        frame_times.push_back(now);

        // 维护队列长度，只留最近 30 个时间点
        if (frame_times.size() > fps_window_size) {
            frame_times.pop_front();
        }
        
        // 2. 预处理 (BGR -> RGB)
        // [RGA 区域] 如果用 imcrop，在这里替代 cvtColor
        cvtColor(frame, rgb_frame, COLOR_BGR2RGB);

        // [问题 2 & 6] Clone 与 内存拷贝
        // 加时间戳验证 clone 是否耗时
        double t_clone_start = (double)getTickCount();
        
        Mat thread_img = rgb_frame.clone(); // 深拷贝，分配新内存
        
        double t_clone_end = (double)getTickCount();
        double clone_cost_ms = ((t_clone_end - t_clone_start) / getTickFrequency()) * 1000.0;
        
        // 每 60 帧打印一次 Clone 耗时，证明发生了内存拷贝
        if (frame_id % 60 == 0) {
            printf("[Debug] Frame %d: Clone cost = %.3f ms (Memory Copy Confirmed)\n", frame_id, clone_cost_ms);
        }
        
        // 3. 将任务派发给 NPU (放入队列)
        // 为了防止队列堆积导致延迟太大，限制队列长度
        // 丢帧策略:如果堆积超过 2 帧，就只更新最新的，丢掉旧的，保证实时性
        if (input_queue.size() < 2) {
            InputData data;
            data.id = frame_id++;
            data.img = thread_img;
            input_queue.push(data);
        }

        OutputData out_data; 
        if (output_queue.try_pop(out_data)) {
            // 有新结果，更新
            last_results = out_data.results;
        }

        for (int i = 0; i < last_results.count; i++) {
            object_detect_result *det_result = &(last_results.results[i]);
            rectangle(frame, Point(det_result->box.left, det_result->box.top), 
                      Point(det_result->box.right, det_result->box.bottom), Scalar(255, 0, 0), 2);
        }

        // [问题 5] 计算并显示 平均 FPS
        // 公式：(帧数 - 1) / (最后一帧时间 - 第一帧时间)
        if (frame_times.size() > 1) {
            double duration = frame_times.back() - frame_times.front();
            if (duration > 0) {
                avg_fps = (frame_times.size() - 1) / duration;
            }
        }

        char fps_text[32];
        sprintf(fps_text, "Real FPS: %.2f", avg_fps);
        putText(frame, fps_text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);

        imshow("RK3576 Dual NPU", frame);
        if (waitKey(1) == 'q') break;
    }

    input_queue.push({0, Mat()}); 
    input_queue.push({0, Mat()});
    t1.join();
    t2.join();
    
    return 0;
}
