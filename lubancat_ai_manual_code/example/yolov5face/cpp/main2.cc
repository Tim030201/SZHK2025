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

#include "yolov5face.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include "easy_timer.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>

using namespace cv;
using namespace std;

// ---------------------------------------------------------
// [新增] 线程安全的队列，用于在主线程和NPU线程之间传递数据
// ---------------------------------------------------------
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
    
    // 阻塞等待直到有数据
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

// 定义传递给线程的数据结构
struct InputData {
    int id;          // 帧ID，用于排序（虽然本demo简化处理，不强求顺序）
    Mat img;         // OpenCV图片
};

struct OutputData {
    int id;
    yolov5face_result_list results; // 检测结果
};

// 全局队列
SafeQueue<InputData> input_queue;
SafeQueue<OutputData> output_queue;

// ---------------------------------------------------------
// [新增] NPU 工作线程函数
// ---------------------------------------------------------
void npu_worker(const char* model_path, int thread_id) {
    int ret;
    rknn_app_context_t rknn_ctx;
    memset(&rknn_ctx, 0, sizeof(rknn_app_context_t));

    // 每个线程初始化一个独立的 RKNN 上下文
    // 系统会自动分配 NPU 核心，或者多核心协作
    ret = init_yolov5face_model(model_path, &rknn_ctx);
    if (ret != 0) {
        printf("[Thread %d] Init model failed!\n", thread_id);
        return;
    }
    printf("[Thread %d] Model init success. Ready to work!\n", thread_id);

    InputData in_data;
    image_buffer_t src_image;
    OutputData out_data;

    while (true) {
        // 1. 等待取图
        input_queue.wait_and_pop(in_data);//?是否线程安全?否 在前加锁后解锁;wait是否忙等,让出cpu
        //?无数据sleep1ms,让cpu;不要出现死锁

        // 如果收到空图，说明要退出了
        if (in_data.img.empty()) break;

        // 2. 准备数据
        // 注意：这里需要重新封装 buffer，因为是在不同线程
        memset(&src_image, 0, sizeof(image_buffer_t));
        src_image.width = in_data.img.cols;
        src_image.height = in_data.img.rows;
        src_image.format = IMAGE_FORMAT_RGB888;
        src_image.virt_addr = in_data.img.data;

        // 3. 执行推理 (NPU干活)
        // 这里不需要锁，因为 rknn_ctx 是线程独有的
        yolov5face_result_list od_results;
        ret = inference_yolov5face_model(&rknn_ctx, &src_image, &od_results);
        
        // 4. 发送结果
        if (ret == 0) {
            out_data.id = in_data.id;
            out_data.results = od_results; // 这里的结构体复制是安全的
            output_queue.push(out_data);//?是否线程安全?否 在前加锁后解锁
        }
    }

    // 释放资源
    release_yolov5face_model(&rknn_ctx);
    printf("[Thread %d] Exit.\n", thread_id);
}


/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
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
        return -1;
    }
    
    // 设置分辨率 (保持速度平衡),调整图片尺寸,提高FPS
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    //?改crop

    namedWindow("RK3576 Dual NPU", WINDOW_NORMAL);

    Mat frame, rgb_frame;
    double t_start, t_end, fps;
    char fps_text[32];
    int frame_id = 0;

    // 保存最近一次的检测结果，如果没有新结果就画旧的
    yolov5face_result_list last_results = {0};

    printf("--> Main Loop Start. Using 2 NPU Threads.\n");

    while (true)
    {
        t_start = (double)getTickCount();

        // 1. 抓图
        cap >> frame;
        if (frame.empty()) break;

        // 2. 预处理 (BGR -> RGB)
        // 必须 clone() 一份，因为 frame 后续会被摄像头覆盖，而线程需要独立的数据
        cvtColor(frame, rgb_frame, COLOR_BGR2RGB);

        // ?加时间戳
        Mat thread_img = rgb_frame.clone(); 

        // 3. 将任务派发给 NPU (放入队列)
        // 为了防止队列堆积导致延迟太大，限制队列长度
        // 丢帧策略:如果堆积超过 2 帧，就只更新最新的，丢掉旧的，保证实时性
        if (input_queue.size() < 2) {
            InputData data;
            data.id = frame_id++;
            data.img = thread_img;
            input_queue.push(data);
        }

        // 4. 尝试获取结果 (非阻塞，有就拿，没有就用旧的)
        OutputData out_data;
        if (output_queue.try_pop(out_data)) {
            // 有新结果，更新
            last_results = out_data.results;
        }

        // 5. 绘制 (使用 last_results)
        for (int i = 0; i < last_results.count; i++) {
            object_detect_result *det_result = &(last_results.results[i]);
            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;
            rectangle(frame, Point(x1, y1), Point(x2, y2), Scalar(255, 0, 0), 2);
            
            // 简单画个分
            char text[32];
            sprintf(text, "%.1f%%", det_result->prop * 100);
            putText(frame, text, Point(x1, y1 - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
            
            // 画关键点
            for(int j=0; j<5; j++)
                circle(frame, Point(det_result->ponit[j].x, det_result->ponit[j].y), 2, Scalar(0, 165, 255), -1);
        }

        // 6. 计算 FPS
        t_end = (double)getTickCount();
        fps = 1.0 / ((t_end - t_start) / getTickFrequency()); //?确认计算方式
        sprintf(fps_text, "Dual NPU FPS: %.2f", fps);
        putText(frame, fps_text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);

        imshow("RK3576 Dual NPU", frame);

        if (waitKey(1) == 'q') break;
    }

    // 退出处理
    // 发送空图片通知线程退出
    input_queue.push({0, Mat()}); 
    input_queue.push({0, Mat()});
    
    t1.join();
    t2.join();
    
    return 0;
}



/*
1.用im2d去处理图片裁切问题,读操作文档中的crop,比cap.set速度快;cap.set作为软件操作比2dgpu慢,涉及到cpu内存的memory copy

2.在clone前后加时间戳,看看clone是否有用到memory copy

3.查清楚图像处理后显示帧率为什么能跑到30帧以上,是否同一帧的图片被两个npu同时处理了,导致帧率高于摄像头的30帧；
npu_worker中的input_queue和output_queue是否线程安全，可能会导致两个npu同时取走同一张图片，否需要在前后加锁解锁

4.板卡运行是出现自动终止的情况,可能出现了越界访问、空指针;解决方法:转存core文件并用core文件debug 用gdb调

5.fps的计算方式是否正确，是否应该用滑动平均的方式计算fps，而不是每一帧单独计算fps，是否因为计算方式错误导致帧率高于30帧

6.memory、clone相关问题

*/