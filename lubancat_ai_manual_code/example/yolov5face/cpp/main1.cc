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
#include <mutex>
#include <thread>
#include <condition_variable>
#include <deque> // 用于FPS滑动平均计算

// -------------------------------------------
// [RGA 相关头文件] 
// 如果您的环境包含 RGA，请取消注释
// #include <rga/im2d.h>
// #include <rga/rga.h>
// -------------------------------------------

using namespace cv;
using namespace std;

// ---------------------------------------------------------
// 线程安全队列 (已验证：安全)
// ---------------------------------------------------------
template <typename T>
class SafeQueue {
public:
    void push(const T& item) {
        unique_lock<mutex> lock(mtx); // 加锁：进门关门
        q.push(item);
        cond.notify_one(); // 唤醒一个在等待的线程
    }

    bool try_pop(T& item) {
        unique_lock<mutex> lock(mtx); // 加锁
        if (q.empty()) return false;
        item = q.front();
        q.pop();
        return true;
    }
    
    // 阻塞等待：如果没有数据，线程会在这里睡觉，不占 CPU
    void wait_and_pop(T& item) {
        unique_lock<mutex> lock(mtx);
        // 如果队列空，就睡觉；直到有数据被 notify 唤醒
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

// ---------------------------------------------------------
// NPU 工作线程
// ---------------------------------------------------------
void npu_worker(const char* model_path, int thread_id) {
    int ret;
    rknn_app_context_t rknn_ctx;
    memset(&rknn_ctx, 0, sizeof(rknn_app_context_t));

    ret = init_yolov5face_model(model_path, &rknn_ctx);
    if (ret != 0) {
        printf("[Thread %d] Init fail!\n", thread_id);
        return;
    }

    InputData in_data;
    image_buffer_t src_image;
    OutputData out_data;

    while (true) {
        // 1. 等待取图 (线程安全，自动挂起，不会死锁，不会忙等)
        input_queue.wait_and_pop(in_data);

        // 退出信号
        if (in_data.img.empty()) break;

        // 2. 准备数据
        memset(&src_image, 0, sizeof(image_buffer_t));
        src_image.width = in_data.img.cols;
        src_image.height = in_data.img.rows;
        src_image.format = IMAGE_FORMAT_RGB888;
        src_image.virt_addr = in_data.img.data;

        // 3. 推理
        yolov5face_result_list od_results;
        ret = inference_yolov5face_model(&rknn_ctx, &src_image, &od_results);
        
        // 4. 发送结果 (线程安全)
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
    if (argc < 2) {
        printf("Usage: %s <model_path> [video_device]\n", argv[0]);
        return -1;
    }
    const char *model_path = argv[1];
    const char *video_dev = (argc >= 3) ? argv[2] : "/dev/video11"; 

    // 启动双线程
    std::thread t1(npu_worker, model_path, 1);
    std::thread t2(npu_worker, model_path, 2);

    VideoCapture cap;
    if (isdigit(video_dev[0]) && strlen(video_dev) < 3) cap.open(atoi(video_dev)); 
    else cap.open(video_dev);       

    if (!cap.isOpened()) {
        cout << "Camera open failed!" << endl;
        // 发送退出信号给线程，防止主程序退出导致 core dump
        input_queue.push({0, Mat()}); input_queue.push({0, Mat()});
        t1.join(); t2.join();
        return -1;
    }
    
    // 【Q1: 分辨率设置】
    // 尽量让硬件吐出接近模型输入的尺寸 (640x480)
    // 这样比读 1080P 再用 imcrop 还要快，因为传输带宽小
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);

    namedWindow("RK3576", WINDOW_NORMAL);

    Mat frame, rgb_frame;
    int frame_id = 0;
    yolov5face_result_list last_results = {0};

    // 【Q5: FPS 滑动平均计算变量】
    std::deque<double> frame_times;
    int fps_window_size = 30; // 统计过去30帧的平均值
    double avg_fps = 0.0;

    printf("--> Loop Start.\n");

    while (true)
    {
        cap >> frame;
        if (frame.empty()) break;

        // [RGA 优化思路]
        // 如果您的输入是 1080P，想切成 640x640，应该在这里调用 imcrop
        // rga_buffer_t src = wrapbuffer_virtualaddr(frame.data, 1920, 1080, RK_FORMAT_BGR_888);
        // rga_buffer_t dst = wrapbuffer_virtualaddr(crop_buf, 640, 640, RK_FORMAT_RGB_888);
        // imcrop(src, dst, rect);
        // 由于这里我无法为您链接 RGA 库，暂用 OpenCV 实现格式转换
        
        cvtColor(frame, rgb_frame, COLOR_BGR2RGB);

        // 【Q2 & Q6: Clone 耗时测试】
        // 这里的 clone 是必须的，为了防止多线程内存冲突
        double t_clone_start = (double)getTickCount();
        
        Mat thread_img = rgb_frame.clone(); 
        
        double t_clone_end = (double)getTickCount();
        double clone_cost_ms = ((t_clone_end - t_clone_start) / getTickFrequency()) * 1000.0;
        
        // 如果您想看 clone 花了多久，取消下面这行的注释
        printf("[Time Debug] Clone cost: %.3f ms\n", clone_cost_ms);

        // 丢帧策略：如果积压太多，就不发了
        if (input_queue.size() < 2) {
            InputData data;
            data.id = frame_id++;
            data.img = thread_img;
            input_queue.push(data);
        }

        // 获取结果
        OutputData out_data;
        if (output_queue.try_pop(out_data)) {
            last_results = out_data.results;
        }

        // 绘制
        for (int i = 0; i < last_results.count; i++) {
            object_detect_result *det_result = &(last_results.results[i]);
            rectangle(frame, Point(det_result->box.left, det_result->box.top), 
                      Point(det_result->box.right, det_result->box.bottom), Scalar(255, 0, 0), 2);
        }

        // 【Q5: 科学的 FPS 计算】
        double now = (double)getTickCount() / getTickFrequency();
        frame_times.push_back(now);
        // 保持队列只有最近30帧的时间戳
        if (frame_times.size() > fps_window_size) {
            frame_times.pop_front();
        }
        // 只有攒够几帧才开始算
        if (frame_times.size() > 1) {
            double duration = frame_times.back() - frame_times.front();
            if (duration > 0) {
                // 帧数 / 时间 = FPS
                avg_fps = (frame_times.size() - 1) / duration;
            }
        }

        char fps_text[32];
        sprintf(fps_text, "FPS: %.2f", avg_fps);
        putText(frame, fps_text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);

        imshow("RK3576", frame);
        if (waitKey(1) == 'q') break;
    }

    // 退出
    input_queue.push({0, Mat()}); 
    input_queue.push({0, Mat()});
    t1.join();
    t2.join();
    return 0;
}
