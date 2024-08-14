#include<opencv2/opencv.hpp>
#include<iostream>
#include <cmath>
#include<string>
//#include "matplotlibcpp.h"
#include <filesystem>
#include <algorithm> // 包含 std::max
#include "putText.h"
#pragma execution_character_set("utf-8")
namespace fs = std::filesystem;
using namespace cv;
using namespace std;
#define M_PI 3.1415926

//函数声明
void calculationQualified();
bool compareContourAreas(vector<Point> contour1, vector<Point> contour2);

//合格点计算
void calculationQualified() {
    


}
//result 存储了圆心在图中的坐标
//maskcenter 圆质心截取后的坐标，
//maskrectangle 存储了圆的外接矩形的宽和高
//point_result 存储最终的结果坐标

void processImage(Mat& frame, Point2f result, Point2f& maskcenter,vector<int> maskrectangle,double a,double f,Point& point_result) {
    // 显示原始BGR图像
    cv::imshow("BGR Image", frame);

    // 转换到HSV图像
    cv::Mat hsvImage, framegray;
    cv::cvtColor(frame, hsvImage, cv::COLOR_BGR2HSV);
    cv::cvtColor(frame, framegray, cv::COLOR_BGR2GRAY);

    //对饱和度图进行高斯模糊
    cv::Mat blurredSaturation;
    cv::GaussianBlur(framegray, blurredSaturation, cv::Size(7, 7), 0);

    // 对灰度图像进行二值化操作
    cv::Mat binaryImage;
    double thresh = 100;   // 阈值
    double maxValue = 255; // 最大值

    cv::threshold(blurredSaturation, binaryImage, thresh, maxValue, cv::THRESH_BINARY_INV);
    //膨胀腐蚀
    Mat morphologyImage;
    Mat kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21), Point(-1, -1));
    Mat kerne2 = getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15), Point(-1, -1));
    morphologyEx(binaryImage, morphologyImage, MORPH_CLOSE, kernel, Point(-1, -1), 1);
    morphologyEx(morphologyImage, morphologyImage, MORPH_CLOSE, kernel, Point(-1, -1), 1);
    
    // 检测所有轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binaryImage, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 轮廓为空
    if (contours.empty()) {
        cerr << "Error: No contours found." << endl;
        return;
    }
    // 筛选轮廓：只保留长度大于100的轮廓
    std::vector<std::vector<cv::Point>> filteredContours;
    double minLength = 100.0;
    double circle_D = min(maskrectangle[0], maskrectangle[1]);
    for (const auto& contour : contours) {
        // 寻找轮廓的矩形边界框
        Rect rect = boundingRect(contour);
        if (max(rect.width, rect.height) > (circle_D + 100) || max(rect.width, rect.height) < (circle_D * 0.75)) {
            continue;

        }
        cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::drawContours(mask, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255), cv::FILLED);

        cv::Mat region;
        frame.copyTo(region, mask);
        // 进行裁剪，形成新的Mat图像
        cv::Mat croppedregion = region(rect);
        // 转换到HSV图像
        cv::Mat croppedregionhsv, framegray;
        cv::cvtColor(croppedregion, croppedregionhsv, cv::COLOR_BGR2HSV);
        cv::Mat redMask;
        cv::Scalar lower_red(156, 43, 46); // 红色的下限（HSV）
        cv::Scalar upper_red(180, 255, 255); // 红色的上限（HSV）

        // 创建掩膜图像
        cv::inRange(croppedregionhsv, lower_red, upper_red, redMask); // 查找第一个红色范围

        if (cv::countNonZero(redMask) == 0) {
            continue;
        }
        /*imshow("croppedregion", croppedregion);
        waitKey(0);*/
        
        string finalposition;
        cv::rectangle(frame, rect, cv::Scalar(0,255,0), 2);
        double distance1 = 0, distance2 = 0, distance3 = 0, distance4 = 0;
        Point2f point1 = Point2f(rect.x + maskcenter.x,rect.y + maskcenter.y);
        Point2f point2 = Point2f(rect.x + rect.width - maskrectangle[0] + maskcenter.x, rect.y + maskcenter.y);
        Point2f point3 = Point2f(rect.x + maskcenter.x, rect.y + rect.height - maskrectangle[1] + maskcenter.y);
        Point2f point4 = Point2f(rect.x + rect.width - maskrectangle[0] + maskcenter.x, rect.y + rect.height - maskrectangle[1] + maskcenter.y);
        if (redMask.at<uchar>(point1.y - rect.y, point1.x - rect.x) == 255) {
            distance1 = sqrt(atan((rect.x - result.x) * a / f) * atan((rect.x - result.x) * a / f) + atan((result.y - rect.y) * a / f) * atan((result.y - rect.y) * a / f));
         }
        if (redMask.at<uchar>(point2.y - rect.y, point2.x - rect.x) == 255) {
            distance2 = sqrt(atan((rect.x - result.x) * a / f) * atan((rect.x - result.x) * a / f) + atan((result.y - rect.y) * a / f) * atan((result.y - rect.y) * a / f));
        }
        if (redMask.at<uchar>(point3.y - rect.y, point3.x - rect.x) == 255) {
            distance3 = sqrt(atan((rect.x - result.x) * a / f) * atan((rect.x - result.x) * a / f) + atan((result.y - rect.y) * a / f) * atan((result.y - rect.y) * a / f));
        }
        if (redMask.at<uchar>(point4.y - rect.y, point4.x - rect.x) == 255) {
            distance4 = sqrt(atan((rect.x - result.x) * a / f) * atan((rect.x - result.x) * a / f) + atan((result.y - rect.y) * a / f) * atan((result.y - rect.y) * a / f));
        }
        // 假设 a 是最大值
        double max_dis = 0;
        max_dis = distance1;
        string maxVar = "distance1";

        // 比较并更新最大值
        if (distance2 > max_dis) {
            max_dis = distance2;
            maxVar = "distance2";
        }
        if (distance3 > max_dis) {
            max_dis = distance3;
            maxVar = "distance3";
        }
        if (distance4 > max_dis) {
            max_dis = distance4;
            maxVar = "distance4";
        }
        if (maxVar == "distance1") {
            point_result = point1;
        }else if (maxVar == "distance2") {
            point_result = point2;
        }else if (maxVar == "distance3") {
            point_result = point3;
        }else if (maxVar == "distance4") {
            point_result = point4;
        }
        
        //for (const auto& point : contour) {
        //    if (rect.contains(point)) {
        //        std::string position;
        //        if (redMask.at<uchar>(point.y - rect.y, point.x - rect.x) != 255) {
        //            continue;
        //        }
        //        // Determine which edge the point is closest to
        //        if (point.x == rect.x) position = "left";
        //        else if (point.x == rect.x + rect.width - 1) position = "right";
        //        else if (point.y == rect.y) position = "top";
        //        else if (point.y == rect.y + rect.height - 1) position = "bottom";
        //        else position = "inside";
        //        double distance = sqrt(atan((point.x - result.x) * a / f) * atan((point.x - result.x) * a / f) + atan(( result.y-point.y) * a / f) * atan((result.y - point.y) * a / f));
        //        if (distance > max_dis) {
        //            max_dis = distance;
        //            if (position == "right") {
        //                point_result = Point(point.x - maskrectangle[0] + maskcenter.x, point.y);
        //            }else if (position == "left") {
        //                point_result = Point(point.x + maskcenter.x, point.y);
        //            }else if (position == "top") {
        //                point_result = Point(point.x, point.y + maskcenter.y);
        //            }else if (position == "bottom") {
        //                point_result = Point(point.x, point.y - maskrectangle[1] + maskcenter.y);
        //            }
        //        }
        //    }
        //}


    }

    //// 创建一个全黑图像作为背景
    //cv::Mat blackBackground(binaryImage.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    //// 绘制筛选后的轮廓
    //cv::drawContours(blackBackground, filteredContours, -1, cv::Scalar(0, 255, 0), 2);
    /*imshow("blackBackground", blackBackground);
    waitKey(10);*/



    // 拆分HSV通道
    //std::vector<cv::Mat> hsvChannels;
    //cv::split(hsvImage, hsvChannels);

    //// 显示HSV通道
    ///*cv::imshow("Saturation Channel", hsvChannels[1]);*/

    //// 获取饱和度通道
    //cv::Mat saturationChannel = hsvChannels[1];

    

    //// 进行图像处理，这里以转换为灰度图为例
    //cv::Mat processedImage;
    //cv::cvtColor(frame, processedImage, cv::COLOR_BGR2GRAY);

    
}


//找轨迹点
void identifyPointCoordinates(Mat& frame, Point2f *result,Point2f& maskcenter,vector<int>* maskrectangle) {

    // 灰度处理
    Mat grayFrame;
    cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
   
    Mat gaussianBlurImage;
    GaussianBlur(grayFrame, gaussianBlurImage, Size(5, 5), 0);

    // 计算图像的梯度
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Sobel(gaussianBlurImage, grad_x, CV_16S, 1, 0); // x方向梯度
    Sobel(gaussianBlurImage, grad_y, CV_16S, 0, 1); // y方向梯度
    convertScaleAbs(grad_x, abs_grad_x); // x方向梯度的绝对值
    convertScaleAbs(grad_y, abs_grad_y); // y方向梯度的绝对值

    // 合并梯度图像
    Mat edges;
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, edges);

    // 应用阈值，只保留像素值差距大于等于100的边缘
    threshold(edges, edges, 50, 255, THRESH_BINARY);

    
    Mat morphologyImage;
    Mat kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21), Point(-1, -1));
    morphologyEx(edges, morphologyImage, MORPH_CLOSE,kernel, Point(-1, -1), 1);

    // 使用Canny边缘检测找到图像边缘
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(morphologyImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 轮廓为空
    if (contours.empty()) {
        cerr << "Error: No contours found." << endl;
        return ;
    }
    int maxlength = 0;
    vector<Mat> imageVector;
    vector<Point> imagePoint;
    // 创建一个与原始图像大小相同的全零图像
    Mat resultImage = Mat::zeros(frame.size(), frame.type());
    // 遍历所有轮廓
    for (size_t i = 0; i < contours.size(); i++) {
        // 寻找轮廓的矩形边界框
        Rect rect = boundingRect(contours[i]);
        // 计算矩形的长宽比
        float aspectRatio1 = static_cast<float>(rect.width) / rect.height;
        float aspectRatio2 = static_cast<float>(rect.height) / rect.width;
        float area = static_cast<float>(rect.height) * rect.width;
        // 如果矩形长宽比大于2，则跳过
        if (aspectRatio1 > 1.5 || aspectRatio2 > 1.5 || area < 100) {
            continue;
        }
        //提取二值图的面积
        Mat roidilateImage = morphologyImage(rect);
       /* imshow("roidilateImage", roidilateImage);
        waitKey(0);*/
        if (countNonZero(roidilateImage) / area <= 0.75)
            continue;

        // 提取矩形区域的ROI
        Mat roiImage = frame(rect);
        Mat roigray;
        cvtColor(roiImage, roigray, COLOR_BGR2GRAY);
        Mat roigaussianBlur;
        GaussianBlur(roigray, roigaussianBlur, Size(3, 3), 0);
        Mat edges1;
        Canny(roigaussianBlur, edges1, 50, 150); // 使用Canny边缘检测，参数可以根据具体情况调整
        vector<Vec3f> circles;
        // 检测边缘图像中是否存在非零像素
        if (!edges1.empty()) {
             // 使用霍夫变换检测圆形
            
            HoughCircles(edges1, circles, HOUGH_GRADIENT, 1,
                edges1.rows / 4,  // 圆心之间的最小距离
                100, 30, 10, 100 // 最小半径和最大半径
            );

            // 检查是否检测到了圆
            if (circles.empty()) {
                continue;
                
            }
          
        }
        else {
            continue;
        }

        // 转换图像到HSV格式
        Mat hsv_image;
        cvtColor(roiImage, hsv_image, COLOR_BGR2HSV);

        // 定义红色在HSV中的阈值范围
        Scalar lower_red = Scalar(156, 43, 46);   // 低阈值 (H, S, V)
        Scalar upper_red = Scalar(180, 255, 255); // 高阈值 (H, S, V)

        // 创建掩码，通过阈值过滤图像
        Mat mask;
        inRange(hsv_image, lower_red, upper_red, mask);
        
        if (countNonZero(mask) > 0) {
            // 找到轮廓
            std::vector<std::vector<Point>> maskcontours;
            findContours(mask, maskcontours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            // 计算轮廓的矩
            Moments M = moments(maskcontours[0]);

            // 计算质心
           maskcenter = Point(M.m10 / M.m00, M.m01 / M.m00);
           if (norm(maskcenter - Point2f(rect.width / 2, rect.height / 2)) > 30) {
                continue;
            }
            
           /*imshow("roidilateImage", roidilateImage);
           waitKey(0);*/
        }
        else {
            continue;
        }

        // 在结果图像上将矩形区域设为原始图像对应区域
        Mat roi = resultImage(rect);
        frame(rect).copyTo(roi);
        *result = Point(rect.x + maskcenter.x, rect.y + maskcenter.y);
        cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);
        maskrectangle->push_back(rect.width);
        maskrectangle->push_back(rect.height);
        
    }
    
    //找
    // 显示结果图像
    
    if (result->x != 0.0f && result->y != 0.0f) {
        circle(frame, *result, 2, Scalar(0, 0, 255), 2);
    }
    cout << "x:" << result->x << "y: "<< result->y << endl;
    /*imshow("frame", frame);
    waitKey(0);*/
    return;
}

// 比较轮廓面积
bool compareContourAreas(vector<Point> contour1, vector<Point> contour2) {
    double area1 = fabs(contourArea(contour1));
    double area2 = fabs(contourArea(contour2));
    return (area1 > area2);
}


int main() {
    //Mat image = imread("D:/sample/9.jpg");
    //Point2f result;
    //Point2f maskcenter;
    //vector<int> maskrectangle;
    //identifyPointCoordinates(image, &result, maskcenter, &maskrectangle);
    //circle(image, result, 2, Scalar(0, 0, 255), 2);
    //// 创建文件名
    //std::string filename = "D:/photo_yes/a.jpg";

    //// 保存当前帧
    //cv::imwrite(filename, image);
	//读取视频
    VideoCapture cap("C:/Users/Honor/Desktop/video/a (2).mp4");

    // 视频获取是否成功
    if (!cap.isOpened()) {
        cerr << "Error: Failed to open video file." << endl;
        return -1;
    }

    // 
    double fps = cap.get(CAP_PROP_FPS);
    cout << "Frames per second: " << fps << endl;

    // 视频窗口
    namedWindow("Video", WINDOW_NORMAL);
    vector<Point2f> move_coordinate;
    int frameCount = 0;
    while (true) {
        Mat frame;
        // 读取帧
        if (!cap.read(frame)) {
            cerr << "Error: Failed to read frame." << endl;
            break;
        }

        // 创建文件名
        std::string filename = "D:/photo/a(2)/" + std::to_string(frameCount) + ".jpg";

        // 保存当前帧
        cv::imwrite(filename, frame);

        frameCount++;

    }

    //释放窗口
    cap.release();
    destroyAllWindows();
    //fs::path inputFolder = "D:/photo";
    //fs::path outputFolder = "D:/photo/processed";

    //if (!fs::is_directory(inputFolder)) {
    //    std::cerr << "Error: Input folder does not exist or is not a directory." << std::endl;
    //    return -1;
    //}

    //if (!fs::exists(outputFolder)) {
    //    fs::create_directory(outputFolder);
    //}
    //int counta = 0;
    //int countb = 246;
    //for (const auto& entry : fs::directory_iterator(inputFolder)) {
    //    if (entry.is_regular_file() &&
    //        (entry.path().extension() == ".jpg" || entry.path().extension() == ".png")) {
    //        fs::path outputPath = outputFolder / entry.path().filename();
    //        cv::Mat frame = cv::imread(entry.path().string());

    //        if (frame.empty()) {
    //            std::cerr << "No get frame " << std::endl;
    //            break;
    //        }
    //        Point points;
    //        processImage(frame, result, maskcenter, maskrectangle, 0.003645,75, points);
    //        if (points.x == 0) {
    //           // 创建文件名
    //           std::string filename = "D:/photo_no/" + std::to_string(counta) + ".jpg";

    //           // 保存当前帧
    //           cv::imwrite(filename, frame);
    //           counta++;
    //        }
    //        else {
    //            // 创建文件名
    //            std::string filename = "D:/photo_yes/" + std::to_string(countb) + ".jpg";

    //            circle(frame, points, 2, Scalar(0, 0, 255), 2);
    //            // 保存当前帧
    //            cv::imwrite(filename, frame);
    //            countb++;
    //        
    //        }
    //    }
    //}


	return 0;


}