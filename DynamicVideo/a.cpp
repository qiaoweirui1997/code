//// Plot.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
////
//
//#include <iostream>
//
//#include "matplotlibcpp.h"
//
//namespace plt = matplotlibcpp;
//
//int main() {
//	int n = 1000;
//	std::vector<double> x, y, z;
//	int count = 1;
//	for (int i = 0; i < n; i++) {
//		x.push_back(i * i);
//		y.push_back(sin(2 * 3.14 * i / 360.0));
//		z.push_back(log(i));
//
//		if (i % 10 == 0) {
//			// Clear previous plot
//			plt::clf();
//			// Plot line from given x and y data. Color is selected automatically.
//			plt::plot(x, y);
//			// Plot a line whose name will show up as "log(x)" in the legend.
//			plt::named_plot("log(x)", x, z);
//
//			// Set x-axis to interval [0,1000000]
//			plt::xlim(0, n * n);
//
//			// Add graph title
//			plt::title("Sample figure");
//			// Enable legend.
//			plt::legend();
//			// Display plot continuously
//			std::string pathObj = "animation//" + std::to_string(count);
//			//plt::save(pathObj );
//			plt::pause(0.01);
//			count++;
//		}
//	}
//}


    //imshow("dilateImage", dilateImage);
   // Mat dilateImage;
   // Mat kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(33, 33), Point(-1, -1));
   // dilate(thresholdImage, dilateImage, kernel, Point(-1, -1), 1);
   // Mat erodeImage;
   // erode(dilateImage, erodeImage, kernel, Point(-1, -1), 2);
   // 
   // vector<vector<Point>> contours;
   // vector<Vec4i> hierarchy;
   // findContours(erodeImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

   // // 轮廓为空
   // if (contours.empty()) {
   //     cerr << "Error: No contours found." << endl;
   //     return ;
   // }
   // int maxlength = 0;
   // // 遍历所有轮廓
   // for (size_t i = 0; i < contours.size(); i++) {
   //     // 拟合最小外接矩形
   //     RotatedRect boundingBox = minAreaRect(contours[i]);
   //     // 计算轮廓的白色像素面积
   //     Mat mask = Mat::zeros(erodeImage.size(), CV_8UC1);
   //     drawContours(mask, contours, static_cast<int>(i), Scalar(255), FILLED);
   //  
   //     double contourArea = countNonZero(mask);
   //     // 计算矩形的面积
   //     double rectArea = boundingBox.size.width * boundingBox.size.height;

   //     // 计算白色像素占比
   //     double areaRatio = contourArea / rectArea;
   //     // 计算长宽比
   //     float aspectRatio = static_cast<float>(boundingBox.size.width) / boundingBox.size.height;
   //     int lensize = max(boundingBox.size.width, boundingBox.size.height);
   //     if (lensize > maxlength)
   //         maxlength = lensize;

   //     // 如果长宽比大于2，则删除对应轮廓
   //     if (aspectRatio > 2.0 || areaRatio < 0.8) {
   //         contours.erase(contours.begin() + i);
   //         i--;  // 因为删除了一个元素，需要调整索引
   //     }
   // }
   // if (contours.empty()) {
   //     cerr << "Error: No contours found." << endl;
   //     return;
   // }
   // // 轮廓按照面积排序
   // sort(contours.begin(), contours.end(), compareContourAreas);
   // // 创建新的二值图像，绘制保留的轮廓
   // Mat filteredContoursImage = Mat::zeros(erodeImage.size(), CV_8UC1);
   // drawContours(filteredContoursImage, contours, 0, Scalar(255), FILLED);

   // Mat theExtractedImage;
   // grayFrame.copyTo(theExtractedImage, filteredContoursImage);
   // 
   // for (int y = 0; y < theExtractedImage.cols; y++) {
   //     for (int x = 0; x < theExtractedImage.rows; x++) {
   //         if (theExtractedImage.at<uchar>(x, y) == 0) {
   //             theExtractedImage.at<uchar>(x, y) = 255;
   //         }
   //     }
   // }
   // Mat eqgaussianBlurImage;
   // GaussianBlur(theExtractedImage, eqgaussianBlurImage, Size(5, 5), 0);
   // // 自适应直方图均衡化
   // Ptr<CLAHE> clahe = createCLAHE();
   // clahe->setClipLimit(4.0); 

   // Mat equalizedImage;
   // clahe->apply(eqgaussianBlurImage, equalizedImage);
   // equalizedImage.convertTo(equalizedImage, CV_8U,1.3,10);
   // 
   // Mat secondThresholdImage;
   // threshold(equalizedImage, secondThresholdImage, 180, 255, THRESH_BINARY_INV);
   //
   // 
   // Mat morResultImage;
   // Mat morkernel = getStructuringElement(cv::MORPH_RECT, cv::Size(5,5), Point(-1, -1));
   // morphologyEx(secondThresholdImage, morResultImage, MORPH_OPEN, morkernel, Point(-1, -1), 1);

   // Mat verticalMorImage, levelMorImage;
   // Mat morkernel1 = getStructuringElement(cv::MORPH_RECT, cv::Size(maxlength / 3, 3), Point(-1, -1));
   // Mat morkernel2 = getStructuringElement(cv::MORPH_RECT, cv::Size(3, maxlength / 3), Point(-1, -1));
   // morphologyEx(morResultImage, levelMorImage, MORPH_OPEN, morkernel1, Point(-1, -1),1);
   // morphologyEx(morResultImage, verticalMorImage, MORPH_OPEN, morkernel2, Point(-1, -1), 1);

   // // 滑动窗口设置
   //Size window_size(5, 5); // (width, height)
   //int step = 2;

   //// 最小平均像素
   //double min_avg_pixel_value = numeric_limits<double>::infinity();
   //Point min_avg_pixel_pos;

   //// 窗口遍历图片
   //for (int y = 0; y < theExtractedImage.rows - window_size.height + 1; y += step) {
   //    for (int x = 0; x < theExtractedImage.cols - window_size.width + 1; x += step) {
   //        
   //        Mat window = equalizedImage(Rect(x, y, window_size.width, window_size.height));

   //      
   //        Scalar avg_pixel_value = mean(window);

   //        //计算窗口对应图片的平均像素值
   //        if (avg_pixel_value[0] < min_avg_pixel_value) {
   //            min_avg_pixel_value = avg_pixel_value[0];
   //            min_avg_pixel_pos = Point(x, y);
   //        }
   //    }
   //}
   // imshow("eqgaussianBlurImage", eqgaussianBlurImage);
   // imshow("levelMorImage", levelMorImage);
   // imshow("verticalMorImage", verticalMorImage);
   // waitKey(0);
   // ////////////////////////////////
   // // // 找到白色像素的位置
   // vector<Point> points1, points2;
   // // 遍历图像的像素
   // for (int y = 0; y < levelMorImage.rows; ++y) {
   //     for (int x = 0; x < levelMorImage.cols; ++x) {
   //         if (levelMorImage.at<uchar>(y, x) == 255) {
   //             points1.push_back(Point(x, y));
   //         }
   //     }
   // }
   // for (int y = 0; y < verticalMorImage.rows; ++y) {
   //     for (int x = 0; x < verticalMorImage.cols; ++x) {
   //         if (verticalMorImage.at<uchar>(y, x) == 255) {
   //             points2.push_back(Point(x, y));
   //         }
   //     }
   // }
   // // 使用 fitLine 拟合直线
   // Vec4f lineParams1, lineParams2;
   // fitLine(points1, lineParams1, DIST_L2, 0, 0.01, 0.01);
   // fitLine(points2, lineParams2, DIST_L2, 0, 0.01, 0.01);

   // // 提取直线参数
   // float vx1 = lineParams1[0], vy1 = lineParams1[1];
   // float x01 = lineParams1[2], y01 = lineParams1[3];

   // float vx2 = lineParams2[0], vy2 = lineParams2[1];
   // float x02 = lineParams2[2], y02 = lineParams2[3];

   // // 解线性方程组计算交点
   // float t, s;
   // if (vx1 * vy2 - vy1 * vx2 != 0) {
   //     t = ((x02 - x01) * vy2 - (y02 - y01) * vx2) / (vx1 * vy2 - vy1 * vx2);
   //     s = ((x02 - x01) * vy1 - (y02 - y01) * vx1) / (vx1 * vy2 - vy1 * vx2);
   // }
   // else {
   //     cout << "Lines are parallel or coincident, no unique intersection." << endl;
   //     return ;
   // }

   // // 计算交点坐标
   // float intersectionX = x01 + t * vx1;
   // float intersectionY = y01 + t * vy1;
   // 
   // circle(frame,Point(intersectionX, intersectionY),2,Scalar(0,0,255),2);


    //Mat intersectionArea = levelMorImage & verticalMorImage;
    //// 寻找轮廓
    //vector<vector<Point>> contours1;
    //findContours(intersectionArea, contours1, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    //for (size_t i = 0; i < contours1.size(); ++i) {
    //    // 计算包围轮廓的最小矩形
    //    Rect boundingRect = cv::boundingRect(contours1[i]);

    //    // 在结果图像中绘制矩形
    //    rectangle(frame, boundingRect, Scalar(0, 0, 255), 2); // 使用绿色绘制边界框
    //}
    //vector<vector<Point>> contours;
    //vector<Vec4i> hierarchy;
    //findContours(erodeImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    //// 轮廓为空
    //if (contours.empty()) {
    //    cerr << "Error: No contours found." << endl;
    //    return ;
    //}

    //// 轮廓按照面积排序
    //sort(contours.begin(), contours.end(), compareContourAreas);

    //Rect boundRect = boundingRect(contours[0]);

    //Point2f center = Point2d(boundRect.x + boundRect.width / 2, boundRect.y + boundRect.height / 2);
    //// 找标靶位置
    //int x_min = center.x - 150;
    //int y_min = center.y - 150;
    //int x_max = center.x + 150;
    //int y_max = center.y + 150;


    //x_min = max(x_min, 0);
    //y_min = max(y_min, 0);
    //x_max = min(x_max, grayFrame.cols - 1);
    //y_max = min(y_max, grayFrame.rows - 1);

    //// 绘制标靶区域
    //Rect roiRect(x_min, y_min, x_max - x_min, y_max - y_min);

    ////裁剪标靶图
    //Mat boundingImage = grayFrame(roiRect);

    ////额外处理
    //Mat binarizationImage;
    //threshold(boundingImage, binarizationImage, 50, 255, THRESH_BINARY_INV + THRESH_OTSU);
    //Mat binerodeImage;
    //Mat binerodekernel = getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5), Point(-1, -1));
    //erode(binarizationImage, binerodeImage, binerodekernel, Point(-1, -1),1);
    //Mat bindilateImage;
    //Mat bindilatekernel = getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5), Point(-1, -1));
    //dilate(binerodeImage, bindilateImage, bindilatekernel, Point(-1, -1), 2);
   /* imshow("binarizationImage", binarizationImage);
    waitKey(0);*/
    //// 自适应直方图均衡化
    //Ptr<CLAHE> clahe = createCLAHE();
    //clahe->setClipLimit(4.0); 

    //Mat equalizedImage;
    //clahe->apply(boundingImage, equalizedImage);
