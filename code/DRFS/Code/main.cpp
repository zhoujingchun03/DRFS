#include <iostream>
#include "./Stitching/NISwGSP_Stitching.h"
#include "./Debugger/TimeCalculator.h"
#include "Test/Test.h"



using namespace std;


int GRID_SIZE_w = 40;
int GRID_SIZE_h = 40;



int main(int argc, const char* argv[]) {
	Eigen::initParallel(); /* remember to turn off "Hardware Multi-Threading */
	//TestSP::testContours();

	CV_DNN_REGISTER_LAYER_CLASS(Crop, CropLayer);
	cout << "nThreads = " << Eigen::nbThreads() << endl;
	cout << "[#Images : " << argc - 1 << "]" << endl;
	time_t start = clock();
	TimeCalculator timer;

	for (int i = 1; i < argc; ++i) {
		cout << "i = " << i << ", [images : " << argv[i] << "]" << endl;
		//1-3:1.parameter类输入输出文件夹以及预处理txt文件
		//1-3:2.multiimages类初始化匹配点数量及结果图
		//1-3:3.multiimages类ransac筛选后的匹配点数量及结果图
		MultiImages multi_images(argv[i], LINES_FILTER_WIDTH, LINES_FILTER_LENGTH);
		///* 2d */
		NISwGSP_Stitching niswgsp(multi_images);
		niswgsp.setWeightToAlignmentTerm(1);    //设置对齐项权重1
		niswgsp.setWeightToLocalSimilarityTerm(0.75);   //设置局部相似项权重0.75
		niswgsp.setWeightToGlobalSimilarityTerm(6, 10, GLOBAL_ROTATION_2D_METHOD);  //设置全局相似项权重
		niswgsp.setWeightToContentPreservingTerm(1.5);   //设置内容保护权重1.5
		Mat blend_linear;
		Mat blend_linear1;
		vector<vector<Point2> > original_vertices;
		if (RUN_TYPE == 1) {
			blend_linear = niswgsp.solve_content(BLEND_LINEAR, original_vertices);
			//blend_linear1 = niswgsp.solve_content(BLEND_AVERAGE, original_vertices);
		}
		else {
			blend_linear = niswgsp.solve(BLEND_LINEAR, original_vertices);
		}
		time_t end = clock();
		cout << "Time:" << double(end - start) / CLOCKS_PER_SEC << endl;
		niswgsp.writeImage(blend_linear, BLENDING_METHODS_NAME[BLEND_LINEAR]);
		//niswgsp.writeImage(blend_linear1, BLENDING_METHODS_NAME[BLEND_AVERAGE]);
		niswgsp.assessment(original_vertices);
	}


	return 0;
}


//
//#include <opencv2/opencv.hpp>
//
//int main() {
//    // 读取图像
//    cv::Mat imag11 = cv::imread("C:/Users/grey/Desktop/ESP/GES-GSP-Stitching-master/TestImgs/0.jpg");
//    cv::Mat imag22 = cv::imread("C:/Users/grey/Desktop/ESP/GES-GSP-Stitching-master/TestImgs/1.jpg");
//    cv::Mat image1 = cv::imread("C:/Users/grey/Desktop/ESP/GES-GSP-Stitching-master/TestImgs/0.jpg", cv::IMREAD_GRAYSCALE); 
//    cv::Mat image2 = cv::imread("C:/Users/grey/Desktop/ESP/GES-GSP-Stitching-master/TestImgs/1.jpg", cv::IMREAD_GRAYSCALE);
//    resize(imag11, imag11, cv::Size(500, 500 * (double)imag11.rows / imag11.cols));
//    resize(imag22, imag22, cv::Size(500, 500 * (double)imag22.rows / imag22.cols));
//    resize(image1, image1, cv::Size(500, 500 * (double)image1.rows / image1.cols));
//    resize(image2, image2, cv::Size(500, 500 * (double)image2.rows / image2.cols));
//    // 创建SIFT检测器
//    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
//
//    // 检测关键点和计算描述符
//    std::vector<cv::KeyPoint> keypoints1, keypoints2;
//    cv::Mat descriptors1, descriptors2;
//    sift->detectAndCompute(image1, cv::Mat(), keypoints1, descriptors1);
//    sift->detectAndCompute(image2, cv::Mat(), keypoints2, descriptors2);
//
//    // 创建FLANN匹配器
//    cv::Ptr<cv::DescriptorMatcher> flann = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
//
//    // 使用KNN匹配
//    std::vector<std::vector<cv::DMatch>> knn_matches;
//    flann->knnMatch(descriptors1, descriptors2, knn_matches, 2);
//
//    // 仅保留良好的匹配
//    std::vector<cv::DMatch> good_matches;
//    for (size_t i = 0; i < knn_matches.size(); ++i) {
//        if (knn_matches[i][0].distance < 0.7 * knn_matches[i][1].distance) {
//            good_matches.push_back(knn_matches[i][0]);
//        }
//    }
//
//    // 设置RANSAC参数
//    double ransac_reproj_threshold = 5.0;
//    double ransac_confidence = 0.99;
//
//    // 进行RANSAC
//    std::vector<cv::Point2f> src_pts, dst_pts;
//    for (size_t i = 0; i < good_matches.size(); ++i) {
//        src_pts.push_back(keypoints1[good_matches[i].queryIdx].pt);
//        dst_pts.push_back(keypoints2[good_matches[i].trainIdx].pt);
//    }
//
//    cv::Mat H = cv::findHomography(src_pts, dst_pts, cv::RANSAC, ransac_reproj_threshold);
//
//    // 剔除误匹配点
//    std::vector<cv::DMatch> final_matches;
//    for (size_t i = 0; i < good_matches.size(); ++i) {
//        cv::Mat pt1(3, 1, CV_64FC1);
//        pt1.at<double>(0, 0) = src_pts[i].x;
//        pt1.at<double>(1, 0) = src_pts[i].y;
//        pt1.at<double>(2, 0) = 1.0;
//
//        cv::Mat pt2_estimated = H * pt1;
//        pt2_estimated /= pt2_estimated.at<double>(2, 0);
//
//        double distance = cv::norm(dst_pts[i] - cv::Point2f(pt2_estimated.at<double>(0, 0), pt2_estimated.at<double>(1, 0)));
//        if (distance < ransac_reproj_threshold) {
//            final_matches.push_back(good_matches[i]);
//        }
//    }
//
//    // 特征点
//    cv::Mat keypoint_img;
//    drawKeypoints(imag22, keypoints2, keypoint_img);
//    imshow("KeyPoints Image", keypoint_img);
//    imwrite("keypoint_img2.jpg", keypoint_img);
//
//    // 绘制匹配结果
//    cv::Mat result_image;
//    cv::drawMatches(image1, keypoints1, image2, keypoints2, final_matches, result_image);
//
//    cv::imshow("Matches", result_image);
//    cv::waitKey(0);
//    cv::destroyAllWindows();
//
//    return 0;
//}
