#include "RetinaFace.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "timer.h"

using namespace std;
using NetOutputs = std::vector<std::vector<std::vector<float>>>;

void DrawResult(cv::Mat& src, const std::vector<FaceDetectInfo>& detect_info, cv::Size rz, int pixel_sz = 8) {
    if (src.empty()) return;
    anchor_box rect;
    float scale_ratio;
    int width = src.cols;
    int height = src.rows;
    float x1, x2, x3, x4, y1, y2, y3, y4;
    std::vector<float> output_lab;
    for (auto& it : detect_info) {
        Mat roi;
        rect = it.rect;
        scale_ratio = it.scale_ratio;
        /*
````````*    p1(x1,y1)p2(x2,y1)
````````*    p3(x1,y2)p4(x2,y2)
````````*/
        x1 = it.rect.x1 * scale_ratio * (1.0 * width / rz.width);
        y1 = it.rect.y1 * scale_ratio * (1.0 * height / rz.height);
        x2 = it.rect.x2 * scale_ratio * (1.0 * width / rz.width);
        y2 = it.rect.y2 * scale_ratio * (1.0 * height / rz.height);
        cv::Point p1(x1, y1), p2(x2, y1), p3(x1, y2), p4(x2, y2);
        cv::line(src, p1, p2, cv::Scalar(0, 255, 0), pixel_sz, cv::LineTypes::LINE_AA);
        cv::line(src, p1, p3, cv::Scalar(0, 255, 0), pixel_sz, cv::LineTypes::LINE_AA);
        cv::line(src, p2, p4, cv::Scalar(0, 255, 0), pixel_sz, cv::LineTypes::LINE_AA);
        cv::line(src, p3, p4, cv::Scalar(0, 255, 0), pixel_sz, cv::LineTypes::LINE_AA);

        for (size_t j = 0; j < 5; j++) {
            cv::Point2f pt = cv::Point2f(it.pts.x[j] * scale_ratio * (1.0 * width / rz.width),
                it.pts.y[j] * scale_ratio * (1.0 * height / rz.height));
            cv::circle(src, pt, pixel_sz, Scalar(0, 255, 0), cv::LineTypes::LINE_AA);
        }
    }
}

TEST(PLUGINS, RetinaFace) {
	std::string rtfm = "../../../data/models/retinaface";
	RetinaFace rf(rtfm, "net3", 0.4);
	std::vector<std::string> img_lists = {
	"../../../data/images/test.jpg",
	};
	std::vector<std::string> save_img_lists = {
		"../../../data/images/rf/result.jpg",
	};
	std::vector<FaceDetectInfo> detect_info;
	size_t sz = img_lists.size();
	for (int i = 0; i < sz; ++i) {
		cv::Mat src = cv::imread(img_lists[i], cv::IMREAD_COLOR);
		assert(!test_img.empty());
		cv::Size rz(640, 640);
		cv::Mat img_rsz;
		cv::resize(src, img_rsz, rz);
		detect_info = rf.detect(img_rsz, 0.9, 1.0);
		DrawResult(src, detect_info, rz);
		cv::imwrite(save_img_lists[i], src);
	}
}