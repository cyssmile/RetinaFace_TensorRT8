#include <glog/logging.h>
#include "RetinaFace.h"
#include <cuda_runtime_api.h>

void imageROIResize8U3C(void* src, int srcWidth, int srcHeight, cv::Rect imgROI, void* dst, int dstWidth, int dstHeight);
void convertBGR2RGBfloat(void* src, void* dst, int width, int height, cudaStream_t stream);
void imageSplit(const void* src, float* dst, int width, int height, cudaStream_t stream);

//processing
anchor_win  _whctrs(anchor_box anchor)
{
    //Return width, height, x center, and y center for an anchor (window).
    anchor_win win;
    win.w = anchor.x2 - anchor.x1 + 1;
    win.h = anchor.y2 - anchor.y1 + 1;
    win.x_ctr = anchor.x1 + 0.5 * (win.w - 1);
    win.y_ctr = anchor.y1 + 0.5 * (win.h - 1);

    return win;
}

anchor_box _mkanchors(anchor_win win)
{
    //Given a vector of widths (ws) and heights (hs) around a center
    //(x_ctr, y_ctr), output a set of anchors (windows).
    anchor_box anchor;
    anchor.x1 = win.x_ctr - 0.5 * (win.w - 1);
    anchor.y1 = win.y_ctr - 0.5 * (win.h - 1);
    anchor.x2 = win.x_ctr + 0.5 * (win.w - 1);
    anchor.y2 = win.y_ctr + 0.5 * (win.h - 1);

    return anchor;
}

vector<anchor_box> _ratio_enum(anchor_box anchor, vector<float> ratios)
{
    //Enumerate a set of anchors for each aspect ratio wrt an anchor.
    vector<anchor_box> anchors;
    for (size_t i = 0; i < ratios.size(); i++) {
        anchor_win win = _whctrs(anchor);
        float size = win.w * win.h;
        float scale = size / ratios[i];

        win.w = std::round(sqrt(scale));
        win.h = std::round(win.w * ratios[i]);

        anchor_box tmp = _mkanchors(win);
        anchors.push_back(std::move(tmp));
    }

    return anchors;
}

vector<anchor_box> _scale_enum(anchor_box anchor, vector<int> scales)
{
    //Enumerate a set of anchors for each scale wrt an anchor.
    vector<anchor_box> anchors;
    for (size_t i = 0; i < scales.size(); i++) {
        anchor_win win = _whctrs(anchor);

        win.w = win.w * scales[i];
        win.h = win.h * scales[i];

        anchor_box tmp = _mkanchors(win);
        anchors.push_back(std::move(tmp));
    }

    return anchors;
}

vector<anchor_box> generate_anchors(int base_size = 16, vector<float> ratios = { 0.5, 1, 2 },
    vector<int> scales = { 8, 64 }, int stride = 16, bool dense_anchor = false)
{
    //Generate anchor (reference) windows by enumerating aspect ratios X
    //scales wrt a reference (0, 0, 15, 15) window.

    anchor_box base_anchor;
    base_anchor.x1 = 0;
    base_anchor.y1 = 0;
    base_anchor.x2 = base_size - 1;
    base_anchor.y2 = base_size - 1;

    vector<anchor_box> ratio_anchors;
    ratio_anchors = _ratio_enum(base_anchor, ratios);

    vector<anchor_box> anchors;
    for (size_t i = 0; i < ratio_anchors.size(); i++) {
        vector<anchor_box> tmp = _scale_enum(ratio_anchors[i], scales);
        anchors.insert(anchors.end(), tmp.begin(), tmp.end());
    }

    if (dense_anchor) {
        assert(stride % 2 == 0);
        vector<anchor_box> anchors2 = anchors;
        for (size_t i = 0; i < anchors2.size(); i++) {
            anchors2[i].x1 += stride / 2;
            anchors2[i].y1 += stride / 2;
            anchors2[i].x2 += stride / 2;
            anchors2[i].y2 += stride / 2;
        }
        anchors.insert(anchors.end(), anchors2.begin(), anchors2.end());
    }

    return anchors;
}
/*
* @brief use anchor_cfg to generate achors
* for i to anchor_cfg.size:
*   for j in anchor_cfg[i].ratios:
*      for k in anchor_cfg[i].ratios[j]:
*        anchor =  ratios(scale(k))
*        add anchor to anchors
*/
vector<vector<anchor_box>> generate_anchors_fpn(bool dense_anchor = false, vector<anchor_cfg> cfg = {})
{
    //Generate anchor (reference) windows by enumerating aspect ratios X
    //scales wrt a reference (0, 0, 15, 15) window.

    vector<vector<anchor_box>> anchors;
    for (size_t i = 0; i < cfg.size(); i++) {
        //stride[32 16 8]
        anchor_cfg tmp = cfg[i];
        int bs = tmp.BASE_SIZE;
        vector<float> ratios = tmp.RATIOS;
        vector<int> scales = tmp.SCALES;
        int stride = tmp.STRIDE;

        vector<anchor_box> r = generate_anchors(bs, ratios, scales, stride, dense_anchor);
        anchors.push_back(std::move(r));
    }

    return anchors;
}

/*
* transform image anchors to plane anchors(relative displacement)
*/
vector<anchor_box> anchors_plane(int height, int width, int stride, vector<anchor_box> base_anchors)
{
    /*
    height: height of plane
    width:  width of plane
    stride: stride ot the original image
    anchors_base: a base set of anchors
    */

    vector<anchor_box> all_anchors;
    for (size_t k = 0; k < base_anchors.size(); k++) {
        for (int ih = 0; ih < height; ih++) {
            int sh = ih * stride;
            for (int iw = 0; iw < width; iw++) {
                int sw = iw * stride;

                anchor_box tmp;
                tmp.x1 = base_anchors[k].x1 + sw;
                tmp.y1 = base_anchors[k].y1 + sh;
                tmp.x2 = base_anchors[k].x2 + sw;
                tmp.y2 = base_anchors[k].y2 + sh;
                all_anchors.push_back(std::move(tmp));
            }
        }
    }

    return all_anchors;
}

void clip_boxes(vector<anchor_box>& boxes, int width, int height)
{
    //Clip boxes to image boundaries.
    for (size_t i = 0; i < boxes.size(); i++) {

        boxes[i].x1 = (boxes[i].x1 < 0) ? 0 : boxes[i].x1;
        boxes[i].y1 = (boxes[i].y1 < 0) ? 0 : boxes[i].y1;
        boxes[i].x2 = (boxes[i].x2 > width - 1) ? (width - 1) : boxes[i].x2;
        boxes[i].y2 = (boxes[i].y2 > height - 1) ? (height - 1) : boxes[i].y2;

    }
}

void clip_boxes(anchor_box& box, int width, int height)
{
    //Clip boxes to image boundaries.
    box.x1 = (box.x1 < 0) ? 0 : box.x1;
    box.y1 = (box.y1 < 0) ? 0 : box.y1;
    box.x2 = (box.x2 > width - 1) ? (width - 1) : box.x2;
    box.y2 = (box.y2 > height - 1) ? (height - 1) : box.y2;

}

//######################################################################
//retinaface
//######################################################################

RetinaFace::RetinaFace(string& model, float nms)
    : network(network), nms_threshold(nms) {
    // backbone net
    int fmc = 3;
    _ratio = { 1.0 };

    //anchor_cfg
    if (fmc == 3) {
        _feat_stride_fpn = { 32, 16, 8 };
        anchor_cfg tmp;
        tmp.SCALES = { 32, 16 };
        tmp.BASE_SIZE = 16;
        tmp.RATIOS = _ratio;
        tmp.ALLOWED_BORDER = 9999;
        tmp.STRIDE = 32;
        cfg.push_back(tmp);

        tmp.SCALES = { 8, 4 };
        tmp.BASE_SIZE = 16;
        tmp.RATIOS = _ratio;
        tmp.ALLOWED_BORDER = 9999;
        tmp.STRIDE = 16;
        cfg.push_back(tmp);

        tmp.SCALES = { 2, 1 };
        tmp.BASE_SIZE = 16;
        tmp.RATIOS = _ratio;
        tmp.ALLOWED_BORDER = 9999;
        tmp.STRIDE = 8;
        cfg.push_back(tmp);
    }
    else {
        LOG(ERROR) << "[retinaface]:" << "please reconfig anchor_cfg" << network;
    }

    //load network
    trtNet = new TrtRetinaFaceNet("retina");
    trtNet->buildTrtContext(model + "/mnet025_v1.onnx", model + "/mnet025_v1.cache");

    int maxbatchsize = trtNet->getMaxBatchSize();
    int channels = trtNet->getChannel();
    int inputW = trtNet->getNetWidth();
    int inputH = trtNet->getNetHeight();
    //
    int inputsize = maxbatchsize * channels * inputW * inputH * sizeof(float);
    cpuBuffers = (float*)malloc(inputsize);
    memset(cpuBuffers, 0, inputsize);

    vector<int> outputW = trtNet->getOutputWidth();
    vector<int> outputH = trtNet->getOutputHeight();

    bool dense_anchor = false;
    vector<vector<anchor_box>> anchors_fpn = generate_anchors_fpn(dense_anchor, cfg);
    for (size_t i = 0; i < anchors_fpn.size(); i++) {
        int stride = _feat_stride_fpn[i];
        string key = "stride" + std::to_string(_feat_stride_fpn[i]);
        _anchors_fpn[key] = anchors_fpn[i];
        _num_anchors[key] = anchors_fpn[i].size();
        _anchors[key] = anchors_plane(outputH[i], outputW[i], stride, _anchors_fpn[key]);
    }

}

RetinaFace::~RetinaFace()
{
    delete trtNet;
    free(cpuBuffers);
}

vector<anchor_box> RetinaFace::bbox_pred(vector<anchor_box> anchors, vector<cv::Vec4f> regress)
{
    //"""
    //  Transform the set of class-agnostic boxes into class-specific boxes
    //  by applying the predicted offsets (box_deltas)
    //  :param boxes: !important [N 4]
    //  :param box_deltas: [N, 4 * num_classes]
    //  :return: [N 4 * num_classes]
    //  """

    vector<anchor_box> rects(anchors.size());
    for (size_t i = 0; i < anchors.size(); i++) {
        float width = anchors[i].x2 - anchors[i].x1 + 1;
        float height = anchors[i].y2 - anchors[i].y1 + 1;
        float ctr_x = anchors[i].x1 + 0.5 * (width - 1.0);
        float ctr_y = anchors[i].y1 + 0.5 * (height - 1.0);

        float pred_ctr_x = regress[i][0] * width + ctr_x;
        float pred_ctr_y = regress[i][1] * height + ctr_y;
        float pred_w = exp(regress[i][2]) * width;
        float pred_h = exp(regress[i][3]) * height;

        rects[i].x1 = pred_ctr_x - 0.5 * (pred_w - 1.0);
        rects[i].y1 = pred_ctr_y - 0.5 * (pred_h - 1.0);
        rects[i].x2 = pred_ctr_x + 0.5 * (pred_w - 1.0);
        rects[i].y2 = pred_ctr_y + 0.5 * (pred_h - 1.0);
    }

    return rects;
}

anchor_box RetinaFace::bbox_pred(anchor_box anchor, cv::Vec4f regress)
{
    anchor_box rect;

    float width = anchor.x2 - anchor.x1 + 1;
    float height = anchor.y2 - anchor.y1 + 1;
    float ctr_x = anchor.x1 + 0.5 * (width - 1.0);
    float ctr_y = anchor.y1 + 0.5 * (height - 1.0);

    float pred_ctr_x = regress[0] * width + ctr_x;
    float pred_ctr_y = regress[1] * height + ctr_y;
    float pred_w = exp(regress[2]) * width;
    float pred_h = exp(regress[3]) * height;

    rect.x1 = pred_ctr_x - 0.5 * (pred_w - 1.0);
    rect.y1 = pred_ctr_y - 0.5 * (pred_h - 1.0);
    rect.x2 = pred_ctr_x + 0.5 * (pred_w - 1.0);
    rect.y2 = pred_ctr_y + 0.5 * (pred_h - 1.0);

    return rect;
}

vector<FacePts> RetinaFace::landmark_pred(vector<anchor_box> anchors, vector<FacePts> facePts)
{
    vector<FacePts> pts(anchors.size());
    for (size_t i = 0; i < anchors.size(); i++) {
        float width = anchors[i].x2 - anchors[i].x1 + 1;
        float height = anchors[i].y2 - anchors[i].y1 + 1;
        float ctr_x = anchors[i].x1 + 0.5 * (width - 1.0);
        float ctr_y = anchors[i].y1 + 0.5 * (height - 1.0);

        for (size_t j = 0; j < 5; j++) {
            pts[i].x[j] = facePts[i].x[j] * width + ctr_x;
            pts[i].y[j] = facePts[i].y[j] * height + ctr_y;
        }
    }

    return pts;
}

FacePts RetinaFace::landmark_pred(anchor_box anchor, FacePts facePt)
{
    FacePts pt;
    float width = anchor.x2 - anchor.x1 + 1;
    float height = anchor.y2 - anchor.y1 + 1;
    float ctr_x = anchor.x1 + 0.5 * (width - 1.0);
    float ctr_y = anchor.y1 + 0.5 * (height - 1.0);

    for (size_t j = 0; j < 5; j++) {
        pt.x[j] = facePt.x[j] * width + ctr_x;
        pt.y[j] = facePt.y[j] * height + ctr_y;
    }

    return pt;
}

bool RetinaFace::CompareBBox(const FaceDetectInfo& a, const FaceDetectInfo& b)
{
    return a.score > b.score;
}

std::vector<FaceDetectInfo> RetinaFace::nms(std::vector<FaceDetectInfo>& bboxes, float threshold)
{
    std::vector<FaceDetectInfo> bboxes_nms;
    std::sort(bboxes.begin(), bboxes.end(), CompareBBox);

    int32_t select_idx = 0;
    int32_t num_bbox = static_cast<int32_t>(bboxes.size());
    std::vector<int32_t> mask_merged(num_bbox, 0);
    bool all_merged = false;

    while (!all_merged) {
        while (select_idx < num_bbox && mask_merged[select_idx] == 1)
            select_idx++;
        if (select_idx == num_bbox) {
            all_merged = true;
            continue;
        }

        bboxes_nms.push_back(bboxes[select_idx]);
        mask_merged[select_idx] = 1;

        anchor_box select_bbox = bboxes[select_idx].rect;
        float area1 = static_cast<float>((select_bbox.x2 - select_bbox.x1 + 1) * (select_bbox.y2 - select_bbox.y1 + 1));
        float x1 = static_cast<float>(select_bbox.x1);
        float y1 = static_cast<float>(select_bbox.y1);
        float x2 = static_cast<float>(select_bbox.x2);
        float y2 = static_cast<float>(select_bbox.y2);

        select_idx++;
        for (int32_t i = select_idx; i < num_bbox; i++) {
            if (mask_merged[i] == 1)
                continue;

            anchor_box& bbox_i = bboxes[i].rect;
            float x = std::max<float>(x1, static_cast<float>(bbox_i.x1));
            float y = std::max<float>(y1, static_cast<float>(bbox_i.y1));
            float w = std::min<float>(x2, static_cast<float>(bbox_i.x2)) - x + 1;   //<- float �Ͳ���1
            float h = std::min<float>(y2, static_cast<float>(bbox_i.y2)) - y + 1;
            if (w <= 0 || h <= 0)
                continue;

            float area2 = static_cast<float>((bbox_i.x2 - bbox_i.x1 + 1) * (bbox_i.y2 - bbox_i.y1 + 1));
            float area_intersect = w * h;


            if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > threshold) {
                mask_merged[i] = 1;
            }
        }
    }

    return bboxes_nms;
}


vector<FaceDetectInfo> RetinaFace::postProcess(int inputW, int inputH, float threshold)
{
    string name_bbox = "face_rpn_bbox_pred_";
    string name_score = "face_rpn_cls_prob_reshape_";
    string name_landmark = "face_rpn_landmark_pred_";

    vector<FaceDetectInfo> faceInfo;
    for (size_t i = 0; i < _feat_stride_fpn.size(); i++) {
        double s1 = (double)getTickCount();
        string key = "stride" + std::to_string(_feat_stride_fpn[i]);

        string str = name_score + key;
        TrtBlob* score_blob = trtNet->blob_by_name(str);
        std::vector<float> score = score_blob->result[0];
        std::vector<float>::iterator begin = score.begin() + score.size() / 2;
        std::vector<float>::iterator end = score.end();
        score = std::vector<float>(begin, end);

        str = name_bbox + key;
        TrtBlob* bbox_blob = trtNet->blob_by_name(str);
        std::vector<float> bbox_delta = bbox_blob->result[0];

        str = name_landmark + key;
        TrtBlob* landmark_blob = trtNet->blob_by_name(str);
        std::vector<float> landmark_delta = landmark_blob->result[0];
        int width = score_blob->outputDims.d[3];
        int height = score_blob->outputDims.d[2];
        size_t count = width * height;
        size_t num_anchor = _num_anchors[key];

        ///////////////////////////////////////////////
        s1 = (double)getTickCount() - s1;
        std::cout << "s1 compute time :" << s1 * 1000.0 / cv::getTickFrequency() << " ms \n";
        ///////////////////////////////////////////////

        for (size_t num = 0; num < num_anchor; num++) {
            for (size_t j = 0; j < count; j++) {
                float conf = score[j + count * num];
                if (conf <= threshold) {
                    continue;
                }

                cv::Vec4f regress;
                float dx = bbox_delta[j + count * (0 + num * 4)];
                float dy = bbox_delta[j + count * (1 + num * 4)];
                float dw = bbox_delta[j + count * (2 + num * 4)];
                float dh = bbox_delta[j + count * (3 + num * 4)];
                regress = cv::Vec4f(dx, dy, dw, dh);

                anchor_box rect = bbox_pred(_anchors[key][j + count * num], regress);

                clip_boxes(rect, inputW, inputH);

                FacePts pts;
                for (size_t k = 0; k < 5; k++) {
                    pts.x[k] = landmark_delta[j + count * (num * 10 + k * 2)];
                    pts.y[k] = landmark_delta[j + count * (num * 10 + k * 2 + 1)];
                }

                FacePts landmarks = landmark_pred(_anchors[key][j + count * num], pts);

                FaceDetectInfo tmp;
                tmp.score = conf;
                tmp.rect = rect;
                tmp.pts = landmarks;
                faceInfo.push_back(tmp);
            }
        }
    }

    faceInfo = nms(faceInfo, 0.4);

    return faceInfo;
}

std::vector<FaceDetectInfo> RetinaFace::detect(const Mat& img, float threshold, float scales)
{
    if (img.empty()) {
        return std::vector<FaceDetectInfo>();
    }

    int inputW = trtNet->getNetWidth();
    int inputH = trtNet->getNetHeight();

    float scale = 1.0;
    float sw = 1.0 * img.cols / inputW;
    float sh = 1.0 * img.rows / inputH;
    scale = sw > sh ? sw : sh;
    scale = scale > 1.0 ? scale : 1.0;

    cv::Mat resize;
    if (scale > 1) {
        if (sw > sh) {
            cv::resize(img, resize, cv::Size(), 1 / scale, 1 / scale);
            cv::copyMakeBorder(resize, resize, 0, inputH - resize.rows, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0));
        }
        else {
            cv::resize(img, resize, cv::Size(), 1 / scale, 1 / scale);
            cv::copyMakeBorder(resize, resize, 0, 0, 0, inputW - resize.cols, cv::BORDER_CONSTANT, cv::Scalar(0));
        }
    }
    else {
        cv::copyMakeBorder(img, resize, 0, inputH - img.rows, 0, inputW - img.cols, cv::BORDER_CONSTANT, cv::Scalar(0));
    }

    //to float
    resize.convertTo(resize, CV_32FC3);

    //rgb
    cv::cvtColor(resize, resize, COLOR_BGR2RGB);

    vector<Mat> input_channels;
    float* input_data = cpuBuffers;

    for (int i = 0; i < trtNet->getChannel(); ++i) {
        Mat channel(inputH, inputW, CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += inputW * inputH;
    }

    /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the Mat
    * objects in input_channels. */
    split(resize, input_channels);

    float* inputData = (float*)trtNet->getBuffer(0);
    cudaMemcpy(inputData, cpuBuffers, inputW * inputH * 3 * sizeof(float), cudaMemcpyHostToDevice);

    trtNet->doInference(1);

    string name_bbox = "face_rpn_bbox_pred_";
    string name_score = "face_rpn_cls_prob_reshape_";
    string name_landmark = "face_rpn_landmark_pred_";
    vector<FaceDetectInfo> faceInfo;
    for (size_t i = 0; i < _feat_stride_fpn.size(); i++) {
        string key = "stride" + std::to_string(_feat_stride_fpn[i]);
        string str = name_score + key;
        TrtBlob* score_blob = trtNet->blob_by_name(str);
        std::vector<float> score = score_blob->result[0];
        std::vector<float>::iterator begin = score.begin() + score.size() / 2;
        std::vector<float>::iterator end = score.end();
        score = std::vector<float>(begin, end);

        str = name_bbox + key;
        TrtBlob* bbox_blob = trtNet->blob_by_name(str);
        std::vector<float> bbox_delta = bbox_blob->result[0];

        str = name_landmark + key;
        TrtBlob* landmark_blob = trtNet->blob_by_name(str);
        std::vector<float> landmark_delta = landmark_blob->result[0];
        int width = score_blob->outputDims.d[3];
        int height = score_blob->outputDims.d[2];
        size_t count = width * height;
        size_t num_anchor = _num_anchors[key];

        for (size_t num = 0; num < num_anchor; num++) {
            for (size_t j = 0; j < count; j++) {
                float conf = score[j + count * num];
                if (conf <= threshold) {
                    continue;
                }

                cv::Vec4f regress;
                float dx = bbox_delta[j + count * (0 + num * 4)];
                float dy = bbox_delta[j + count * (1 + num * 4)];
                float dw = bbox_delta[j + count * (2 + num * 4)];
                float dh = bbox_delta[j + count * (3 + num * 4)];
                regress = cv::Vec4f(dx, dy, dw, dh);

                anchor_box rect = bbox_pred(_anchors[key][j + count * num], regress);

                clip_boxes(rect, inputW, inputH);

                FacePts pts;
                for (size_t k = 0; k < 5; k++) {
                    pts.x[k] = landmark_delta[j + count * (num * 10 + k * 2)];
                    pts.y[k] = landmark_delta[j + count * (num * 10 + k * 2 + 1)];
                }
                FacePts landmarks = landmark_pred(_anchors[key][j + count * num], pts);

                FaceDetectInfo tmp;
                tmp.score = conf;
                tmp.rect = rect;
                tmp.pts = landmarks;
                tmp.scale_ratio = scale;
                faceInfo.push_back(tmp);
            }
        }
    }
    faceInfo = nms(faceInfo, nms_threshold);
    return faceInfo;
}

vector<vector<FaceDetectInfo>> RetinaFace::detectBatchImages(vector<cv::Mat> imgs, float threshold)
{
    int inputW = trtNet->getNetWidth();
    int inputH = trtNet->getNetHeight();

    vector<float> scales(imgs.size(), 1.0);

    double t2 = (double)getTickCount();
    for (size_t i = 0; i < imgs.size(); i++) {
        float sw = 1.0 * imgs[i].cols / inputW;
        float sh = 1.0 * imgs[i].rows / inputH;
        scales[i] = sw > sh ? sw : sh;
        scales[i] = scales[i] > 1.0 ? scales[i] : 1.0;

        if (sw > 1.0 || sh > 1.0) {
            if (sw > sh) {
                cv::resize(imgs[i], imgs[i], cv::Size(), 1 / sw, 1 / sw);
                cv::copyMakeBorder(imgs[i], imgs[i], 0, inputH - imgs[i].rows, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0));
            }
            else {
                cv::resize(imgs[i], imgs[i], cv::Size(), 1 / sh, 1 / sh);
                cv::copyMakeBorder(imgs[i], imgs[i], 0, 0, 0, inputW - imgs[i].cols, cv::BORDER_CONSTANT, cv::Scalar(0));
            }
        }
        else {
            
            cv::copyMakeBorder(imgs[i], imgs[i], 0, inputH - imgs[i].rows, 0, inputW - imgs[i].cols, cv::BORDER_CONSTANT, cv::Scalar(0));
        }

        //to float
        imgs[i].convertTo(imgs[i], CV_32FC3);

        //rgb
        cv::cvtColor(imgs[i], imgs[i], COLOR_BGR2RGB);
    }

    vector<vector<Mat>> input_channels;
    float* input_data = (float*)cpuBuffers;
    for (size_t j = 0; j < imgs.size(); j++) {
        vector<Mat> input_chans;
        for (int i = 0; i < trtNet->getChannel(); ++i) {
            Mat channel(inputH, inputW, CV_32FC1, input_data);
            input_chans.push_back(channel);
            input_data += inputW * inputH;
        }
        input_channels.push_back(input_chans);
    }

    /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the Mat
    * objects in input_channels. */
    for (size_t j = 0; j < imgs.size(); j++) {
        split(imgs[j], input_channels[j]);
    }

    float* inputData = (float*)trtNet->getBuffer(0);
    cudaMemcpy(inputData, cpuBuffers, imgs.size() * inputW * inputH * 3 * sizeof(float), cudaMemcpyHostToDevice);

    t2 = (double)getTickCount() - t2;

    double t1 = (double)getTickCount();
    trtNet->doInference(imgs.size());
    t1 = (double)getTickCount() - t1;


    double post = (double)getTickCount();
    string name_bbox = "face_rpn_bbox_pred_";
    string name_score = "face_rpn_cls_prob_reshape_";
    string name_landmark = "face_rpn_landmark_pred_";

    vector<vector<FaceDetectInfo>> faceInfos;
    for (size_t batch = 0; batch < imgs.size(); batch++) {
        vector<FaceDetectInfo> faceInfo;
        for (size_t i = 0; i < _feat_stride_fpn.size(); i++) {
            string key = "stride" + std::to_string(_feat_stride_fpn[i]);
            string str = name_score + key;
            TrtBlob* score_blob = trtNet->blob_by_name(str);
            std::vector<float> score = score_blob->result[batch];
            std::vector<float>::iterator begin = score.begin() + score.size() / 2;
            std::vector<float>::iterator end = score.end();
            score = std::vector<float>(begin, end);

            str = name_bbox + key;
            TrtBlob* bbox_blob = trtNet->blob_by_name(str);
            std::vector<float> bbox_delta = bbox_blob->result[batch];

            str = name_landmark + key;
            TrtBlob* landmark_blob = trtNet->blob_by_name(str);
            std::vector<float> landmark_delta = landmark_blob->result[batch];

            int width = score_blob->outputDims.d[3];
            int height = score_blob->outputDims.d[2];
            size_t count = width * height;
            size_t num_anchor = _num_anchors[key];

            for (size_t num = 0; num < num_anchor; num++) {
                for (size_t j = 0; j < count; j++) {
                    float conf = score[j + count * num];
                    if (conf <= threshold) {
                        continue;
                    }

                    cv::Vec4f regress;
                    float dx = bbox_delta[j + count * (0 + num * 4)];
                    float dy = bbox_delta[j + count * (1 + num * 4)];
                    float dw = bbox_delta[j + count * (2 + num * 4)];
                    float dh = bbox_delta[j + count * (3 + num * 4)];
                    regress = cv::Vec4f(dx, dy, dw, dh);

                    anchor_box rect = bbox_pred(_anchors[key][j + count * num], regress);

                    clip_boxes(rect, inputW, inputH);

                    FacePts pts;
                    for (size_t k = 0; k < 5; k++) {
                        pts.x[k] = landmark_delta[j + count * (num * 10 + k * 2)];
                        pts.y[k] = landmark_delta[j + count * (num * 10 + k * 2 + 1)];
                    }
                    FacePts landmarks = landmark_pred(_anchors[key][j + count * num], pts);

                    FaceDetectInfo tmp;
                    tmp.score = conf;
                    tmp.rect = rect;
                    tmp.pts = landmarks;
                    tmp.scale_ratio = scales[i];
                    faceInfo.push_back(tmp);
                }
            }
        }

        faceInfos.push_back(faceInfo);
    }
    for (size_t batch = 0; batch < imgs.size(); batch++) {
        faceInfos[batch] = nms(faceInfos[batch], nms_threshold);
    }

    post = (double)getTickCount() - post;
    return faceInfos;
}