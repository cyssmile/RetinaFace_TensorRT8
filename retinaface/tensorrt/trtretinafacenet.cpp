#include "trtretinafacenet.h"
#include "trtutility.h"
#include <map>
#include <assert.h>
#include<functional>

using namespace std;

TrtRetinaFaceNet::TrtRetinaFaceNet(string netWorkName) : TrtNetBase(netWorkName)
{
    numBinding = 10;
    buffers = new void* [numBinding];
    for (int i = 0; i < numBinding; i++)
    {
        buffers[i] = NULL;
    }

    inputBuffer = NULL;

    workSpaceSize = 1 << 24;
    outputs = { "face_rpn_cls_prob_reshape_stride32",
               "face_rpn_bbox_pred_stride32",
               "face_rpn_landmark_pred_stride32",
               "face_rpn_cls_prob_reshape_stride16",
               "face_rpn_bbox_pred_stride16",
               "face_rpn_landmark_pred_stride16",
               "face_rpn_cls_prob_reshape_stride8",
               "face_rpn_bbox_pred_stride8",
               "face_rpn_landmark_pred_stride8" };

    for (size_t i = 0; i < outputs.size(); i++) {
        outputBuffers.push_back(NULL);
    }

    results.resize(outputs.size());
    for (size_t i = 0; i < outputs.size(); i++) {
        results[i].layer_name = outputs[i];
    }
}

TrtRetinaFaceNet::~TrtRetinaFaceNet()
{
    delete[]buffers;
}

void TrtRetinaFaceNet::doInference(int batchSize, float* input)
{
    if (!enableTrtProfiler) {
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));
        // DMA the input to the GPU,  execute the batch asynchronously
        // and DMA it back
        if (input != NULL) {  //NULL means we have use GPU to map memory
            //inputSize = batchSize * inputDims.c() * inputDims.h() * inputDims.w() * sizeof(float);
            inputSize = batchSize * inputDims.d[1] * inputDims.d[2] * inputDims.d[3] * sizeof(float);
            CHECK(cudaMemcpyAsync(buffers[inputIndex], input, inputSize, cudaMemcpyDeviceToDevice, stream));
        }
        context->enqueue(batchSize, buffers, stream, nullptr);

        for (size_t i = 0; i < outputDims.size(); i++) {
            CHECK(cudaMemcpyAsync(outputBuffers[i], buffers[outputIndexs[i]],
                batchSize * outputsizes[i] / maxBatchSize,
                cudaMemcpyDeviceToHost, stream));
        }

        cudaStreamSynchronize(stream);
        // release the stream and the buffers
        cudaStreamDestroy(stream);
    }
    else {
        // DMA the input to the GPU, execute the batch synchronously
        // and DMA it back
        if (input != NULL) {   //NULL means we have use GPU to map memory
            //inputSize = batchSize * inputDims.c() * inputDims.h() * inputDims.w() * sizeof(float);
            inputSize = batchSize * inputDims.d[1] * inputDims.d[2] * inputDims.d[3] * sizeof(float);
            CHECK(cudaMemcpy(buffers[inputIndex], input, inputSize, cudaMemcpyDeviceToDevice));
        }

        context->execute(batchSize, buffers);

        for (size_t i = 0; i < outputDims.size(); i++) {
            CHECK(cudaMemcpyAsync(outputBuffers[i], buffers[outputIndexs[i]], outputsizes[i], cudaMemcpyDeviceToHost));
        }
    }

    //output to vector
    for (size_t i = 0; i < outputBuffers.size(); i++) {
        int count = outputsizes[i] / (sizeof(float) * maxBatchSize);
        results[i].result.clear();
        for (int j = 0; j < batchSize; j++) {
            const float* confidenceBegin = outputBuffers[i] + j * count;
            const float* confidenceEnd = confidenceBegin + count;
            std::vector<float> ret = std::vector<float>(confidenceBegin, confidenceEnd);
            results[i].result.push_back(ret);
        }
        results[i].batchsize = batchSize;
    }
}

TrtBlob* TrtRetinaFaceNet::blob_by_name(string layer_name)
{
    auto str_equal = [&](TrtBlob trt_blob) { return (trt_blob.layer_name == layer_name); };
    vector<TrtBlob>::iterator it = std::find_if(results.begin(), results.end(), str_equal);

    if (it == results.end()) {
        return NULL;
    }

    return &(*it);
}

vector<int> TrtRetinaFaceNet::getOutputWidth()
{
    if (outputDims.size() == 0) {
        return vector<int>();
    }
    vector<int> out;
    out.push_back(outputDims[0].d[3]);
    out.push_back(outputDims[3].d[3]);
    out.push_back(outputDims[6].d[3]);

    return out;
}

vector<int> TrtRetinaFaceNet::getOutputHeight()
{
    if (outputDims.size() == 0) {
        return vector<int>();
    }

    vector<int> out;
    out.push_back(outputDims[0].d[2]);
    out.push_back(outputDims[3].d[2]);
    out.push_back(outputDims[6].d[2]);
    return out;
}

void TrtRetinaFaceNet::allocateMemory(bool bUseCPUBuf)
{
    const ICudaEngine& cudaEngine = context->getEngine();
    // input and output buffer pointers that we pass to the engine
    // the engine requires exactly IEngine::getNbBindings() of these
    // but in this case we know that there is exactly one input and one output
    assert(cudaEngine.getNbBindings() == numBinding);

    // In order to bind the buffers, we need to know the names of the input
    // and output tensors. note that indices are guaranteed to be less than
    // IEngine::getNbBindings()
    inputIndex = cudaEngine.getBindingIndex(inputBlobName.c_str());
    Dims input_dim = context->getBindingDimensions(inputIndex);
    netWidth = input_dim.d[3];
    netHeight = input_dim.d[2];
    channel = input_dim.d[1];
    outputIndexs.resize(outputs.size());

    for (size_t i = 0; i < outputs.size(); i++) {
        int idx = cudaEngine.getBindingIndex(outputs[i].c_str());
        outputIndexs[i] = idx;
        results[i].layer_index = idx;
    }

    // allocate GPU buffers
    // inputDims = static_cast<DimsCHW&&>(cudaEngine.getBindingDimensions(inputIndex));
    inputDims = static_cast<Dims4&&>(cudaEngine.getBindingDimensions(inputIndex));
    outputDims.resize(outputIndexs.size());
    for (size_t i = 0; i < outputIndexs.size(); i++) {
        Dims4 dim = static_cast<Dims4&&>(cudaEngine.getBindingDimensions(outputIndexs[i]));
        outputDims[i] = dim;
        results[i].outputDims = dim;
    }
    inputSize = maxBatchSize * inputDims.d[1] * inputDims.d[2] * inputDims.d[3] * sizeof(float);
    //inputSize = inputDims.d[1] * inputDims.d[2] * inputDims.d[3] * sizeof(float);
    outputsizes.resize(outputDims.size());
    for (size_t i = 0; i < outputDims.size(); i++) {
        size_t sz = maxBatchSize * outputDims[i].d[1] * outputDims[i].d[2] * outputDims[i].d[3] * sizeof(float);
        outputsizes[i] = sz;
        results[i].outputSize = sz;
    }

    if (bUseCPUBuf && inputBuffer == NULL) {
        inputBuffer = (float*)malloc(inputSize);
        assert(inputBuffer != NULL);
    }

    for (size_t i = 0; i < outputBuffers.size(); i++) {
        if (outputBuffers[i] == NULL) {
            outputBuffers[i] = (float*)malloc(outputsizes[i]);
            assert(outputBuffers[i] != NULL);
        }
    }

    // create GPU buffers and a stream
    if (buffers[inputIndex] == NULL) {
        CHECK(cudaMalloc(&buffers[inputIndex], inputSize));
    }

    for (size_t i = 0; i < outputIndexs.size(); i++) {
        if (buffers[outputIndexs[i]] == NULL) {
            CHECK(cudaMalloc(&buffers[outputIndexs[i]], outputsizes[i]));
        }
    }

    if (dumpResult) {
        fstream.open(resultFile.c_str(), ios::out);
    }
}

void TrtRetinaFaceNet::releaseMemory(bool bUseCPUBuf)
{
    for (int i = 0; i < numBinding; i++) {
        if (buffers[i] != NULL) {
            CHECK(cudaFree(buffers[i]));
            buffers[i] = NULL;
        }
    }

    if (bUseCPUBuf && inputBuffer != NULL) {
        free(inputBuffer);
        inputBuffer = NULL;
    }

    for (size_t i = 0; i < outputBuffers.size(); i++) {
        if (outputBuffers[i] != NULL) {
            free(outputBuffers[i]);
            outputBuffers[i] = NULL;
        }
    }

    if (dumpResult) {
        fstream.close();
    }
}