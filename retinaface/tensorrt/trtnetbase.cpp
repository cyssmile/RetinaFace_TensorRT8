#include "trtnetbase.h"
#include "trtutility.h"
#include <assert.h>
#include <iterator>
#include <memory>
#include <string>
#include <sstream>

using namespace std;

//This function is used to trim space
string TrtNetBase::stringtrim(string s)
{
    int i = 0;
    while (s[i] == ' ') {
        i++;
    }
    s = s.substr(i);
    i = s.size() - 1;
    while (s[i] == ' ') {
        i--;
    }

    s = s.substr(0, i + 1);
    return s;
}

uint32_t TrtNetBase::getBatchSize() const
{
    return batchSize;
}

uint32_t TrtNetBase::getMaxBatchSize() const
{
    return maxBatchSize;
}

int TrtNetBase::getNetWidth() const
{
    return netWidth;
}

int TrtNetBase::getNetHeight() const
{
    return netHeight;
}

int TrtNetBase::getChannel() const
{
    return channel;
}

void*& TrtNetBase::getBuffer(const int& index)
{
    assert(index >= 0 && index < numBinding);
    return buffers[index];
}

float*& TrtNetBase::getInputBuf()
{
    return inputBuffer;
}

void TrtNetBase::setForcedFp32(const bool& forcedFp32)
{
    useFp32 = forcedFp32;
}

void TrtNetBase::setDumpResult(const bool& dumpResult)
{
    this->dumpResult = dumpResult;
}

void TrtNetBase::setTrtProfilerEnabled(const bool& enableTrtProfiler)
{
    this->enableTrtProfiler = enableTrtProfiler;
}

TrtNetBase::TrtNetBase(string netWorkName)
{
    pLogger = new Logger();
    profiler = new Profiler();
    runtime = NULL;
    engine = NULL;
    context = NULL;

    batchSize = 0;
    channel = 0;
    netWidth = 0;
    netHeight = 0;

    useFp32 = false;

    dumpResult = false;
    resultFile = "result.txt";
    enableTrtProfiler = false;
    this->netWorkName = netWorkName;
}

TrtNetBase::~TrtNetBase()
{
    delete pLogger;
    delete profiler;
}
/*
*   no use this function.its for caffe parser
*/
bool TrtNetBase::parseNet(const string& deployfile)
{
    return true;
}

void TrtNetBase::buildTrtContext(const std::string& modelfile, const std::string& cachefile, bool bUseCPUBuf)
{  
    ifstream trtModelFile(cachefile, std::ios_base::in | std::ios_base::binary);
    if (trtModelFile.good()) {
        // get cache file length
        size_t size = 0;
        size_t i = 0;
        printf("Using cached tensorRT model.\n");
        // Get the length
        trtModelFile.seekg(0, ios::end);
        size = trtModelFile.tellg(); // 3564925
        trtModelFile.seekg(0, ios::beg);

        char* buff = new char[size];
        trtModelFile.read(buff, size);

        trtModelFile.close();
        runtime = createInferRuntime(*pLogger);
        engine = runtime->deserializeCudaEngine((void*)buff, size, NULL);
        delete buff;
    }
    else {
        OnnxToTRTModel(modelfile, cachefile);
        printf("Create tensorRT model cache.\n");
        ofstream trtModelFile(cachefile, std::ios_base::out | std::ios_base::binary);
        trtModelFile.write((char*)trtModelStream->data(), trtModelStream->size());
        trtModelFile.close();
        runtime = createInferRuntime(*pLogger);
        engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), NULL);
        trtModelStream->destroy();
    }
    context = engine->createExecutionContext();
    context->setProfiler(profiler);
    allocateMemory(bUseCPUBuf);
}

void TrtNetBase::destroyTrtContext(bool bUseCPUBuf)
{
    releaseMemory(bUseCPUBuf);
    context->destroy();
    engine->destroy();
    runtime->destroy();
}
/*
*   Now I just use onnx to TrtModel
*/
void TrtNetBase::OnnxToTRTModel(const std::string& modelfile, const std::string& cachefile) {
    IBuilder* builder = nullptr;
    nvinfer1::INetworkDefinition* network = nullptr;
    nvonnxparser::IParser* parser = nullptr;
    IBuilderConfig* config = nullptr;
    nvinfer1::IOptimizationProfile* profile = nullptr;
    ICudaEngine* engine = nullptr;
    bool ret = false;
    do {
        builder = createInferBuilder(*pLogger);
        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        network = builder->createNetworkV2(explicitBatch);
        parser = nvonnxparser::createParser(*network, *pLogger);
        ret = parser->parseFromFile(modelfile.c_str(), static_cast<int>(ILogger::Severity::kWARNING));
        if (!ret) {
            for (int i = 0; i < parser->getNbErrors(); i++) {
                std::cout << parser->getError(i)->desc();
            }
            std::cout << "parser retinaface Model failed" << std::endl;
            break;
        }
        builder->setMaxBatchSize(1);
        config = builder->createBuilderConfig();
        config->setMaxWorkspaceSize(1 << 20);
        if (!builder->isNetworkSupported(*network, *config)) {
            std::cout << "this network not support!" << std::endl;
            break;
        }
        profile = builder->createOptimizationProfile();
        for (int i = 0; i < network->getNbInputs(); ++i) {
            auto* input = network->getInput(i);
            Dims d = input->getDimensions();
            channel = d.d[1];
            netHeight = d.d[2];
            netWidth = d.d[3];
            const char* input_name = input->getName();
            profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, Dims4{ 1,d.d[1],d.d[2],d.d[3] });
            profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, Dims4{ 1,d.d[1],d.d[2],d.d[3] });
            profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, Dims4{ 1,d.d[1],d.d[2],d.d[3] });
        }
        config->addOptimizationProfile(profile);
        ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
        if (!engine) {
            break;
        }
        trtModelStream = engine->serialize();
        engine->destroy();
        network->destroy();
        parser->destroy();
        config->destroy();
        builder->destroy();
        return;
    } while (0);
    // load model failed
    if (network) {
        network->destroy();
    }
    if (config) {
        config->destroy();
    }
    if (parser) {
        parser->destroy();
    }
    if (builder) {
        builder->destroy();
    }
}