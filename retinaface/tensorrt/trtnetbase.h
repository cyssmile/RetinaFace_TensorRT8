#ifndef TRTNETBASE_H
#define TRTNETBASE_H

#include <fstream>
#include <vector>
#include <map>
#include <iostream>
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"

using namespace nvinfer1;
using namespace nvonnxparser;
using namespace std;

class Logger;
class Profiler;

class TrtNetBase
{
public:
   /**
    *	@brief  getBatchSize	        
    *   @return                         
    *
    *   @note
    */
    uint32_t getMaxBatchSize() const;

   /**
    *	@brief  getBatchSize	        
    *   @return                         
    *
    *   @note
    */
    uint32_t getBatchSize() const;
	   
   /**
    *	@brief  getNetWidth	            
    *   @return                         
    *
    *   @note
    */
    int getNetWidth() const;
	
   /**
    *	@brief  getNetHeight	        
    *   @return                         
    *
    *   @note
    */
    int getNetHeight() const;
	
   /**
    *	@brief  getNetHeight	        
    *   @return                         
    *
    *   @note
    */
    int getChannel() const;

    // Buffer is allocated in TRT_Conxtex,
    // Expose this interface for inputing data
   /**
    *	@brief  getBuffer	            
    *   @param  index		            
    *   @return null
    *
    *   @note					        
    */
    void*& getBuffer(const int& index);
	
   /**
    *	@brief  getBuffer	            获取CPU Buffer输入地址
    *   @return                         返回地址指针
    *
    *   @note					        
    */
    float*& getInputBuf();

   /**
    *	@brief  setForcedFp32	        是否使用32位浮点计算
    *   @param  forcedFp32		        true表示是，false表示使用fp16
    *   @return 
    *
    *   @note					        
    */
    void setForcedFp32(const bool& forcedFp32);
	
    void setDumpResult(const bool& dumpResult);
	
   /**
    *	@brief  setTrtProfilerEnabled	        是否启用性能测试
    *   @param  enableTrtProfiler		        true表示是，false表示否
    *   @return 
    *
    *   @note					        
    */
    void setTrtProfilerEnabled(const bool& enableTrtProfiler);

    TrtNetBase(std::string netWorkName);
    virtual ~TrtNetBase();

   /**
    *	@brief  buildTrtContext	         创建tensorRT上下文环境
    *   @param  cachefile		         序列化引擎模型文件
	*   @param  modelfile		         模型文件
	*   @param  bUseCPUBuf		         使用CPU buffer
    *   @return 
    *
    *   @note					        
    */
    void buildTrtContext(const std::string &modelfile, const std::string &cachefile, bool bUseCPUBuf = false);
	
   /**
    *	@brief  destroyTrtContext	     销毁tensorRT上下文环境
	*   @param  bUseCPUBuf		         使用CPU buffer
    *   @return 
    *
    *   @note					        
    */
    void destroyTrtContext(bool bUseCPUBuf = false);

    /**
     *	 @brief  doInference	        TensorRT推理函数
     *   @param  batchSize		        批量数
     *   @param  confs		            返回置信度
     *   @param  regBoxes		        返回回归框
     *   @param  landMarks		        返回关键点
     *   @param  input		            数据输入
     *   @return
     *
     *   @note
     */
    virtual void doInference(int batchSize, float *input = NULL) = 0;

private:

   /**
    *	@brief  OnnxToTRTModel	         将Caffe模型转为TensorRT模型
	*   @param  deployfile		         模型文件
	*   @param  modelfile		         模型文件
	*   @param  pluginFactory		     插件工厂
    *   @return 
    *
    *   @note					        
    */
    void OnnxToTRTModel(const std::string& modelfile, const std::string& cachefile);
						 
   /**
    *	@brief  parseNet	             解析网络文件
    *   @return 
    *
    *   @note					        
    */
    bool parseNet(const std::string &deployfile);
	
   /**
    *	@brief  allocateMemory	         开辟内存空间
    *   @param  bUseCPUBuf		         使用CPU buffer
    *   @return 
    *
    *   @note					         
    */
    virtual void allocateMemory(bool bUseCPUBuf) = 0;
	   
   /**
    *	@brief  releaseMemory	         释放内存空间
    *   @param  bUseCPUBuf		         使用CPU buffer
    *   @return 
    *
    *   @note					         
    */
    virtual void releaseMemory(bool bUseCPUBuf) = 0;
	
   /**
    *	@brief  stringtrim	             字符串处理函数
    *   @param  s		                 输入字符串
    *   @return 
    *
    *   @note					         
    */
    std::string stringtrim(std::string s);

protected:
    Logger *pLogger;
    Profiler *profiler;
    IRuntime *runtime;
    ICudaEngine *engine;
    IExecutionContext *context;
    IHostMemory *trtModelStream{nullptr};
    bool useFp32;
    std::size_t workSpaceSize;
    unsigned int maxBatchSize = 1;

    int batchSize;
    int channel;
    int netWidth;
    int netHeight;

    std::vector<std::string> outputs;
    int numBinding;
    float *inputBuffer;
    void  **buffers;
    bool dumpResult;
    bool enableTrtProfiler;
    std::ofstream fstream;
    std::string resultFile;
    std::string netWorkName;
};

#endif // TRTNETBASE_H
