
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/3rdparty/TensorRT/include)
link_directories(${PROJECT_SOURCE_DIR}/3rdparty/TensorRT/lib)

include_directories(${PROJECT_SOURCE_DIR}/3rdparty/TensorRT/samples/common/)
include_directories(${PROJECT_SOURCE_DIR}/3rdparty/TensorRT/samples/common/windows/)

file(GLOB_RECURSE srcs ${PROJECT_SOURCE_DIR}/3rdparty/TensorRT/samples/common/*.cpp)
list(APPEND srcs ${PROJECT_SOURCE_DIR}/3rdparty/TensorRT/samples/common/windows/getopt.c)

include_directories(${3RDPARTY_INCLUDE_DIRS})
link_directories(${3RDPARTY_LIBS})
include_directories(${GLOG_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/3rdparty/glog/include)
link_directories(${PROJECT_SOURCE_DIR}/3rdparty/glog/lib)

set(plugin_list retinaface)
foreach(plugin ${plugin_list})
  include_directories(${PROJECT_SOURCE_DIR}/${plugin}/include)
  file(GLOB_RECURSE plugin_src ${PROJECT_SOURCE_DIR}/${plugin}/*.cpp)
  list(APPEND srcs ${plugin_src})
endforeach()

add_library(easysa_plugins ${srcs})
target_link_libraries(easysa_plugins easysa_components ${Opencv_LIBS} glogd ${CUDA_LIBRARIES} nvinfer nvinfer_plugin nvonnxparser nvparsers)
add_subdirectory(unitest)