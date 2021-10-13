option(build_plugins_test "build plugins unitest" ON)

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

include_directories(${PROJECT_SOURCE_DIR}/3rdparty/googletest/include)
link_directories(${PROJECT_SOURCE_DIR}/3rdparty/googletest/lib)

set(test_plugins_srcs "")
list(APPEND test_plugins_srcs ${PROJECT_SOURCE_DIR}/unitest/test_plugins.cpp)

option(build_retinaface_test "build face detection plugin unitest" ON)

if(build_retinaface_test)
	include_directories(${PROJECT_SOURCE_DIR}/retinaface)
	include_directories(${PROJECT_SOURCE_DIR}/retinaface/tensorrt)
	file(GLOB_RECURSE test_retinaface_src ${PROJECT_SOURCE_DIR}/unitest/retinaface/*.cpp )
	list(APPEND test_plugins_srcs ${test_retinaface_src})
endif()

cuda_add_executable(test_plugins ${srcs} ${test_plugins_srcs})
target_link_libraries(test_plugins easysa_plugins gtestd glogd ${OpenCV_LIBS})
