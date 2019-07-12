cmake_minimum_required(VERSION 3.1)
project(XFusionLib VERSION 0.1 LANGUAGES CUDA CXX)

find_package(CUDA 10.1 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(OpenCV 3.4 REQUIRED)

add_library(XFusion SHARED "")

target_sources(XFusion
PRIVATE
    # core modules
    src/xfusion/core/cuda_imgproc.cu
    src/xfusion/core/device_malloc.cpp
    src/xfusion/core/build_pyramid.cpp

    # icp
    src/xfusion/icp/estimator.cu
    src/xfusion/icp/m_estimator.cu

    # voxel mapping
    src/xfusion/mapping/depth_fusion.cu
    src/xfusion/mapping/map_struct.cu
    src/xfusion/mapping/ray_tracing.cu
    src/xfusion/mapping/scene_mesh.cu
)

target_include_directories(XFusion
PUBLIC
    $<INSTALL_INTERFACE:include>   
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(XFusion 
PRIVATE 
    cxx_std_11
)

set_target_properties(XFusion
PROPERTIES 
    CUDA_STANDARD 11
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION}
)

target_link_directories(XFusion
PRIVATE
    Eigen3::Eigen
    ${OpenCV_LIBRARIES}
    ${CUDA_LIBRARIES}
)

install(TARGETS XFusion
        EXPORT XFusion-targets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        CONFIGURATIONS Release)

install(EXPORT XFusion-targets
        NAMESPACE XFusion::
        FILE XFusionConfig.cmake
        DESTINATION lib/cmake/xfusion)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include
        DESTINATION ${CMAKE_INSTALL_PREFIX})
        
#for unit test
add_executable(test "")

target_sources(test
PRIVATE
    test/unitest.cu
)