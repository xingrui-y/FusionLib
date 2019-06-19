cmake_minimum_required(VERSION 3.1)
project(FusionLIB VERSION 0.1 LANGUAGES CUDA CXX)

find_package(CUDA 10.1 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(OpenCV 3.4 REQUIRED)

add_library(fusion STATIC "")

target_sources(fusion
PRIVATE
    # core modules
    src/fusion/core/cuda_imgproc.cu
    src/fusion/core/intrinsic_matrix.cpp

    # icp
    src/fusion/icp/estimator.cu
    src/fusion/icp/m_estimator.cu

    # voxel mapping
    src/fusion/mapping/depth_fusion.cu
    src/fusion/mapping/map_struct.cu
    src/fusion/mapping/ray_tracing.cu
    src/fusion/mapping/scene_mesh.cu
)

target_include_directories(fusion
PUBLIC
    $<INSTALL_INTERFACE:include>   
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(fusion 
PRIVATE 
    cxx_std_11
)

set_target_properties(fusion
PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_STANDARD 11
)

target_link_directories(fusion
PRIVATE
    Eigen3::Eigen
    ${OpenCV_LIBRARIES}
    ${CUDA_LIBRARIES}
)

install(TARGETS fusion
        EXPORT fusion-targets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        CONFIGURATIONS Release)

install(EXPORT fusion-targets
        NAMESPACE fusion::fusion
        FILE fusion-config.cmake
        DESTINATION lib/cmake/fusion)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include
        DESTINATION ${CMAKE_INSTALL_PREFIX})