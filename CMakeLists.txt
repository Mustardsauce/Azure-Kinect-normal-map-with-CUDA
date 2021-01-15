cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(AzureKinectNormalMapCuda VERSION 1.0.0)

if(WIN32)
    # For Azure Kinect Sensor SDK Installer
    # find_package(k4a) is search library using cmake/Findk4a.cmake.
    set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
endif()

find_package(k4a REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

if(k4a_FOUND AND CUDA_FOUND AND OpenCV_FOUND)
    cuda_add_executable(cuda_normal_map main.cpp cuda_normal.cu)
    target_link_libraries(cuda_normal_map k4a::k4a)
    target_link_libraries(cuda_normal_map ${OpenCV_LIBS})
    target_link_libraries(cuda_normal_map ${CUDA_LIBRARIES})
endif()
