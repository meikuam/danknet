cmake_minimum_required(VERSION 3.1)
set(project_name "edge_detector")
PROJECT(${project_name})

add_definitions( -std=c++11 )

# find_package(OpenMP)
# if (OPENMP_FOUND)
#     set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#     set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# endif()


set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

find_package(Qt5Core REQUIRED)

set(SOURCES
    main.cpp
)
set(HEADERS
    data2d.h
)

add_executable(${project_name} ${SOURCES} ${HEADERS})
target_link_libraries(${project_name} Qt5::Core)
