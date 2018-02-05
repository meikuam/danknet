cmake_minimum_required(VERSION 3.1)
set(project_name "danknet")
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
    ${CMAKE_CURRENT_SOURCE_DIR}/src/danknet
    ${CMAKE_CURRENT_SOURCE_DIR}/include/danknet
    ${CMAKE_CURRENT_SOURCE_DIR}/src/danknet/layers
    ${CMAKE_CURRENT_SOURCE_DIR}/include/danknet/layers
    ${CMAKE_CURRENT_BINARY_DIR}
)

find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)

set(SOURCES
    main.cpp

    src/danknet/net.cpp
    src/danknet/layer.cpp
    src/danknet/solver.cpp

    src/danknet/layers/convolutional_layer.cpp
    src/danknet/layers/fully_connected_layer.cpp
    src/danknet/layers/pooling_layer.cpp
    src/danknet/layers/loss_layer.cpp
)
set(HEADERS
    include/danknet/data.h

    include/danknet/danknet.h
    include/danknet/common.h

    include/danknet/net.h
    include/danknet/layer.h
    include/danknet/solver.h
    include/danknet/params.h

    include/danknet/layers/convolutional_layer.h
    include/danknet/layers/fully_connected_layer.h
    include/danknet/layers/pooling_layer.h
    include/danknet/layers/loss_layer.h
)

add_executable(${project_name} ${SOURCES} ${HEADERS})
target_link_libraries(${project_name} Qt5::Core Qt5::Gui)
