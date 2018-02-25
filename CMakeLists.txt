cmake_minimum_required(VERSION 3.1)
set(project_name "danknet")
PROJECT(${project_name})

add_definitions( -std=c++11 )

set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")

 find_package(OpenMP)
 if (OPENMP_FOUND)
     set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
     set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
 endif()


set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(HDF5 COMPONENTS C CXX HL REQUIRED)
#find_package(OpenCV REQUIRED)

include_directories(
    ${HDF5_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/danknet
    ${CMAKE_CURRENT_SOURCE_DIR}/include/danknet
    ${CMAKE_CURRENT_SOURCE_DIR}/src/danknet/layers
    ${CMAKE_CURRENT_SOURCE_DIR}/include/danknet/layers
    ${CMAKE_CURRENT_BINARY_DIR}
)


set(SOURCES

    src/danknet/net.cpp
    src/danknet/layer.cpp
    src/danknet/solver.cpp

    src/danknet/layers/data_layer.cpp
    src/danknet/layers/image_data_layer.cpp
    src/danknet/layers/convolutional_layer.cpp
    src/danknet/layers/fully_connected_layer.cpp
    src/danknet/layers/pooling_layer.cpp
    src/danknet/layers/loss_layer.cpp
    src/danknet/layers/softmax_layer.cpp

)
set(HEADERS
    include/danknet/data.h
    include/danknet/data2d.h

    include/danknet/danknet.h
    include/danknet/common.h

    include/danknet/net.h
    include/danknet/layer.h
    include/danknet/solver.h
    include/danknet/params.h

    include/danknet/layers/data_layer.h
    include/danknet/layers/image_data_layer.h
    include/danknet/layers/convolutional_layer.h
    include/danknet/layers/fully_connected_layer.h
    include/danknet/layers/pooling_layer.h
    include/danknet/layers/loss_layer.h
    include/danknet/layers/softmax_layer.h
)

##main project
#add_executable(${project_name} main.cpp ${SOURCES} ${HEADERS})
#target_link_libraries(${project_name} Qt5::Core Qt5::Gui ${HDF5_CXX_LIBRARIES}) #${OpenCV_LIBS})

##xor project
#add_executable(xor xor.cpp ${SOURCES} ${HEADERS})
#target_link_libraries(xor Qt5::Core Qt5::Gui ${HDF5_CXX_LIBRARIES})

##edge_detector project
#add_executable(edge_detector edge_detector.cpp ${SOURCES} ${HEADERS})
#target_link_libraries(edge_detector Qt5::Core Qt5::Gui ${HDF5_CXX_LIBRARIES})

##edge_detector_train project
#add_executable(edge_detector_train edge_detector_train.cpp ${SOURCES} ${HEADERS})
#target_link_libraries(edge_detector_train Qt5::Core Qt5::Gui ${HDF5_CXX_LIBRARIES})

##pred_data project
#add_executable(pred_data pred_data.cpp ${SOURCES} ${HEADERS})
#target_link_libraries(pred_data Qt5::Core Qt5::Gui ${HDF5_CXX_LIBRARIES})

##pred_data_train project
#add_executable(pred_data_train pred_data_train.cpp ${SOURCES} ${HEADERS})
#target_link_libraries(pred_data_train Qt5::Core Qt5::Gui ${HDF5_CXX_LIBRARIES})

#bagging project
add_executable(bagging bagging.cpp ${SOURCES} ${HEADERS})
target_link_libraries(bagging Qt5::Core Qt5::Gui ${HDF5_CXX_LIBRARIES})

#bagging_train project
add_executable(bagging_train bagging_train.cpp ${SOURCES} ${HEADERS})
target_link_libraries(bagging_train Qt5::Core Qt5::Gui ${HDF5_CXX_LIBRARIES})
