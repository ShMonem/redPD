cmake_minimum_required(VERSION 3.15)

project(doubleHRPD)# LANGUAGE CXX)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(libigl)
igl_include(glfw)
igl_include(opengl)
igl_include(png)

#find_package(IGL REQUIRED)

# Edit files to build here.
set(SRCS 
  src/doubleHRPD.cpp
  src/glad.c
  src/predicates.cxx
  src/ProjDynConstraints.cpp
  src/ProjDynMeshSampler.cpp
  src/ProjDynRHSInterpol.cpp
  src/ProjDynSimulator.cpp
  src/ProjDynTetGen.cpp
  src/ProjDynUtil.cpp
  src/StopWatch.cpp
  src/STVD.cpp
  src/tetgen.cxx
)

#set(GLFW_LIBRARIES
#  glfw3.lib
#
#)

#set(GLAD_LIBRARIES
#  glad.lib


	FIND_PACKAGE(Eigen3 3.3 REQUIRED NO_MODULE)
FIND_PACKAGE(OpenMP REQUIRED)
#FIND_PACKAGE(SuiteSparse CONFIG REQUIRED )
#set(SuiteSparse_DIR "/home/shaimaa/SuitSparse/SuiteSparse/") 

#add_definitions(-D_CRT_SECURE_NO_WARNINGS) 
add_executable(${PROJECT_NAME}_bin ${SRCS})

#include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")

#find_package(GLFW REQUIRED)
#find_package(GLAD REQUIRED)
#include_directories(${PROJECT_NAME} PUBLIC "../../external/glad/include/")
#include_directories(${PROJECT_NAME} PUBLIC "../../external/glfw/include/")

#link_libraries(${GLFW_LIBRARIES} ${GLAD_LIBRARIES})

#target_link_libraries(${PROJECT_NAME}_bin OpenMP::OpenMP_CXX igl::core igl::opengl igl::png igl::opengl_glfw)
target_link_libraries(${PROJECT_NAME}_bin PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX igl::core igl::png igl::glfw igl::opengl)
target_include_directories(${PROJECT_NAME}_bin PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")

target_compile_options(${PROJECT_NAME}_bin PRIVATE -openmp:llvm)
