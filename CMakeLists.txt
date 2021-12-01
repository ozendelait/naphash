cmake_minimum_required(VERSION 2.8.12)

project(naphash_cpp)
set(CMAKE_BUILD_TYPE Release)
find_package(pybind11 REQUIRED)

if(${CMAKE_VERSION} VERSION_GREATER "3.11.99999")
  find_package(Python ${PY_VERSION} REQUIRED)
  #set(PYTHON_LIBRARIES 
else()
  find_package(PythonLibs)
endif()

execute_process(
    COMMAND "python3" "-c" "import os, numpy.distutils; print(os.pathsep.join(numpy.distutils.misc_util.get_numpy_include_dirs()))"
    RESULT_VARIABLE _numpy_process
    OUTPUT_VARIABLE _numpy_include_dirs
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (NOT _numpy_process EQUAL 0)
  execute_process(
      COMMAND "python" "-c" "import os, numpy.distutils; print(os.pathsep.join(numpy.distutils.misc_util.get_numpy_include_dirs()))"
      RESULT_VARIABLE _numpy_process
      OUTPUT_VARIABLE _numpy_include_dirs
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endif()

if (NOT _numpy_process EQUAL 0)
    message(FATAL_ERROR "Can't locate numpy include dirs")
endif()
find_package(OpenCV)

include_directories(${PYTHON_INCLUDE_DIRS} ${_numpy_include_dirs} ${CMAKE_CURRENT_SOURCE_DIR}/include)
message("Python libs: ${PYTHON_LIBRARIES}")
message("OpenCV libs: ${OpenCV_LIBS}")

pybind11_add_module(naphash_cpp src/pynaphash.cpp src/naphash.cpp)
target_link_libraries(naphash_cpp PRIVATE ${OpenCV_LIBS} ${PYTHON_LIBRARIES})