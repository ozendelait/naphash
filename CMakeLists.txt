cmake_minimum_required(VERSION 3.0.2)

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

project(naphash_cpp VERSION 0.7 LANGUAGES CXX)
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
include(CPack)
set(CMAKE_BUILD_TYPE Release)
find_package(pybind11 REQUIRED)

if(${CMAKE_VERSION} VERSION_GREATER "3.11.99999")
  find_package(Python ${PY_VERSION} REQUIRED)
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

add_library(naphash_cpp src/naphash_cpp.cpp)
target_link_libraries(naphash_cpp PRIVATE opencv_core opencv_imgproc)
target_include_directories(naphash_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

pybind11_add_module(naphash_py src/naphash_py.cpp )
add_dependencies(naphash_py naphash_cpp)
target_link_libraries(naphash_py PRIVATE ${PYTHON_LIBRARIES} naphash_cpp)
target_include_directories(naphash_py PRIVATE ${PYTHON_INCLUDE_DIRS} ${_numpy_include_dirs})

install(TARGETS naphash_cpp
          EXPORT naphash-export COMPONENT dev
          LIBRARY DESTINATION lib COMPONENT dev
          ARCHIVE DESTINATION lib COMPONENT dev
          )

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include COMPONENT dev
        FILES_MATCHING
        PATTERN "*.hpp"
        )

install(TARGETS naphash_py
          EXPORT naphash-export COMPONENT dev
          LIBRARY DESTINATION lib COMPONENT dev
          ARCHIVE DESTINATION lib COMPONENT dev
          )

install(EXPORT naphash-export DESTINATION CMake COMPONENT dev)