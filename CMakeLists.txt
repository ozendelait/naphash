cmake_minimum_required(VERSION 3.0.2)

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

project(naphash_cpp CXX)
set(CPACK_PACKAGE_VERSION ${PYBIND11_VERSION_INFO})
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

if (NOT WIN32 AND NOT OpenCV_FOUND)
    # Add libopencv from conda user directory to searchpath
    execute_process(COMMAND whoami OUTPUT_VARIABLE current_user OUTPUT_STRIP_TRAILING_WHITESPACE)
    file(GLOB OCV_USER_DIR LIST_DIRECTORIES true /home/${current_user}/.conda/pkgs/libopencv-4*)
    list(FILTER OCV_USER_DIR EXCLUDE REGEX bz2)
    if(OCV_USER_DIR)
        set(OCV_USER_DIR "${OCV_USER_DIR}/lib/cmake/opencv4/")
    endif()
endif()
find_package(OpenCV PATHS OCV_USER_DIR)


include_directories(${PYTHON_INCLUDE_DIRS} ${_numpy_include_dirs} ${CMAKE_CURRENT_SOURCE_DIR}/include)
message("Python libs: ${PYTHON_LIBRARIES}")
message("OpenCV libs: ${OpenCV_LIBS}")

add_library(naphash_cpp SHARED src/naphash_cpp.cpp)
target_link_libraries(naphash_cpp PRIVATE opencv_core opencv_imgproc)

if (WIN32)
  target_compile_definitions(naphash_cpp PRIVATE _WIN32DLL_BUILD_NAPHASH_EXPORTS)
endif()

#set_target_properties(naphash_cpp PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
target_include_directories(naphash_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

pybind11_add_module(naphash_py src/naphash_py.cpp )
add_dependencies(naphash_py naphash_cpp)
target_link_libraries(naphash_py PUBLIC ${PYTHON_LIBRARIES})
target_link_libraries(naphash_py PRIVATE naphash_cpp)
target_include_directories(naphash_py PRIVATE ${PYTHON_INCLUDE_DIRS} ${_numpy_include_dirs})
target_compile_features(naphash_py PRIVATE cxx_std_17)
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

if(WIN32)
  install(FILES ${PROJECT_BINARY_DIR}/$<CONFIG>/naphash_cpp.dll DESTINATION lib)
endif()

install(EXPORT naphash-export DESTINATION CMake COMPONENT dev)
