cmake_minimum_required(VERSION 3.28)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB_RECURSE MATH_SOURCES CONFIGURE_DEPENDS math/*.cpp)
file(GLOB_RECURSE MATH_HEADERS CONFIGURE_DEPENDS math/*.hpp)

add_library(math)
target_sources(math
    # PRIVATE ${MATH_SOURCES}
    PUBLIC ${MATH_HEADERS}
  )
set_target_properties(math PROPERTIES LINKER_LANGUAGE CXX)

# Example 1
project(example-1)
file(GLOB_RECURSE EXAMPLE_SOURCES CONFIGURE_DEPENDS 1-linear-function/*.cpp)

add_executable(example-1 ${EXAMPLE_SOURCES})
target_include_directories(example-1 PUBLIC math)
target_link_libraries(example-1 math)