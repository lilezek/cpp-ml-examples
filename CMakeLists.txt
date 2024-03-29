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

# Example 2
project(example-2)
file(GLOB_RECURSE EXAMPLE_SOURCES CONFIGURE_DEPENDS 2-computing-grads/*.cpp)

add_executable(example-2 ${EXAMPLE_SOURCES})
target_include_directories(example-2 PUBLIC math)
target_link_libraries(example-2 math)
set_target_properties(example-2 PROPERTIES LINKER_LANGUAGE CXX)

# Example 3
project(example-3)
file(GLOB_RECURSE EXAMPLE_SOURCES CONFIGURE_DEPENDS 3-linear-multivariable/*.cpp)

add_executable(example-3 ${EXAMPLE_SOURCES})
target_include_directories(example-3 PUBLIC math)
target_link_libraries(example-3 math)
set_target_properties(example-3 PROPERTIES LINKER_LANGUAGE CXX)
