cmake_minimum_required(VERSION 2.8.12)
project(fsss CXX)

message("${CMAKE_CXX_FLAGS}")

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRCS)
add_library(fsss STATIC ${SRCS})
target_link_libraries(fsss ${CommonLibs})
set_target_properties(fsss PROPERTIES LINKER_LANGUAGE CXX)
