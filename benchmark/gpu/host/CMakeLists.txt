add_executable(gpu_array_map_host_perf gpu_array_map_host_perf.cpp)
target_link_libraries(gpu_array_map_host_perf PRIVATE runtime)
target_include_directories(gpu_array_map_host_perf PRIVATE
  ${CMAKE_SOURCE_DIR}/runtime/include
)

