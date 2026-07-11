cmake_minimum_required(VERSION 3.15)
project(edge_event_systemc LANGUAGES CXX)

find_package(SystemCLanguage CONFIG REQUIRED)

add_executable(event_pipeline_sim event_pipeline_sim.cpp)
target_link_libraries(event_pipeline_sim PRIVATE SystemC::systemc)
target_compile_features(event_pipeline_sim PRIVATE cxx_std_17)

add_executable(npu_queue_sim npu_queue_sim.cpp)
target_link_libraries(npu_queue_sim PRIVATE SystemC::systemc)
target_compile_features(npu_queue_sim PRIVATE cxx_std_17)
