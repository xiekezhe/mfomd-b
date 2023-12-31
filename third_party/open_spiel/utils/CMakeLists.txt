add_library (utils OBJECT
  circular_buffer.h
  reservior_buffer.h
  dict_buffer.h
  data_logger.h
  data_logger.cc
  file.h
  file.cc
  json.h
  json.cc
  logger.h
  lru_cache.h
  run_python.h
  run_python.cc
  stats.h
  tensor_view.h
  barrier.h
  barrier.cc
  thread.h
  thread.cc
  thread_pool.h
  threaded_queue.h
  threaded_unordered_map.h
)
target_include_directories (utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(circular_buffer_test circular_buffer_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(circular_buffer_test circular_buffer_test)

add_executable(reservior_buffer_test reservior_buffer_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(reservior_buffer_test reservior_buffer_test)

add_executable(dict_buffer_test dict_buffer_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(dict_buffer_test dict_buffer_test)

add_executable(data_logger_test data_logger_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(data_logger_test data_logger_test)

add_executable(file_test file_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(file_test file_test)

add_executable(json_test json_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(json_test json_test)

add_executable(logger_test logger_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(logger_test logger_test)

add_executable(lru_cache_test lru_cache_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(lru_cache_test lru_cache_test)

# add_executable(run_python_test run_python_test.cc ${OPEN_SPIEL_OBJECTS}
#                $<TARGET_OBJECTS:tests>)
# add_test(run_python_test run_python_test)

add_executable(stats_test stats_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(stats_test stats_test)

add_executable(tensor_view_test tensor_view_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(tensor_view_test tensor_view_test)

add_executable(thread_test thread_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(thread_test thread_test)

add_executable(threaded_queue_test threaded_queue_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(threaded_queue_test threaded_queue_test)
