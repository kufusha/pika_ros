cmake_minimum_required(VERSION 3.8)
project(g1_pika_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch meshes rviz scripts urdf
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
)

ament_package()
