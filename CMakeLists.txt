cmake_minimum_required(VERSION 2.8)
project(velma_model_sym)

find_package(catkin REQUIRED COMPONENTS
)

catkin_package()

catkin_install_python(PROGRAMS
    scripts/velma_model_generator.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
