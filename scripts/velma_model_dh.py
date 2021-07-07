#!/usr/bin/env python

# This script calculates symbolic model for forward kinematics
# for a robot defined in an URDF file loaded from ros param.
#
# Author: Dawid Seredynski, 2020
#

import numpy as np
from sympy import pprint, init_printing, Symbol, sin, cos, Matrix, sympify,\
                    lambdify
import math

from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
import PyKDL

import rospy
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Vector3, Point, Pose, Quaternion
from visualization_msgs.msg import Marker, MarkerArray

def toKDLFrame(pose):
    return PyKDL.Frame(PyKDL.Rotation.RPY(pose.rpy[0], pose.rpy[1], pose.rpy[2]),
                                            PyKDL.Vector(pose.xyz[0], pose.xyz[1], pose.xyz[2]))

def getRequiredJointTransform(axis):
    if axis is None:
        return False, PyKDL.Frame()
    elif axis[0] == 0 and axis[1] == 0 and axis[2] == 1:
        return False, PyKDL.Frame()
    elif axis[0] == 0 and axis[1] == 0 and axis[2] == -1:
        return True, PyKDL.Frame(PyKDL.Rotation.RotX(math.pi), PyKDL.Vector())
    elif axis[0] == 0 and axis[1] == 1 and axis[2] == 0:
        return True, PyKDL.Frame(PyKDL.Rotation.RotX(-math.pi/2)*PyKDL.Rotation.RotZ(math.pi), PyKDL.Vector())
    elif axis[0] == 0 and axis[1] == -1 and axis[2] == 0:
        return True, PyKDL.Frame(PyKDL.Rotation.RotX(math.pi/2), PyKDL.Vector())
    else:
        raise Exception('Not supported joint axis: {}'.format(axis))

def printFrame(T):
    q = T.M.GetQuaternion()
    print('PyKDL.Frame(PyKDL.Rotation.Quaternion({}, {}, {}, {}), PyKDL.Vector({}, {}, {}))'.format(
            q[0], q[1], q[2], q[3], T.p.x(), T.p.y(), T.p.z()))

def main():

    robot = URDF.from_parameter_server()
    tree = kdl_tree_from_urdf_model(robot)

    print('robot:')
    print(dir(robot))

    print('robot.parent_map:')
    print(robot.parent_map)
    #parent_joint_map = {}

    dot = 'digraph gr {\n'
    print('joints:')
    for joint in robot.joints:
        print('  name: {}, type: {}, axis: {}, parent: {}, child: {}'.format(joint.name, joint.type, joint.axis, joint.parent, joint.child))
        dot += '  {} [label="{}\\n{}\\n{}"];\n'.format(joint.name, joint.name, joint.type, joint.axis)
        dot += '  {} -> {}\n;'.format(joint.parent, joint.name)
        dot += '  {} -> {}\n;'.format(joint.name, joint.child)
    print(dir(robot.joints[0]))

    print('links:')
    for link in robot.links:
        print('  name: {}'.format(link.name))
        dot += '  {} [shape=box];\n'.format(link.name)
    print(dir(robot.links[0]))

    dot += '}\n'
    with open('velma_tree.dot', 'w') as f:
        f.write(dot)

    end_effectors = ['right_arm_7_link', 'left_arm_7_link', 'stereo_left_link']

    for end_effector in end_effectors:
        link_name = end_effector
        link = robot.link_map[link_name]
        #print('{} (origin: {})'.format(link_name, link.origin))
        prev_joint = None
        while link_name in robot.parent_map:
            link = robot.link_map[link_name]

            parent_joint_name, parent_link_name = robot.parent_map[link_name]
            joint = robot.joint_map[parent_joint_name]
            #print('{} ({}, {}, origin: {})'.format(parent_joint_name, joint.type, joint.axis, joint.origin))

            # If the axis is other than [0, 0, 1], transform the joint origin
            jtf_needed, jtf = getRequiredJointTransform( joint.axis )
            #printFrame(jtf)

            if jtf_needed:
                new_rpy = jtf.M.GetRPY()
                new_xyz = [jtf.p.x(), jtf.p.y(), jtf.p.z()]
                print('joint {} new origin: [rpy: {}, xyz: {}]'.format(joint.name, new_rpy, new_xyz))

                if not prev_joint is None:
                    new_pose = jtf.Inverse()*toKDLFrame(prev_joint.origin)
                    new_rpy = new_pose.M.GetRPY()
                    new_xyz = [new_pose.p.x(), new_pose.p.y(), new_pose.p.z()]
                    print('prev_joint {} new origin: [rpy: {}, xyz: {}]'.format(prev_joint.name, new_rpy, new_xyz))

                print('visuals of link {}:'.format(link.name))
                for visual in link.visuals:
                    pose = toKDLFrame(visual.origin)
                    new_pose = jtf.Inverse()*pose
                    new_rpy = new_pose.M.GetRPY()
                    new_xyz = [new_pose.p.x(), new_pose.p.y(), new_pose.p.z()]
                    print('  name: {}, old origin: [{}], new origin: [rpy: {}, xyz: {}]'.format(visual.name,
                            visual.origin, new_rpy, new_xyz))

                print('collisions of link {}:'.format(link.name))
                for collision in link.collisions:
                    pose = toKDLFrame(collision.origin)
                    new_pose = jtf.Inverse()*pose
                    new_rpy = new_pose.M.GetRPY()
                    new_xyz = [new_pose.p.x(), new_pose.p.y(), new_pose.p.z()]
                    print('  old origin: [{}], new origin: [rpy: {}, xyz: {}]'.format(
                            collision.origin, new_rpy, new_xyz))

                if not link.inertial is None:
                    print('inertial of link {}:'.format(link.name))
                    pose = toKDLFrame(link.inertial.origin)
                    new_pose = jtf.Inverse()*pose
                    new_rpy = new_pose.M.GetRPY()
                    new_xyz = [new_pose.p.x(), new_pose.p.y(), new_pose.p.z()]
                    print('  old origin: [{}], new origin: [rpy: {}, xyz: {}]'.format(
                            link.inertial.origin, new_rpy, new_xyz))
            
            parent_link = robot.link_map[parent_link_name]
            #print('{} (origin: {})'.format(parent_link_name, link.origin))
            link_name = parent_link_name
            prev_joint = joint


    return 0

if __name__ == "__main__":
    exit( main() )
