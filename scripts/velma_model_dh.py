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
from pykdl_utils.kdl_kinematics import KDLKinematics

import PyKDL

import rospy
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Vector3, Point, Pose, Quaternion
from visualization_msgs.msg import Marker, MarkerArray

from sensor_msgs.msg import JointState

def toKDLVector(vec):
    return PyKDL.Vector(vec[0], vec[1], vec[2])

def toKDLFrame(pose):
    return PyKDL.Frame(PyKDL.Rotation.RPY(pose.rpy[0], pose.rpy[1], pose.rpy[2]),
                                                                            toKDLVector(pose.xyz))

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

def printVector(p):
    print('PyKDL.Vector({}, {}, {})'.format(p.x(), p.y(), p.z()))

def printFrame(T):
    q = T.M.GetQuaternion()
    print('PyKDL.Frame(PyKDL.Rotation.Quaternion({}, {}, {}, {}), PyKDL.Vector({}, {}, {}))'.format(
            q[0], q[1], q[2], q[3], T.p.x(), T.p.y(), T.p.z()))

def detectWrongJoints():

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

#def getCommonNormal(T_J_K):
    # Get the point at which the common normal
    # crosses the previous z axis 

def findCommonNormal(pt1, vec1, pt2, vec2):
    norm = vec1*vec2
    if norm.Norm() < 0.00001:
        return None, None

    #pt1 + a*vec1 + norm*b = pt2 + c*vec2

    #a*vec1.x() + b*norm.x() - c*vec2.x() = pt2.x() - pt1.x()
    #a*vec1 + b*norm - c*vec2 = pt2 - pt1

    a = np.array([[vec1.x(), norm.x(), -vec2.x()], [vec1.y(), norm.y(), -vec2.y()], [vec1.z(), norm.z(), -vec2.z()]])
    b = np.array([pt2.x() - pt1.x(), pt2.y() - pt1.y(), pt2.z() - pt1.z()])
    x = np.linalg.solve(a, b)

    result = pt1 + x[0]*vec1 + x[1]*norm - pt2 - x[2]*vec2
    #print('findCommonNormal')
    #printVector(result)

    return pt1 + x[0]*vec1, pt2 + x[2]*vec2

# Solve the system of equations
# x0 + 2 * x1 = 1
# 3 * x0 + 5 * x1 = 2:
# a = np.array([[1, 2], [3, 5]])
# b = np.array([1, 2])
# x = np.linalg.solve(a, b)


def DHizeJoint(joint):
    # We assume here that the current joint z axis is (0, 0, 1)
    current_axis_J = PyKDL.Vector(0, 0, 1)
    if joint.axis is None:
        next_axis_K = PyKDL.Vector(0, 0, 1)
    else:
        assert joint.axis[2] == 1
        next_axis_K = toKDLVector(joint.axis)

    T_J_K = toKDLFrame(joint.origin)

    pt_next_axis_J = T_J_K.p
    next_axis_J = T_J_K.M * next_axis_K
    is_parallel = (abs(next_axis_J.z()) > 0.9999)

    # We can slide the next joint origin along its axis
    if is_parallel:
        d = T_J_K.p.z()
        a = (T_J_K.p - PyKDL.Vector(0, 0, d)).Norm()
        #if abs(a) < 0.0000001:
            # The axes coincide
        new_T_J_K = PyKDL.Frame(PyKDL.Rotation(), T_J_K.p)
        T_K_K_new = T_J_K.Inverse() * new_T_J_K
        alpha = 0.0
        return a, d, alpha, T_K_K_new
    else:

        pt_cn1, pt_cn2 = findCommonNormal(PyKDL.Vector(), PyKDL.Vector(0,0,1), pt_next_axis_J, next_axis_J)

        d = pt_cn1.z()
        a = (pt_cn2-pt_cn1).Norm()
        if a < 0.000001:
            next_frame_x_J = PyKDL.Vector(0,0,1) * next_axis_J
        else:
            next_frame_x_J = (pt_cn2-pt_cn1)

        next_frame_z_J = next_axis_J
        next_frame_x_J.Normalize()
        next_frame_y_J = next_frame_z_J * next_frame_x_J
        next_frame_y_J.Normalize()

        next_frame = PyKDL.Frame(PyKDL.Rotation(next_frame_x_J, next_frame_y_J, next_frame_z_J), pt_cn2)
        new_T_J_K = next_frame
        T_K_K_new = T_J_K.Inverse() * new_T_J_K

        alpha = math.atan2(next_axis_J.y(), next_axis_J.z())

        return a, d, alpha, T_K_K_new
        #next_x_J = PyKDL.Vector(0, 0, 1) * next_axis_J
        # Project the next axis to the YZ plane of the current joint
        #pt = (pt_next_axis_J.y(), pt_next_axis_J.z())
        #v = (next_axis_J.y(), next_axis_J.z())
        #(pt + f * v)[0] = 0
        #f = -pt_next_axis_J.y() / next_axis_J.y()
        #pt_cn = pt_next_axis_J + f * next_axis_J
        #d = pt_cn.z()
        #print('getDH( {} )'.format(joint.name))
        #printVector( pt_cn )

def getJointNamesForChain(robot, link_name):
    result = []
    while link_name in robot.parent_map:
        parent_joint_name, parent_link_name = robot.parent_map[link_name]
        result.append(parent_joint_name)
        link_name = parent_link_name
    return result

def calculateDH():
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

    #end_effectors = ['stereo_left_link']
    #end_effectors = ['right_arm_ee_link']
    #end_effectors = ['left_arm_ee_link']

    end_effectors = ['stereo_left_link', 'right_arm_ee_link', 'left_arm_ee_link']

    result = RobotModelTest()

    for end_effector in end_effectors:
        chain_joints = getJointNamesForChain(robot, end_effector)

        for joint_name in reversed(chain_joints):
            joint = robot.joint_map[joint_name]
            print('***** Joint {} is of type {}, current link: {}, parent link: {}'.format(
                    joint.name, joint.type, joint.child, joint.parent))
            if joint.type == 'fixed':
                print('pose: xyz=[{}, {}, {}], rpy=[{}, {}, {}]'.format(
                                joint.origin.xyz[0], joint.origin.xyz[1], joint.origin.xyz[2],
                                joint.origin.rpy[0], joint.origin.rpy[1], joint.origin.rpy[2]))
                result.addFixed( joint_name, joint.parent, joint.child, toKDLFrame(joint.origin) )
                print('PyKDL.Frame(PyKDL.Rotation.RPY({}, {}, {}), PyKDL.Vector({}, {}, {}))'.format(
                                joint.origin.rpy[0], joint.origin.rpy[1], joint.origin.rpy[2],
                                joint.origin.xyz[0], joint.origin.xyz[1], joint.origin.xyz[2]))
                continue
            # else:
            #print('*** DH for joint {}'.format(joint_name))
            dh_item = DHizeJoint(joint)
            if dh_item is None:
                print('could not calculate DH for joint {}'.format(joint_name))
            else:
                a, d, alpha, jtf = dh_item
                #print('joint transform:')
                #printFrame(jtf)
                #print(poseToURDFstr(jtf))
                print('DH [d, a, alpha]: [{}, {}, {}]'.format(d, a, alpha))
                result.addDH( joint_name, joint.parent, joint.child, d, a, alpha )
                transformJoint(robot, joint_name, jtf)

    return result

def poseToURDFstr(pose):
    rpy = pose.M.GetRPY()
    return 'xyz: [{} {} {}], rpy: [{} {} {}]'.format(pose.p.x(), pose.p.y(), pose.p.z(),
                                                                            rpy[0], rpy[1], rpy[2])

def getAllChildLinks(robot, joint_name):
    result = []
    for link_name in robot.parent_map:
        parent_joint_name, parent_link_name = robot.parent_map[link_name]
        if parent_joint_name == joint_name:
            result.append(link_name)
    return result

def getAllChildJointNames(robot, joint_name):
    child_links = getAllChildLinks(robot, joint_name)
    result = []
    for link_name in child_links:
        for joint in robot.joints:
            if joint.parent == link_name:
                result.append(joint.name)
    return result

def poseToURDFOriginStr(pose):
    rpy = pose.M.GetRPY()
    return '<origin xyz="{} {} {}" rpy="{} {} {}" />'.format(pose.p.x(), pose.p.y(), pose.p.z(),
                                                                            rpy[0], rpy[1], rpy[2])

def transformJoint(robot, joint_name, jtf):
    # Transform the given joint, child link and child joints
    joint = robot.joint_map[joint_name]
    T_I_J = toKDLFrame(joint.origin)

    print_joint_tf = False

    new_T_I_J = T_I_J * jtf
    if print_joint_tf:
        print('* Transformation of joint {}'.format(joint_name))
        print('old origin of joint {}: {}'.format(joint_name, poseToURDFOriginStr(T_I_J)))
        print('new origin of joint {}:\n{}'.format(joint_name, poseToURDFOriginStr(new_T_I_J)))

    for child_joint_name in getAllChildJointNames(robot, joint_name):
        joint_next = robot.joint_map[child_joint_name]
        T_J_K = toKDLFrame(joint_next.origin)
        new_T_J_K = jtf.Inverse() * T_J_K
        if print_joint_tf:
            print('old origin of child joint {}: {}'.format(child_joint_name, poseToURDFOriginStr(T_J_K)))
            print('new origin of child joint {}:\n{}'.format(child_joint_name, poseToURDFOriginStr(new_T_J_K)))

    link = robot.link_map[joint.child]
    if print_joint_tf:
        print('visuals of link {}:'.format(link.name))
    for visual in link.visuals:
        pose = toKDLFrame(visual.origin)
        new_pose = jtf.Inverse()*pose
        if print_joint_tf:
            print('  name: {}, old origin: [{}], new origin:\n{}'.format(visual.name,
                    visual.origin, poseToURDFOriginStr(new_pose)))

    if print_joint_tf:
        print('collisions of link {}:'.format(link.name))
    for collision in link.collisions:
        pose = toKDLFrame(collision.origin)
        new_pose = jtf.Inverse()*pose
        if print_joint_tf:
            print('  old origin: [{}], new origin:\n{}'.format(
                    collision.origin, poseToURDFOriginStr(new_pose)))

    if not link.inertial is None:
        if print_joint_tf:
            print('inertial of link {}:'.format(link.name))
        pose = toKDLFrame(link.inertial.origin)
        new_pose = jtf.Inverse()*pose
        if print_joint_tf:
            print('  old origin: [{}], new origin:\n{}'.format(
                    link.inertial.origin, poseToURDFOriginStr(new_pose)))

def joinLinks(robot, link_name1, link_name2):
    raise Exception('not implemented')

    tree = kdl_tree_from_urdf_model(robot)

    parent_joint_name, parent_link_name = robot.parent_map[link_name2]
    if link_name1 != parent_link_name:
        raise Exception('Could not join link {} to link {}. The parent link of {} is {}'.format(
            link_name2, link_name1, link_name2, parent_link_name))

    joint = robot.joint_map[parent_joint_name]
    if joint.type != 'fixed':
        raise Exception('Could not joint links {} and {}, because joint {} is not fixed'.format(
            link_name1, link_name2, parent_joint_name))

    # Transform all visuals to the parent link frame
    T_2_1 = joint.origin
    T_1_2 = T_2_1.Inverse()
    link = robot.link_map[link_name2]
    print('visuals of link {} transformed into link {}:'.format(link.name, link_name1))
    for visual in link.visuals:
        pose_2 = toKDLFrame(visual.origin)
        pose_1 = T_1_2 * pose_2
        print('  name: {}, new origin:\n  {}'.format(visual.name, poseToURDFOriginStr(pose_1)))

    print('collisions of link {}:'.format(link.name))
    for collision in link.collisions:
        pose_2 = toKDLFrame(collision.origin)
        pose_1 = T_1_2 * pose_2
        print('  new origin:\n  {}'.format(poseToURDFOriginStr(pose_1)))

    if not link.inertial is None:
        seg = tree.getSegment()
        print seg
        print dir(seg)

        # TODO: the new origin must be at the new center of mass, and inertias must be transformed
        # and added
        #print('inertial of link {}:'.format(link.name))
        #pose_2 = toKDLFrame(link.inertial.origin)
        #pose_1 = T_1_2 * pose_2
        #link.inertial
        #print('  old origin: [{}], new origin:\n{}'.format(
        #        link.inertial.origin, poseToURDFOriginStr(new_pose)))

from rcprg_ros_utils import MarkerPublisher

class RobotModelTest:
    def __init__(self):
        self.__fixed_map = {}
        self.__dh_map = {}
        self.__parent_link_map = {}
        self.__parent_joint_map = {}
        self.__link_names = set()

        self.__poses_map = {}
        self.__config = {}

        self.__marker_pub = MarkerPublisher('obj_model')

    def setConfig(self, config):
        self.__config = config
        self.__poses_map = {}

    def getAllPoses(self):
        for link_name in self.__link_names:
            self.getPose(link_name)
        return self.__poses_map

    def addFixed(self, joint_name, parent_link, child_link, pose):
        self.__fixed_map[child_link] = pose
        self.__parent_link_map[child_link] = parent_link
        self.__parent_joint_map[child_link] = joint_name
        self.__link_names.add(parent_link)
        self.__link_names.add(child_link)

    def addDH(self, joint_name, parent_link, child_link, d, a, alpha ):
        self.__dh_map[child_link] = (d, a, alpha)
        self.__parent_link_map[child_link] = parent_link
        self.__parent_joint_map[child_link] = joint_name

    def getParentLinkName(self, link_name):
        if link_name in self.__parent_link_map:
            return self.__parent_link_map[link_name]
        # else:
        return None

    def getParentJointName(self, link_name):
        return self.__parent_joint_map[link_name]

    def getLocalTransform(self, link_name):
        if link_name in self.__fixed_map:
            return self.__fixed_map[link_name]
        # else:
        d, a, alpha = self.__dh_map[link_name]
        theta = self.__config[self.getParentJointName(link_name)]
        #print('{}: {}'.format(self.getParentJointName(link_name), theta))
        return PyKDL.Frame(PyKDL.Vector(0, 0, d)) * PyKDL.Frame(PyKDL.Rotation.RotZ(theta)) *\
                    PyKDL.Frame(PyKDL.Vector(a, 0, 0)) * PyKDL.Frame(PyKDL.Rotation.RotX(alpha))

    def getPose(self, link_name):
        if link_name in self.__poses_map:
            return self.__poses_map[link_name]
        # else:
        parent_link_name = self.getParentLinkName(link_name)
        if parent_link_name is None:
            return PyKDL.Frame()
        # else:

        parent_pose = self.getPose( parent_link_name )
        local_tf = self.getLocalTransform(link_name)
        pose = parent_pose * local_tf
        #print poseToURDFstr(local_tf)
        self.__poses_map[link_name] = pose
        return pose

    def cb(self, data):
        show_links = ['head_tilt_link_dummy', 'stereo_left_link']
        #show_links = ['head_tilt_link', 'stereo_left_link']
        config = {}
        for idx, name in enumerate(data.name):
            config[name] = data.position[idx]
        self.setConfig(config)
        poses_map = self.getAllPoses()
        m_id = 0
        for link_name, pose in poses_map.iteritems():
            if link_name in show_links:
                m_id = self.__marker_pub.addFrameMarker(pose, m_id, scale=0.1, frame='torso_base', namespace='default')
        self.__marker_pub.publishAll()

import random

def main():

    robot = URDF.from_parameter_server()
    tree = kdl_tree_from_urdf_model(robot)


    print('robot:')
    print(dir(robot))

    #detectWrongJoints()
    #calculateDH()
#getJointNamesForChain()
    #transformJoint(robot, 'right_arm_0_joint', PyKDL.Frame(PyKDL.Rotation.RotZ(math.pi/2)))

    #model = calculateDH()
    #return 0
    model = RobotModelTest()

    model.addDH('torso_0_joint', 'torso_base', 'torso_link0', 0.03, 0, 0 )

    # This is the proper DH for KUKA LWR:
    model.addDH('right_arm_0_joint', 'calib_right_arm_base_link', 'right_arm_1_link', 0.3105, 0, -1.57079632679 )
    model.addDH('right_arm_1_joint', 'right_arm_1_link', 'right_arm_2_link', 0, 0, 1.57079632679 )
    model.addDH('right_arm_2_joint', 'right_arm_2_link', 'right_arm_3_link', 0.4, 0, 1.57079632679 )
    model.addDH('right_arm_3_joint', 'right_arm_3_link', 'right_arm_4_link', 0, 0, -1.57079632679 )
    model.addDH('right_arm_4_joint', 'right_arm_4_link', 'right_arm_5_link', 0.39, 0, -1.57079632679 )
    model.addDH('right_arm_5_joint', 'right_arm_5_link', 'right_arm_6_link', 0, 0, 1.57079632679 )
    model.addDH('right_arm_6_joint', 'right_arm_6_link', 'right_arm_7_link', 0, 0, 0 )

    model.addDH('left_arm_0_joint', 'calib_left_arm_base_link', 'left_arm_1_link', 0.3105, 0, -1.57079632679 )
    model.addDH('left_arm_1_joint', 'left_arm_1_link', 'left_arm_2_link', 0, 0, 1.57079632679 )
    model.addDH('left_arm_2_joint', 'left_arm_2_link', 'left_arm_3_link', 0.4, 0, 1.57079632679 )
    model.addDH('left_arm_3_joint', 'left_arm_3_link', 'left_arm_4_link', 0, 0, -1.57079632679 )
    model.addDH('left_arm_4_joint', 'left_arm_4_link', 'left_arm_5_link', 0.39, 0, -1.57079632679 )
    model.addDH('left_arm_5_joint', 'left_arm_5_link', 'left_arm_6_link', 0, 0, 1.57079632679 )
    model.addDH('left_arm_6_joint', 'left_arm_6_link', 'left_arm_7_link', 0, 0, 0 )

    model.addDH('head_pan_joint', 'head_pan_motor', 'head_pan_link', 0.11, 0, -1.57079632679 )
    model.addDH('head_tilt_joint', 'head_pan_link', 'head_tilt_link_dummy', 0, 0, math.pi)

    # Do not calibrate:
    model.addFixed('torso_base_joint', 'world', 'torso_base',
        PyKDL.Frame(PyKDL.Rotation.RPY(0.0, 0.0, 0.0), PyKDL.Vector(0.0, 0.0, 0.0)))

    # Calibrate:
    model.addFixed('head_base_joint', 'torso_link0', 'head_base',
        PyKDL.Frame(PyKDL.Rotation.RPY(0.00040339, -0.0062724, -0.048944), PyKDL.Vector(0.020988, -0.00041436, 1.3117)))

    # Do not calibrate:
    model.addFixed('head_pan_motor_joint', 'head_base', 'head_pan_motor',
        PyKDL.Frame(PyKDL.Rotation.RPY(0.0, 0.0, 0.0), PyKDL.Vector(0.0, 0.0, 0.025)))

    # Do not calibrate:
    # This fixed transform differs in the models:
    model.addFixed('head_tilt_joint_dummy', 'head_tilt_link_dummy', 'head_tilt_link', 
        PyKDL.Frame(PyKDL.Rotation.RPY(0, 0, 0), PyKDL.Vector(0.0, 0.0, 0.0)))
        #PyKDL.Frame(PyKDL.Rotation.RPY(3.14159265358, 0.0, -3.14159265359), PyKDL.Vector(0.0, 0.0, 0.0))

    # Calibrate:
    model.addFixed('stereo_left_joint', 'head_tilt_link', 'stereo_left_link', 
        PyKDL.Frame(PyKDL.Rotation.RPY(-1.5735, 0.013221, 0.023637), PyKDL.Vector(0.013633, 0.22937, -0.045798)))

    # Calibrate:
    model.addFixed('torso_link0_right_arm_base_joint', 'torso_link0', 'calib_right_arm_base_link', 
        PyKDL.Frame(PyKDL.Rotation.RPY(0.0, -1.0471975512, 1.57079632679), PyKDL.Vector(0.0, -0.000188676, 1.17335)))

    # Do not calibrate:
    model.addFixed('right_arm_ee_joint', 'right_arm_7_link', 'right_arm_ee_link', 
        PyKDL.Frame(PyKDL.Rotation.RPY(0.0, 0.0, 0.0), PyKDL.Vector(0.0, 0.0, 0.078)))

    # Calibrate:
    model.addFixed('torso_link0_left_arm_base_joint', 'torso_link0', 'calib_left_arm_base_link',
        PyKDL.Frame(PyKDL.Rotation.RPY(0.0, 1.0471975512, 1.57079632679), PyKDL.Vector(0.0, 0.000188676, 1.17335)))

    # Do not calibrate:
    model.addFixed('left_arm_ee_joint', 'left_arm_7_link', 'left_arm_ee_link', 
        PyKDL.Frame(PyKDL.Rotation.RPY(0.0, 0.0, 0.0), PyKDL.Vector(0.0, 0.0, 0.078)))

    joint_names = ['torso_0_joint', 'head_pan_joint', 'head_tilt_joint']
    for i in range(7):
        joint_names.append('right_arm_{}_joint'.format(i))
        joint_names.append('left_arm_{}_joint'.format(i))

    
    # Test arm
    chain = tree.getChain('torso_base', 'right_arm_ee_link')
    fk_kdl = PyKDL.ChainFkSolverPos_recursive(chain)

    for test_idx in range(10):
        q = []
        for i in range(8):
            q.append(random.uniform(-1,1))

        endeffec_frame = PyKDL.Frame()
        q_kdl = PyKDL.JntArray(len(q))
        for i, q_i in enumerate(q):
            q_kdl[i] = q_i
        kinematics_status = fk_kdl.JntToCart(q_kdl, endeffec_frame)

        config = {}
        for joint_name in joint_names:
            config[joint_name] = 0.0
        config['torso_0_joint'] = q[0]
        for i in range(7):
            config['right_arm_{}_joint'.format(i)] = q[i+1]

        model.setConfig(config)
        pose = model.getPose('right_arm_ee_link')

        #print(poseToURDFstr(endeffec_frame*pose.Inverse()))
        diff = PyKDL.diff(endeffec_frame, pose)
        print(diff.vel.Norm() + diff.rot.Norm())

    # Test head
    chain = tree.getChain('torso_base', 'stereo_left_link')
    fk_kdl = PyKDL.ChainFkSolverPos_recursive(chain)

    for test_idx in range(10):
        q = []
        for i in range(3):
            q.append(random.uniform(-1,1))

        endeffec_frame = PyKDL.Frame()
        q_kdl = PyKDL.JntArray(len(q))
        for i, q_i in enumerate(q):
            q_kdl[i] = q_i
        kinematics_status = fk_kdl.JntToCart(q_kdl, endeffec_frame)

        config = {}
        for joint_name in joint_names:
            config[joint_name] = 0.0
        config['torso_0_joint'] = q[0]
        config['head_pan_joint'] = q[1]
        config['head_tilt_joint'] = q[2]

        model.setConfig(config)
        pose = model.getPose('stereo_left_link')

        #print(poseToURDFstr(endeffec_frame*pose.Inverse()))
        #print(poseToURDFstr(endeffec_frame))
        #print(poseToURDFstr(pose))
        diff = PyKDL.diff(endeffec_frame, pose)
        print(diff.vel.Norm() + diff.rot.Norm())

    #return 0
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/joint_states", JointState, model.cb)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

    #joinLinks(robot, 'torso_link0', 'calib_right_arm_base_link')

    return 0

if __name__ == "__main__":
    exit( main() )


# calib_right...
# right_arm_joint_0
# right_arm_link_1
# right_arm_joint_1
# right_arm_link_2
# ...
