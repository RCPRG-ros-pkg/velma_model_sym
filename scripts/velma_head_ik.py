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
from interactive_markers.interactive_marker_server import *
import tf_conversions.posemath as pm
from visualization_msgs.msg import *

from sensor_msgs.msg import JointState

from rcprg_ros_utils import MarkerPublisher

def toKDLVector(vec):
    return PyKDL.Vector(vec[0], vec[1], vec[2])

def toKDLFrame(pose):
    return PyKDL.Frame(PyKDL.Rotation.RPY(pose.rpy[0], pose.rpy[1], pose.rpy[2]),
                                                                            toKDLVector(pose.xyz))
def printVector(p):
    print('PyKDL.Vector({}, {}, {})'.format(p.x(), p.y(), p.z()))

def printFrame(T):
    q = T.M.GetQuaternion()
    print('PyKDL.Frame(PyKDL.Rotation.Quaternion({}, {}, {}, {}), PyKDL.Vector({}, {}, {}))'.format(
            q[0], q[1], q[2], q[3], T.p.x(), T.p.y(), T.p.z()))

class IntMarkersCol:
    def __init__(self):
        self.pt_B = PyKDL.Vector(1.0, 0.0, 1.6)
        # create an interactive marker server on the topic namespace simple_marker
        self.server = InteractiveMarkerServer('int_markers_col')
        self.insert6DofGlobalMarker()

    def insert6DofGlobalMarker(self, ):
        T_B_T = PyKDL.Frame(self.pt_B)
        int_size_marker = InteractiveMarker()
        int_size_marker.header.frame_id = 'torso_base'
        int_size_marker.name = 'col_marker_size'
        int_size_marker.scale = 0.2
        int_size_marker.pose = pm.toMsg(T_B_T)

        int_size_marker.controls.append(self.createInteractiveMarkerControl6DOF(InteractiveMarkerControl.MOVE_AXIS,'x'))
        int_size_marker.controls.append(self.createInteractiveMarkerControl6DOF(InteractiveMarkerControl.MOVE_AXIS,'y'))
        int_size_marker.controls.append(self.createInteractiveMarkerControl6DOF(InteractiveMarkerControl.MOVE_AXIS,'z'))

        self.server.insert(int_size_marker, self.processFeedbackSize)

        self.server.applyChanges()

    def processFeedbackSize(self, feedback):
        self.pt_B = pm.fromMsg(feedback.pose).p

    def createInteractiveMarkerControl6DOF(self, mode, axis):
        control = InteractiveMarkerControl()
        control.orientation_mode = InteractiveMarkerControl.INHERIT
        print dir(InteractiveMarkerControl)
        if mode == InteractiveMarkerControl.ROTATE_AXIS:
            control.name = 'rotate_' + axis;
        elif mode == InteractiveMarkerControl.MOVE_AXIS:
            control.name = 'move_' + axis;
        else:
            raise Exception('Wrong axis mode: "{}"'.format(mode))
        if axis == 'x':
            control.orientation = Quaternion(1,0,0,1)
        elif axis == 'y':
            control.orientation = Quaternion(0,1,0,1)
        elif axis == 'z':
            control.orientation = Quaternion(0,0,1,1)
        else:
            raise Exception('Wrong axis name: "{}"'.format(axis))
        control.interaction_mode = mode
        return control

def calculateHeadIK(pt_HB):
    # obj_x, obj_y, obj_z is expressed in head_pan_motor frame
    # head_pan_motor frame wrt head_base:
    T_HB_2 = PyKDL.Frame( PyKDL.Vector(0, 0, 0.025) )
    T_2_HB = T_HB_2.Inverse()

    pt_2 = T_2_HB * pt_HB

    dist = math.sqrt(pt_2.x()*pt_2.x() + pt_2.y()*pt_2.y())
    optical_frame_dist = 0.000441121425188514
    if dist < optical_frame_dist:
        raise Exception()
    hp_ang = math.atan2(pt_2.y(), pt_2.x()) - math.asin(optical_frame_dist/dist)

    # obj_x, obj_y, obj_z is expressed in head_pan_link frame
    # head_pan_link frame wrt head_base:
    T_HB_3 = PyKDL.Frame( PyKDL.Rotation(
        PyKDL.Vector(cos(hp_ang), sin(hp_ang), 0),
        PyKDL.Vector(-sin(hp_ang), cos(hp_ang), 0),
        PyKDL.Vector(0, 0, 1) ),
        PyKDL.Vector(0, 0, 0.087))
    T_3_HB = T_HB_3.Inverse()

    pt_3 = T_3_HB * pt_HB
    obj_x2 = pt_3.x()
    obj_z2 = pt_3.z() - 0.048
    optical_frame_dist = 0.228380936084183
    dist = math.sqrt(obj_x2*obj_x2 + obj_z2*obj_z2)
    if dist < optical_frame_dist:
        raise Exception()
    ht_ang = -math.atan2(obj_z2, obj_x2) + math.asin(optical_frame_dist/dist) - 0.067
    return hp_ang, ht_ang, pt_3

def calculateHeadOpticalFrame_3(ht):
    # pO_3
    pO_3 = PyKDL.Vector(0.226011730455879*sin(ht) + 0.0449872129382484*cos(ht), -0.000147601929240085,
        -0.0449872129382484*sin(ht) + 0.226011730455879*cos(ht) + 0.048)

    # nx_3
    nx_3 = PyKDL.Vector(-0.0624781120251757*sin(ht) + 0.998025090644539*cos(ht), 0.00651183244019694,
        -0.998025090644539*sin(ht) - 0.0624781120251757*cos(ht))
    return pO_3, nx_3

def getHeadBaseFK(torso_angle):
    s0 = math.sin(torso_angle)
    c0 = math.cos(torso_angle)
    m00 = 0.0489234989105084*s0 + 0.998782833637297*c0
    m01 = -0.998802524041464*s0 + 0.0489219301675506*c0
    m02 = 9.60351680637955e-5*s0 - 0.00628458273342522*c0
    m03 = 0.001*s0 + 0.0105*c0
    m10 = 0.998782833637297*s0 - 0.0489234989105084*c0
    m11 = 0.0489219301675506*s0 + 0.998802524041464*c0
    m12 = -0.00628458273342522*s0 - 9.60351680637955e-5*c0
    m13 = 0.0105*s0 - 0.001*c0
    m20 = 0.00627235887090687
    m21 = 0.000403382053799320
    m22 = 0.999980247203470
    m23 = 1.34170000000000

    return PyKDL.Frame(PyKDL.Rotation(
        PyKDL.Vector(m00, m10, m20),
        PyKDL.Vector(m01, m11, m21),
        PyKDL.Vector(m02, m12, m22)),
        PyKDL.Vector(m03, m13, m23))

def main():

    rospy.init_node('head_ik_test', anonymous=True)
    js_pub = rospy.Publisher("/joint_states", JointState)

    marker_pub = MarkerPublisher('obj_model')
    int_marker = IntMarkersCol()

    rospy.sleep(0.5)

    #pt_center_B = PyKDL.Vector(0.7, 0, 1.6)
    torso_angle = 0.5
    #ang = 0.0
    #radius = 0.5
    while not rospy.is_shutdown():
        T_B_HB = getHeadBaseFK(torso_angle)
        #pt_off = PyKDL.Vector(0, radius*math.cos(ang), radius*math.sin(ang))
        #pt_off = PyKDL.Vector(radius*math.cos(ang), 0, radius*math.sin(ang))
        #pt_B = pt_center_B + pt_off
        pt_B = int_marker.pt_B
        #ang += 0.01
        pt_HB = T_B_HB.Inverse() * pt_B
        hp, ht, pt_3 = calculateHeadIK(pt_HB)

        pO_3, nx_3 = calculateHeadOpticalFrame_3(ht)

        js = JointState()
        js.header.stamp = rospy.Time.now()
        js.name = ['torso_0_joint', 'right_arm_0_joint', 'right_arm_1_joint', 'right_arm_2_joint', 'right_arm_3_joint',
          'right_arm_4_joint', 'right_arm_5_joint', 'right_arm_6_joint', 'left_arm_0_joint', 'left_arm_1_joint',
          'left_arm_2_joint', 'left_arm_3_joint', 'left_arm_4_joint', 'left_arm_5_joint', 'left_arm_6_joint',
          'rightFtSensorJoint', 'right_HandFingerThreeKnuckleTwoJoint', 'right_HandFingerThreeKnuckleThreeJoint',
          'right_HandFingerOneKnuckleOneJoint', 'right_HandFingerOneKnuckleTwoJoint', 'right_HandFingerOneKnuckleThreeJoint',
          'right_HandFingerTwoKnuckleOneJoint', 'right_HandFingerTwoKnuckleTwoJoint', 'right_HandFingerTwoKnuckleThreeJoint',
          'leftFtSensorJoint', 'left_HandFingerThreeKnuckleTwoJoint', 'left_HandFingerThreeKnuckleThreeJoint',
          'left_HandFingerOneKnuckleOneJoint', 'left_HandFingerOneKnuckleTwoJoint', 'left_HandFingerOneKnuckleThreeJoint',
          'left_HandFingerTwoKnuckleOneJoint', 'left_HandFingerTwoKnuckleTwoJoint', 'left_HandFingerTwoKnuckleThreeJoint',
          'head_pan_joint', 'head_tilt_joint']
        js.position = [torso_angle, -0.30026644451306783, -1.800342030014444, 1.2497255575978685,
            0.8494866535293841, 0.0, -0.500141550450732, 0.0, 0.29967303256739, 1.799923150993966,
            -1.250912381489224, -0.8503244115703401, 0.0, 0.49972267143025384, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            hp, ht]
        js_pub.publish(js)

        m_id = 0
        scale_pt = 0.03
        m_id = marker_pub.addSinglePointMarker(pt_B, m_id, r=1, g=0, b=0, a=1, namespace='default', frame_id='torso_base', m_type=Marker.SPHERE, scale=Vector3(scale_pt, scale_pt, scale_pt), T=None)
        #m_id = marker_pub.addSinglePointMarker(pt_HB, m_id, r=0, g=0, b=1, a=1, namespace='default', frame_id='head_base', m_type=Marker.CUBE, scale=Vector3(scale_pt*0.5, scale_pt*2, scale_pt), T=None)
        #m_id = marker_pub.addSinglePointMarker(pt_3, m_id, r=0, g=1, b=0, a=1, namespace='default', frame_id='head_pan_link', m_type=Marker.CUBE, scale=Vector3(scale_pt, scale_pt*0.5, scale_pt*2), T=None)

        m_id = marker_pub.addVectorMarker(pO_3, pO_3+nx_3*3, m_id, 0, 1, 0, a=1, frame='head_pan_link', namespace='default', scale=0.01)
        m_id = marker_pub.addVectorMarker(PyKDL.Vector(), PyKDL.Vector(3,0,0), m_id, 0, 1, 0, a=1, frame='head_tilt_link_dummy', namespace='default', scale=0.01)
        marker_pub.publishAll()

        rospy.sleep(0.05)

    return 0

if __name__ == "__main__":
    exit( main() )
