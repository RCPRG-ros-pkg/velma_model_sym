#!/usr/bin/env python

# This script calculates symbolic model for forward kinematics
# for a robot defined in an URDF file loaded from ros param.
#
# Author: Dawid Seredynski, 2020
#

import numpy as np
from sympy import pprint, init_printing, Symbol, sin, cos, Matrix, sympify,\
					lambdify

from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
import PyKDL

import rospy
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Vector3, Point, Pose, Quaternion
from visualization_msgs.msg import Marker, MarkerArray

def visualize(model):
	rospy.init_node('symb_model')
	rospy.sleep(0.5)

	used_links = []
	for idx in range(1, 8):
		used_links.append( 'right_arm_{}_link'.format(idx) )
		used_links.append( 'left_arm_{}_link'.format(idx) )
	used_links.append( 'head_pan_link' )
	used_links.append( 'head_tilt_link_dummy' )

	q_list = [	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[1.5,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[1.5,0,-1,0,-1,0,-0.5,0,0,0,0,0,0,0,0,0,0],
				[-1.5,0,-1,0,-1,0,-0.5,0,0,0,0,0,0,0,0,0,0],
				[0, 0.2, -1, 0.2, -0.1, 0.5, -0.4, 0.2, 0.5, 0.3, -0.4, -0.1, -0.6, 1, 0.2, -0.5, 1.2],
				[1.5, 0.2, -1, 0.2, -0.1, 0.5, -0.4, 0.2, 0.5, 0.3, -0.4, -0.1, -0.6, 1, 0.2, -0.5, 1.2],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			]

	pub_marker = rospy.Publisher('symb_model_vis', MarkerArray, queue_size=1000)
	q_prev = q_list[0]
	dq_prev = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	for trj_idx in range(len(q_list)-1):
		if rospy.is_shutdown():
			break
		q1 = q_list[trj_idx]
		q2 = q_list[trj_idx+1]
		for t in np.linspace(0, 1, 100, endpoint=False):
			q = []
			for q1v, q2v in zip(q1, q2):
				q.append( q1v*(1.0-t) + q2v*t )

			dq = []
			for q_idx in range(len(q)):
				dq.append( q[q_idx] - q_prev[q_idx] )

			ddq = []
			for q_idx in range(len(dq)):
				ddq.append( dq[q_idx] - dq_prev[q_idx] )

			m = MarkerArray()
			m_id = 0
			for link_name in used_links:
				func_R, func_P = model.getFkFunc(link_name)
				func_w, func_e, func_A = model.getFdFunc(link_name)
				sP = func_P(*q)
				sR = func_R(*q)
				A = func_A(*(q+dq+ddq))
				marker = Marker()
				marker.header.frame_id = 'torso_base'
				marker.header.stamp = rospy.Time.now()
				marker.ns = 'symb_model_vis'
				marker.id = m_id
				marker.type = Marker.CUBE
				marker.action = Marker.ADD
				T = PyKDL.Frame(PyKDL.Rotation(sR[0,0], sR[0,1], sR[0,2], sR[1,0], sR[1,1], sR[1,2],
												sR[2,0], sR[2,1], sR[2,2]), PyKDL.Vector(sP[0], sP[1], sP[2]))
				point = T.p
				qt = T.M.GetQuaternion()
				marker.pose = Pose( Point(point.x(),point.y(),point.z()), Quaternion(qt[0],qt[1],qt[2],qt[3]) )
				scale = 0.02
				marker.scale = Vector3(scale, scale, scale)
				marker.color = ColorRGBA(1,0,0,1)
				m.markers.append(marker)
				m_id = m_id + 1

				marker = Marker()
				marker.header.frame_id = 'torso_base'
				marker.header.stamp = rospy.Time.now()
				marker.ns = 'symb_model_vis'
				marker.id = m_id
				marker.type = Marker.ARROW
				marker.action = Marker.ADD
				scale = 0.02
				marker.scale = Vector3(scale, 2.0*scale, 0)
				marker.pose = Pose( Point(point.x(),point.y(),point.z()), Quaternion(qt[0],qt[1],qt[2],qt[3]) )
				marker.color = ColorRGBA(0,1,0,1)
				marker.points.append(Point(0,0,0))
				A_scale = 1000.0
				#print link_name, A_scale*A[0],A_scale*A[1],A_scale*A[2]
				marker.points.append(Point(A_scale*A[0],A_scale*A[1],A_scale*A[2]))
				m.markers.append(marker)
				m_id = m_id + 1
			pub_marker.publish(m)

			q_prev = q
			dq_prev = dq
			rospy.sleep(0.01)

def replaceConstants(in_expr):
	sol = str(in_expr)
	sol = sol.replace('0.499999999997053', '0.5')
	sol = sol.replace('0.86602540378614', 'b')
	sol = sol.replace('2.4482638139034e-12', '0')
	sol = sol.replace('4.89652762780679e-12', '0')
	sol = sol.replace('4.24055235370702e-12', '0')
	sol = sol.replace('5.55111512312578e-17', '0')
	sol = sol.replace('1.31669150582603e-12', '0')
	sol = sol.replace('1.60242835147952e-11', '0')
	sol = sol.replace('2.49168389885672e-10', '0')
	sol = sol.replace('1.0*', '')
	return sympify(sol)

used_joints = ['torso_0_joint', 'right_arm_0_joint', 'right_arm_1_joint', 
	'right_arm_2_joint', 'right_arm_3_joint', 'right_arm_4_joint',
	'right_arm_5_joint', 'right_arm_6_joint', 'left_arm_0_joint',
	'left_arm_1_joint', 'left_arm_2_joint', 'left_arm_3_joint',
	'left_arm_4_joint', 'left_arm_5_joint', 'left_arm_6_joint',
	'head_pan_joint', 'head_tilt_joint']

base_link = 'torso_base'
end_link_list = ['right_arm_7_link', 'left_arm_7_link', 'head_kinect_rgb_link']

joint_name_idx_map = {}
for idx, joint_name in enumerate(used_joints):
	joint_name_idx_map[joint_name] = idx

matlab_code = 'b = 0.86602540378614;\n'
expr_str = None
for idx in range(len(used_joints)):
	if expr_str is None:
		expr_str = 's = [sin(q({}))'.format(idx+1)
	else:
		expr_str += ', sin(q({}))'.format(idx+1)
expr_str += '];'
matlab_code += expr_str + '\n'

expr_str = None
for idx in range(len(used_joints)):
	if expr_str is None:
		expr_str = 'c = [cos(q({}))'.format(idx+1)
	else:
		expr_str += ', cos(q({}))'.format(idx+1)
expr_str += '];'
matlab_code += expr_str + '\n'

robot = URDF.from_parameter_server()
tree = kdl_tree_from_urdf_model(robot)

class RobotSymbModel:

	class Chain:
		def __init__(self):
			self.__segments = []

		def getNrOfSegments(self):
			return len(self.__segments)

		def getSegment(self, idx):
			return self.__segments[idx]

		def addSegment(self, seg):
			self.__segments.append(seg)

	class Segment:
		def __init__(self, name, frame_to_tip, joint):
			self.__name = name
			self.__frame_to_tip = frame_to_tip
			self.__joint = joint

		def getJoint(self):
			return self.__joint

		def getFrameToTip(self):
			return self.__frame_to_tip

		def getName(self):
			return self.__name

	class Joint:
		def __init__(self, kdl_joint):
			self.__kdl_joint = kdl_joint

		def getName(self):
			return self.__kdl_joint.getName()

		def JointAxis(self):
			return self.__kdl_joint.JointAxis()

		def JointOrigin(self):
			return self.__kdl_joint.JointOrigin()

		def getTypeName(self):
			return self.__kdl_joint.getTypeName()


	def __init__(self, tree, base_link, used_joints):
		self.__tree = tree
		self.__base_link = base_link
		self.__used_joints = used_joints

		self.__joint_name_idx_map = {}
		for idx, joint_name in enumerate(self.__used_joints):
			self.__joint_name_idx_map[joint_name] = idx

		self.__symb_map = {}

		self.__link_id_name_map = {}
		self.__link_name_id_map = {}
		self.__link_id = 0

		self.__lambdified_fk_map = {}
		self.__lambdified_fd_map = {}

	def __getLinkId(self, link_name):
		if not link_name in self.__link_name_id_map:
			self.__link_name_id_map[link_name] = self.__link_id
			self.__link_id_name_map[self.__link_id] = link_name
			self.__link_id = self.__link_id + 1
		return self.__link_name_id_map[link_name]

	def __getLinkName(self, link_id):
		return self.__link_id_name_map[link_id]

	def __addSymb(self, symb_str, symb):
		assert not self.__hasSymb(symb_str)
		self.__symb_map[symb_str] = symb

	def __hasSymb(self, symb_str):
		return symb_str in self.__symb_map

	def __getSymb(self, symb_str):
		return self.__symb_map[symb_str]

	def __getSimplifiedChain(self, chain):
		new_chain = RobotSymbModel.Chain()
		frame_to_tip = PyKDL.Frame()
		open_seg = False
		for idx in range(chain.getNrOfSegments()):
			seg = chain.getSegment(idx)
			joint = seg.getJoint()
			frame_to_tip = frame_to_tip * seg.getFrameToTip()
			open_seg = True

			if joint.getTypeName() != 'None':
				new_joint = RobotSymbModel.Joint(joint)
				new_seg = RobotSymbModel.Segment(seg.getName(), frame_to_tip, new_joint)
				new_chain.addSegment(new_seg)
				frame_to_tip = PyKDL.Frame()
				open_seg = False
		if open_seg:
			print 'open seg: ', seg.getName(), joint.getName()
			# TODO: manage this case
		return new_chain

	def getFkFunc(self, link_name):
		if not link_name in self.__lambdified_fk_map:
			expr_R, expr_P = self.getFkSymb(link_name)
			args_list = []
			for q_idx in range(len(self.__used_joints)):
				theta = Symbol('q{}'.format(q_idx))
				args_list.append( theta )
			self.__lambdified_fk_map[link_name] = ( lambdify(args_list, expr_R), lambdify(args_list, expr_P) )
		return self.__lambdified_fk_map[link_name]

	def getFkSymb(self, link_name):
		base_link_id = self.__getLinkId(self.__base_link)
		link_id = self.__getLinkId(link_name)
		symb_P_str = 'P_{}a{}'.format(base_link_id, link_id)
		symb_R_str = 'R_{}a{}'.format(base_link_id, link_id)
		return self.__getSymb(symb_R_str), self.__getSymb(symb_P_str)

	def calculateFk(self, end_link):
		print '*** calculateFk from {} to {} ***'.format(self.__base_link, end_link)
		chain = self.__tree.getChain(self.__base_link, end_link)
		chain = self.__getSimplifiedChain(chain)

		base_link_id = self.__getLinkId(self.__base_link)
		symb_P_str = 'P_{}a{}'.format(base_link_id, base_link_id)
		symb_R_str = 'R_{}a{}'.format(base_link_id, base_link_id)
		if not self.__hasSymb(symb_P_str):
			self.__addSymb(symb_P_str, Matrix([[0],[0],[0]]))
		if not self.__hasSymb(symb_R_str):
			self.__addSymb(symb_R_str, Matrix([[1,0,0], [0,1,0], [0,0,1]]))

		link_id = base_link_id
		for idx in range(chain.getNrOfSegments()):

			print '***** {} *****'.format(idx)
			seg = chain.getSegment(idx)

			parent_link_id = link_id
			link_id = self.__getLinkId(seg.getName())

			joint = seg.getJoint()

			tfR = seg.getFrameToTip().M
			tfP = seg.getFrameToTip().p

			stfR = Matrix([[tfR[(0,0)], tfR[(1,0)], tfR[(2,0)]],
							[tfR[(0,1)], tfR[(1,1)], tfR[(2,1)]],
							[tfR[(0,2)], tfR[(1,2)], tfR[(2,2)]]]).T

			symb_P_prev_str = 'P_{}a{}'.format(base_link_id, parent_link_id)
			symb_R_prev_str = 'R_{}a{}'.format(base_link_id, parent_link_id)
			symb_P_next_str = 'P_{}a{}'.format(base_link_id, link_id)
			symb_R_next_str = 'R_{}a{}'.format(base_link_id, link_id)

			stfP = Matrix([[tfP.x()],[tfP.y()],[tfP.z()]])
			fk_tr = self.__getSymb(symb_P_prev_str) + self.__getSymb(symb_R_prev_str) * stfP
			fk_rot = self.__getSymb(symb_R_prev_str) * stfR

			print '  joint name: {}'.format(joint.getName())
			print '  joint axis: {}'.format(joint.JointAxis())
			print '  joint origin: {}'.format(joint.JointOrigin())
			print '  joint type: {}'.format(joint.getTypeName())
			print '  seg name: {}'.format(seg.getName())
			#inertia = seg.getInertia()
			#print '  seg inertia: {}, {}, {}'.format(inertia.getCOG(), inertia.getMass(), inertia.getRotationalInertia())
			print '  seg to tip: {}'.format(seg.getFrameToTip())
			#print '  seg pose: {}'.format(seg.pose(0))

			if joint.getTypeName() == 'None':
				raise Exception('Fixed joints are not supported')
			#if seg.getFrameToTip().M.GetRot().Norm() > 0.0000001:
			#	raise Exception('There is a bug in KDL/URDF for moveable joints with rotated origin')

			q_idx = self.__joint_name_idx_map[joint.getName()]
			theta = Symbol('q{}'.format(q_idx))
			#theta_vec_symb_map[joint.getName()] = theta
			t = (1-cos(theta))
			s = sin(theta)
			c = cos(theta)
			ax = joint.JointAxis()
			x = ax.x()
			y = ax.y()
			z = ax.z()
			R = Matrix([[t*x*x+c, t*x*y-z*s, t*x*z+y*s],
					[t*x*y+z*s, t*y*y+c, t*y*z-x*s],
					[t*x*z-y*s, t*y*z+x*s, t*z*z+c]])

			fk_rot = fk_rot * R
			pprint(R)

			if not self.__hasSymb(symb_P_next_str):
				self.__addSymb(symb_P_next_str, fk_tr)
			if not self.__hasSymb(symb_R_next_str):
				self.__addSymb(symb_R_next_str, fk_rot)

	def getFdFunc(self, link_name):
		if not link_name in self.__lambdified_fd_map:
			expr_w, expr_e, expr_A = self.getFdSymb(link_name)
			args_list = []
			for q_idx in range(len(self.__used_joints)):
				theta = Symbol('q{}'.format(q_idx))
				args_list.append( theta )
			for q_idx in range(len(self.__used_joints)):
				dtheta = Symbol('dtheta_{}'.format(q_idx))
				args_list.append( dtheta )
			for q_idx in range(len(self.__used_joints)):
				ddtheta = Symbol('ddtheta_{}'.format(q_idx))
				args_list.append( ddtheta )
			self.__lambdified_fd_map[link_name] = ( lambdify(args_list, expr_w), lambdify(args_list, expr_e),\
													lambdify(args_list, expr_A) )
		return self.__lambdified_fd_map[link_name]

	def getFdSymb(self, link_name):
		base_link_id = self.__getLinkId(self.__base_link)
		link_id = self.__getLinkId(link_name)
		symb_w_str = 'w_{}a{}'.format(base_link_id, link_id)
		symb_e_str = 'e_{}a{}'.format(base_link_id, link_id)
		symb_A_str = 'A_{}a{}'.format(base_link_id, link_id)
		return self.__getSymb(symb_w_str), self.__getSymb(symb_e_str), self.__getSymb(symb_A_str)

	def calculateFd(self, end_link):
		print '*** calculateFd from {} to {} ***'.format(self.__base_link, end_link)
		chain = self.__tree.getChain(self.__base_link, end_link)
		chain = self.__getSimplifiedChain(chain)

		base_link_id = self.__getLinkId(self.__base_link)
		symb_w_str = 'w_{}a{}'.format(base_link_id, base_link_id)
		symb_e_str = 'e_{}a{}'.format(base_link_id, base_link_id)
		symb_A_str = 'A_{}a{}'.format(base_link_id, base_link_id)
		if not self.__hasSymb(symb_w_str):
			self.__addSymb(symb_w_str, Matrix([[0],[0],[0]]))
		if not self.__hasSymb(symb_e_str):
			self.__addSymb(symb_e_str, Matrix([[0],[0],[0]]))
		if not self.__hasSymb(symb_A_str):
			self.__addSymb(symb_A_str, Matrix([[0],[0],[0]]))

		link_id = base_link_id
		for idx in range(chain.getNrOfSegments()):

			print '***** {} *****'.format(idx)
			seg = chain.getSegment(idx)

			parent_link_id = link_id
			link_id = self.__getLinkId(seg.getName())

			joint = seg.getJoint()

			tfR = seg.getFrameToTip().M
			tfP = seg.getFrameToTip().p

			stfR = Matrix([[tfR[(0,0)], tfR[(1,0)], tfR[(2,0)]],
							[tfR[(0,1)], tfR[(1,1)], tfR[(2,1)]],
							[tfR[(0,2)], tfR[(1,2)], tfR[(2,2)]]]).T

			stfP = Matrix([[tfP.x()],[tfP.y()],[tfP.z()]])

			print '  joint name: {}'.format(joint.getName())
			print '  seg name: {}'.format(seg.getName())

			if joint.getTypeName() == 'None':
				raise Exception('Fixed joints are not supported')

			q_idx = self.__joint_name_idx_map[joint.getName()]
			theta = Symbol('q{}'.format(q_idx))
			t = (1-cos(theta))
			s = sin(theta)
			c = cos(theta)
			ax = joint.JointAxis()
			x = ax.x()
			y = ax.y()
			z = ax.z()
			R = Matrix([[t*x*x+c, t*x*y-z*s, t*x*z+y*s],
					[t*x*y+z*s, t*y*y+c, t*y*z-x*s],
					[t*x*z-y*s, t*y*z+x*s, t*z*z+c]])

			dtheta = Symbol('dtheta_{}'.format(q_idx))
			ddtheta = Symbol('ddtheta_{}'.format(q_idx))

			# angular velocity:
			symb_w_str = 'w_{}a{}'.format(base_link_id, link_id)
			symb_w_prev_str = 'w_{}a{}'.format(base_link_id, parent_link_id)
			symb_w = (stfR * R).T * self.__getSymb(symb_w_prev_str) +\
						Matrix([[ax.x()],[ax.y()],[ax.z()]]) * dtheta
			if not self.__hasSymb(symb_w_str):
				self.__addSymb(symb_w_str, symb_w)

			# angular acceleration:
			symb_e_str = 'e_{}a{}'.format(base_link_id, link_id)
			symb_e_prev_str = 'e_{}a{}'.format(base_link_id, parent_link_id)
			symb_e = (stfR * R).T * self.__getSymb(symb_e_prev_str) +\
												((stfR * R).T * self.__getSymb(symb_w_prev_str)).cross(Matrix([[ax.x()],[ax.y()],[ax.z()]]) * dtheta) +\
												Matrix([[ax.x()],[ax.y()],[ax.z()]]) * ddtheta
			if not self.__hasSymb(symb_e_str):
				self.__addSymb(symb_e_str, symb_e)
			# linear acceleration:
			symb_A_str = 'A_{}a{}'.format(base_link_id, link_id)
			symb_A_prev_str = 'A_{}a{}'.format(base_link_id, parent_link_id)
			symb_A = (stfR * R).T * ( self.__getSymb(symb_A_prev_str) +\
							self.__getSymb(symb_e_prev_str).cross(stfP) +\
							self.__getSymb(symb_w_prev_str).cross(self.__getSymb(symb_w_prev_str).cross(stfP)))
			if not self.__hasSymb(symb_A_str):
				self.__addSymb(symb_A_str, symb_A)

model = RobotSymbModel(tree, base_link, used_joints)
for end_link_idx, end_link in enumerate(end_link_list):
	model.calculateFk(end_link)
	model.calculateFd(end_link)

visualize(model)

exit(0)

# print symbolic results
for end_link in end_link_list:
	print '*** FK for {}'.format(end_link)
	expr_R, expr_P = model.getFkSymb(end_link)
	print 'R = {}'.format(expr_R)
	print 'p = {}'.format(expr_P)

	print '*** FD (velocities and accelerations) for {}'.format(end_link)
	expr_w, expr_e, expr_A = model.getFdSymb(end_link)
	print 'w = {}'.format(expr_w)
	print 'e = {}'.format(expr_e)
	print 'A = {}'.format(expr_A)

exit(0)

symb_map = {}
for end_link_idx, end_link in enumerate(end_link_list):


	chain = tree.getChain(base_link, end_link)
	print '**** parsing chain from {} to {}, no. of joints: {}'.format(base_link, end_link, chain.getNrOfJoints())

	#seg = chain.getSegment(0)

	#joint = seg.getJoint()
	#print 'seg:', dir(seg)
	#print 'joint:', dir(joint)
	#print 'inertia:', dir(seg.getInertia())
	#print 'rot. inertia:', dir(PyKDL.RotationalInertia())

	#
	#
	#
	fk_tr = Matrix([[0],[0],[0]])
	fk_rot = Matrix([[1,0,0], [0,1,0], [0,0,1]])
	theta_vec_symb_map = {}
	
	symb_map = {'w_0a0':Matrix([[0],[0],[0]]),
				'e_0a0':Matrix([[0],[0],[0]]),
				'A_0a0':Matrix([[0],[0],[0]])}
	for idx in range(chain.getNrOfSegments()):

		print '***** {} *****'.format(idx)
		seg = chain.getSegment(idx)

		joint = seg.getJoint()

		tfR = seg.getFrameToTip().M
		tfP = seg.getFrameToTip().p

		stfR = Matrix([[tfR[(0,0)], tfR[(1,0)], tfR[(2,0)]],
						[tfR[(0,1)], tfR[(1,1)], tfR[(2,1)]],
						[tfR[(0,2)], tfR[(1,2)], tfR[(2,2)]]]).T

		stfP = Matrix([[tfP.x()],[tfP.y()],[tfP.z()]])
		fk_tr = fk_tr + fk_rot * stfP
		fk_rot = fk_rot * stfR

		print '  joint name: {}'.format(joint.getName())
		print '  joint axis: {}'.format(joint.JointAxis())
		print '  joint origin: {}'.format(joint.JointOrigin())
		print '  joint type: {}'.format(joint.getTypeName())
		print '  seg name: {}'.format(seg.getName())
		inertia = seg.getInertia()
		print '  seg inertia: {}, {}, {}'.format(inertia.getCOG(), inertia.getMass(), inertia.getRotationalInertia())
		print '  seg to tip: {}'.format(seg.getFrameToTip())
		print '  seg pose: {}'.format(seg.pose(0))

		if joint.getTypeName() != 'None':
			if seg.getFrameToTip().M.GetRot().Norm() > 0.0000001:
				raise Exception('There is a bug in KDL/URDF for moveable joints with rotated origin')

			q_idx = joint_name_idx_map[joint.getName()]
			theta = Symbol('q{}'.format(q_idx))
			theta_vec_symb_map[joint.getName()] = theta
			t = (1-cos(theta))
			s = sin(theta)
			c = cos(theta)
			ax = joint.JointAxis()
			x = ax.x()
			y = ax.y()
			z = ax.z()
			R = Matrix([[t*x*x+c, t*x*y-z*s, t*x*z+y*s],
					[t*x*y+z*s, t*y*y+c, t*y*z-x*s],
					[t*x*z-y*s, t*y*z+x*s, t*z*z+c]])

			fk_rot = fk_rot * R
			pprint(R)

			q_idx = joint_name_idx_map[joint.getName()]
			ax = joint.JointAxis()
			theta_dot = Symbol('dtheta_{}'.format(q_idx+1))
			theta_dot2 = Symbol('ddtheta_{}'.format(q_idx+1))
			#symb_map['dtheta_{}'.format(q_idx+1)] = theta_dot
			# angular velocity:
			symb_map['w_0a{}'.format(idx+1)] = stfR * R * symb_map['w_0a{}'.format(idx)] +\
						Matrix([[ax.x()],[ax.y()],[ax.z()]]) * theta_dot

			# angular acceleration:
			symb_map['e_0a{}'.format(idx+1)] = stfR * R * symb_map['e_0a{}'.format(idx)] +\
												(stfR * R * symb_map['w_0a{}'.format(idx)]).cross(Matrix([[ax.x()],[ax.y()],[ax.z()]]) * theta_dot) +\
												Matrix([[ax.x()],[ax.y()],[ax.z()]]) * theta_dot2

			# linear acceleration:
			symb_map['A_0a{}'.format(idx+1)] = stfR * R * ( symb_map['A_0a{}'.format(idx)] +\
							symb_map['e_0a{}'.format(idx)].cross(stfP) +\
							symb_map['w_0a{}'.format(idx)].cross(symb_map['w_0a{}'.format(idx)].cross(stfP)))


		else:
			symb_map['w_0a{}'.format(idx+1)] = stfR * symb_map['w_0a{}'.format(idx)]
			symb_map['e_0a{}'.format(idx+1)] = stfR * symb_map['e_0a{}'.format(idx)]
			symb_map['A_0a{}'.format(idx+1)] = stfR * (symb_map['A_0a{}'.format(idx)] +\
							symb_map['e_0a{}'.format(idx)].cross(stfP) +\
							symb_map['w_0a{}'.format(idx)].cross(symb_map['w_0a{}'.format(idx)].cross(stfP)))

	fk_rot = replaceConstants(fk_rot)
	fk_tr = replaceConstants(fk_tr)

	matlab_code += '% chain from {} to {}\n'.format(base_link, end_link)
	s = str(fk_rot)
	for q_idx in reversed(range(len(used_joints))):
		s = s.replace('sin(q{})'.format(q_idx), 's({})'.format(q_idx+1))
		s = s.replace('cos(q{})'.format(q_idx), 'c({})'.format(q_idx+1))
	s = s.replace('Matrix(', 'R{} = '.format(end_link_idx+1))
	s = s.replace(']])', ']]')
	matlab_code += s + ';\n'

	s = str(fk_tr)
	for q_idx in reversed(range(len(used_joints))):
		s = s.replace('sin(q{})'.format(q_idx), 's({})'.format(q_idx+1))
		s = s.replace('cos(q{})'.format(q_idx), 'c({})'.format(q_idx+1))
	s = s.replace('Matrix(', 'p{} = '.format(end_link_idx+1))
	s = s.replace(']])', ']]')
	matlab_code += s + ';\n'

	print '*** Comparison of solutions'

	q = [1, 0.2, -1, 0.2, -0.1, 0.5, -0.4, 0.2, 0.5, 0.3, -0.4, -0.1, -0.6, 1, 0.2, -0.5, 1.2]
	assert len(q) == len(used_joints)
	subs_list = []
	for name, theta in theta_vec_symb_map.iteritems():
		subs_list.append( (theta, q[joint_name_idx_map[name]]) )
	subs_list.append( (Symbol('b'), 0.86602540378614) )
	print 'sympy solution:'

	pprint(fk_rot.subs( subs_list ).evalf())
	pprint(fk_tr.subs( subs_list ).evalf())

	# Calculate FK with KDL
	end_pose = PyKDL.Frame()
	for idx in range(chain.getNrOfSegments()):
		seg = chain.getSegment(idx)
		joint = seg.getJoint()
		if joint.getTypeName() == 'None':
			end_pose = end_pose * seg.pose(0)
		else:
			q_idx = joint_name_idx_map[joint.getName()]
			print joint.getName(), q_idx, q[q_idx]
			end_pose = end_pose * seg.pose( q[q_idx] )

	print 'KDL solution:'
	print end_pose

print '% *** Matlab code:'
print matlab_code

print symb_map.keys()
expr = replaceConstants(symb_map['A_0a9'])

q = [0,0,-1,0,-1,0,-0.5,0,0,0,0,0,0,0,0,0,0]
subs_list = []
for q_idx in range(len(used_joints)):
	theta_dot = Symbol('dtheta_{}'.format(q_idx+1))
	theta_dot2 = Symbol('ddtheta_{}'.format(q_idx+1))
	if used_joints[q_idx] == 'torso_0_joint':
		subs_list.append( (theta_dot, -1) )
	else:		
		subs_list.append( (theta_dot, 0) )
	subs_list.append( (theta_dot2, 0) )
for name, theta in theta_vec_symb_map.iteritems():
	subs_list.append( (theta, q[joint_name_idx_map[name]]) )
subs_list.append( (Symbol('b'), 0.86602540378614) )
print expr.subs( subs_list ).evalf()
#pprint(expr)
exit(0)
