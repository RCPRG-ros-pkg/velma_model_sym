#!/usr/bin/env python

# This script calculates symbolic model for forward kinematics
# for a robot defined in an URDF file loaded from ros param.
#
# Author: Dawid Seredynski, 2020
#

from sympy import pprint, init_printing, Symbol, sin, cos, exp, sqrt, series,\
					Integral, Function, Matrix, sympify

from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
import PyKDL

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
for end_link_idx, end_link in enumerate(end_link_list):
	chain = tree.getChain(base_link, end_link)
	print '**** parsing chain from {} to {}, no. of joints: {}'.format(base_link, end_link, chain.getNrOfJoints())

	seg = chain.getSegment(0)
	joint = seg.getJoint()
	print 'seg:', dir(seg)
	print 'joint:', dir(joint)
	print 'inertia:', dir(seg.getInertia())
	print 'rot. inertia:', dir(PyKDL.RotationalInertia())

	joint_name_idx_map = {}
	for idx, joint_name in enumerate(used_joints):
		joint_name_idx_map[joint_name] = idx
	current_frame = PyKDL.Frame()

	dh_map = {}

	#
	#
	#
	print PyKDL.__file__
	axes = []
	fk_tr = Matrix([[0],[0],[0]])
	fk_rot = Matrix([[1,0,0], [0,1,0], [0,0,1]])
	fk_sol_list = []
	theta_vec_symb_map = {}
	for idx in range(chain.getNrOfSegments()):
		print '***** {} *****'.format(idx)
		seg = chain.getSegment(idx)
		joint = seg.getJoint()

		tfR = seg.getFrameToTip().M
		tfP = seg.getFrameToTip().p

		stfR = Matrix([[tfR[(0,0)], tfR[(1,0)], tfR[(2,0)]],
						[tfR[(0,1)], tfR[(1,1)], tfR[(2,1)]],
						[tfR[(0,2)], tfR[(1,2)], tfR[(2,2)]]]).T

		fk_tr = fk_tr + fk_rot * Matrix([[tfP.x()],[tfP.y()],[tfP.z()]])
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
			fk_sol_list.append( (fk_rot, fk_tr) )

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

exit(0)
