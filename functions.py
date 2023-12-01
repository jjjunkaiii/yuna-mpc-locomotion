import numpy as np
from robot_setup.yunaKinematics import HexapodKinematics

def rot(pos, angle, pivot=np.array([0,0,0])):
    '''
    Calculate the rotated position of a point around z-axis where a pivot point lies
    :param pos: the position of the point to be rotated
    :param angle: the angle of rotation in radians, counterclockwise is positive
    :param pivot: the centre of rotation, only (x ,y) is effective
    :return: the rotated position
    '''
    c, s = np.cos(angle), np.sin(angle)
    rot_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pos_ = np.matmul(rot_z, pos - pivot)
    return pos_ + pivot

def trans(pos, distance, angle):
    '''
    Calculate the translated position of a point in a direction
    :param pos: the position of the point to be translated
    :param distance: the distance of translation
    :param angle: the direction of translation in radians, counterclockwise is positive
    :return: the translated position
    '''
    pos_ = np.copy(pos)
    pos_[0] += distance * np.cos(angle)
    pos_[1] += distance * np.sin(angle)
    return pos_

def skew_matrix(vector):
    '''
    Generate the skew-symmetric matrix of a vector, which is used to calculate the cross product
    :param vector: the vector to be calculated
    :return: the skew-symmetric matrix of the vector
    '''
    return np.array([[0, -vector[2], vector[1]], [vector[2], 0, -vector[0]], [-vector[1], vector[0], 0]])

def hebi2bullet(jointspace_command2hebi):
    '''
    Convert the jointspace command for hebi to the jointspace command for pybullet
    :param jointspace_command2hebi: the jointspace command for hebi, whose size is (18,) and order is [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    :return: the jointspace command for pybullet, whose size is (18,) and order is [1,2,3,7,8,9,13,14,15,4,5,6,10,11,12,16,17,18]
    '''
    if type(jointspace_command2hebi) == list:
        jointspace_command2hebi = np.array(jointspace_command2hebi)
    jointspace_command2bullet = jointspace_command2hebi[[0,1,2,6,7,8,12,13,14,3,4,5,9,10,11,15,16,17,]].copy()# reshaped the IK result: 123456->135246
    return jointspace_command2bullet

def bullet2hebi(jointspace_command2bullet):
    '''
    Convert the jointspace command for pybullet to the jointspace command for hebi
    :param jointspace_command2bullet: the jointspace command for pybullet, whose size is (18,) and order is [1,2,3,7,8,9,13,14,15,4,5,6,10,11,12,16,17,18]
    :return: the jointspace command for hebi, whose size is (18,) and order is [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    '''
    if type(jointspace_command2bullet) == list:
        jointspace_command2bullet = np.array(jointspace_command2bullet)
    jointspace_command2hebi = jointspace_command2bullet[[0,1,2,9,10,11,3,4,5,12,13,14,6,7,8,15,16,17,]].copy()# reshaped the IK result: 135246->123456
    return jointspace_command2hebi
    
def solveIK(workspace_command):
    '''
    Use HexapodKinematics module to solve the inverse kinematics
    :param workspace_command: the workspace command, whose size is (3, 6)
    :return: the jointspace command for pybullet, the jointspace command for hebi
    '''
    xmk = HexapodKinematics()
    jointspace_command2hebi = xmk.getLegIK(workspace_command)
    jointspace_command2bullet = hebi2bullet(jointspace_command2hebi)
    return jointspace_command2bullet, jointspace_command2hebi

def solveFK(jointspace_command2hebi):
    '''
    Use HexapodKinematics module to solve the forward kinematics
    :param jointspace_command2hebi: the jointspace command for hebi, whose size is (18,) and order is [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    :return: the workspace command, whose size is (3, 6)
    '''
    xmk = HexapodKinematics()
    workspace_command = xmk.getLegPositions(np.array([jointspace_command2hebi]))
    return workspace_command