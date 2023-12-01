import pybullet as p
import pybullet_data
import numpy as np
import os, sys
import time
import collections
import robot_setup
from robot_setup.yunaKinematics import *
from functions import hebi2bullet, bullet2hebi, solveIK, solveFK

class YunaEnv:
    def __init__(self, real_robot_control=False, pybullet_on=True, visualiser=True, camerafollow=True, complex_terrain=False):
        self.real_robot_control = real_robot_control
        self.visualiser = visualiser
        self.camerafollow = camerafollow
        self.dt = 1 /240
        self.xmk, self.imu, self.hexapod, self.fbk_imu, self.fbk_hp, self.group_command, self.group_feedback = self.robot_connect()
        self.error = np.zeros((18,))
        self.p = p
        self.time_steps = 0
        self.complex_terrain = complex_terrain
        self.pybullet_on = pybullet_on
        if self.pybullet_on:
            self._load_env()
        self._init_robot()
        self.reset_velocity_estimator()

    def step(self, command, iteration=1, sleep='auto'):
        for i in range(iteration):
            t_start = time.perf_counter()
            # pybullet control
            if self.pybullet_on:
                # set joint command
                for joint_index in range(18):
                    if command[0,joint_index] == self.p.TORQUE_CONTROL:
                        self.p.setJointMotorControl2(
                            bodyIndex=self.YunaID, 
                            jointIndex=self.actuator[joint_index], 
                            controlMode=self.p.TORQUE_CONTROL,
                            force=command[self.p.TORQUE_CONTROL,joint_index])
                    else:
                        self.p.setJointMotorControl2(
                            bodyIndex=self.YunaID, 
                            jointIndex=self.actuator[joint_index], 
                            controlMode=self.p.POSITION_CONTROL,
                            targetPosition=command[self.p.POSITION_CONTROL,joint_index],
                            force=40,) # positionGain=50,velocityGain=0.03, maxVelocity=3.14 strategy 4 in hebi doc
                        
                self.p.stepSimulation()
                if self.camerafollow:
                    self._cam_follow()
            t_stop = time.perf_counter()
            t_step = t_stop - t_start
            if sleep == 'auto':
                time.sleep(max(0, self.dt - t_step))
            else:
                time.sleep(sleep)
            self.time_steps += 1
        self.reset_joints()

    def close(self):
        '''
        Close the pybullet simulation, disable the real robot motors, and terminate the program
        :return: None
        '''
        if self.real_robot_control:
                arr = np.zeros([1, 18])[0]
                self.group_command.effort = arr
                self.group_command.position = np.nan * arr
                self.group_command.velocity_limit_max = arr
                self.group_command.velocity_limit_min = arr
                self.hexapod.send_command(self.group_command)
        if self.pybullet_on:
            try:
                self.p.disconnect()
            except self.p.error as e:
                print('Termination of simulation failed:', e)
        sys.exit()

    def robot_connect(self):
        '''
        Initialise connection to the real robot
        :return: xmk, imu, hexapod, fbk_imu, fbk_hp, group_command, group_feedback
        '''
        if self.real_robot_control:
            xmk, imu, hexapod, fbk_imu, fbk_hp = robot_setup.setup_xmonster()
            group_command = hebi.GroupCommand(hexapod.size)
            group_feedback = hebi.GroupFeedback(hexapod.size)
            hexapod.feedback_frequency = 100.0
            hexapod.command_lifetime = 0
            while True:
                group_feedback = hexapod.get_next_feedback(reuse_fbk=group_feedback)
                if type(group_feedback) != None:
                    break
            return xmk, imu, hexapod, fbk_imu, fbk_hp, group_command, group_feedback
        else:
            return HexapodKinematics(), False, False, False, False, False, False

    def _load_env(self):
        '''
        Load and initialise the pybullet simulation environment
        :return: None
        '''
        # initialise interface
        if self.visualiser:
            self.physicsClient = self.p.connect(self.p.GUI)
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING, 0)
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0)
        else:
            self.physicsClient = self.p.connect(self.p.DIRECT)
        # physical parameters
        self.gravity = -9.81
        self.friction = 0.7#0.7
        # load ground
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        if self.complex_terrain:
            heightPerturbationRange = 0.1
            numHeightfieldRows = 256
            numHeightfieldColumns = 256
            heightfieldData = [0]*numHeightfieldRows*numHeightfieldColumns 
            for j in range (int(numHeightfieldColumns/2)):
                for i in range (int(numHeightfieldRows/2) ):
                    height = random.uniform(0,heightPerturbationRange)
                    heightfieldData[2*i+2*j*numHeightfieldRows]=height
                    heightfieldData[2*i+1+2*j*numHeightfieldRows]=height
                    heightfieldData[2*i+(2*j+1)*numHeightfieldRows]=height
                    heightfieldData[2*i+1+(2*j+1)*numHeightfieldRows]=height
            
            terrainShape = p.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD, meshScale=[.05,.05,1], heightfieldTextureScaling=(numHeightfieldRows-1)/2, heightfieldData=heightfieldData, numHeightfieldRows=numHeightfieldRows, numHeightfieldColumns=numHeightfieldColumns)
            self.groundID  = p.createMultiBody(0, terrainShape)
        else:
            self.groundID = self.p.loadURDF('plane.urdf')
        self.p.changeDynamics(self.groundID, -1, lateralFriction=self.friction)
        self.p.setGravity(0, 0, self.gravity)
        # load Yuna robot
        Yuna_init_pos = [0,0,0.5]
        Yuna_init_orn = self.p.getQuaternionFromEuler([0,0,0])
        Yuna_file_path = os.path.abspath(os.path.dirname(__file__)) + '/urdf/yuna.urdf'
        self.YunaID = self.p.loadURDF(Yuna_file_path, Yuna_init_pos, Yuna_init_orn)
        self.joint_num = self.p.getNumJoints(self.YunaID)
        self.actuator = [i for i in range(self.joint_num) if self.p.getJointInfo(self.YunaID,i)[2] != p.JOINT_FIXED]
        self._get_foot_id()
        
        if self.visualiser:
            # self._add_reference_line()
            self._cam_follow()
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING, 1)
            
    def _init_robot(self):
        '''
        Initialise the robot to the neutral position in the pybullet simulation
        :return: None
        '''
        # parameters
        self.h = 0.2249#0.12 # body height
        self.eePos = np.array( [[0.51589,    0.51589,   0.0575,     0.0575,     -0.45839,   -0.45839],
                                [0.23145,   -0.23145,   0.5125,     -0.5125,    0.33105,    -0.33105],
                                [-self.h,   -self.h,    -self.h,    -self.h,    -self.h,    -self.h]]) # neutral position for the robot
        init_pos = self.eePos.copy()
        init_joint_angles, _ = solveIK(init_pos)
        command = np.zeros((3,18))
        command[0,:] = [self.p.POSITION_CONTROL for i in range(18)]
        command[self.p.POSITION_CONTROL,:] = init_joint_angles # command[2,:]
        self.step(command, iteration=180, sleep='auto') # when iteration=180, the robot is fully on the ground and stable
        self.time_steps = 0 # reset time steps counter

    def reset_joints(self):
        '''
        Reset the robot joints in the pybullet simulation
        :return: None
        '''
        for joint_id in range(18):
            self.p.setJointMotorControl2(self.YunaID, self.actuator[joint_id], controlMode=self.p.VELOCITY_CONTROL,force=0) #reset joints

    def _cam_follow(self):
        '''
        Follow the robot with the camera in the pybullet simulation
        :return: None
        '''
        def _get_body_pose(self):
            '''
            Get the position and orientation of the robot in the pybullet simulation
            :return pos: position of the robot in world frame
            :return orn: orientation of the robot in world frame in Euler angles
            '''
            pos, orn = self.p.getBasePositionAndOrientation(self.YunaID)
            return pos, self.p.getEulerFromQuaternion(orn)
        cam_pos, cam_orn = _get_body_pose(self)
        # p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=np.rad2deg(cam_orn[2])-90, cameraPitch=-35, cameraTargetPosition=cam_pos)#
        self.p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=-90, cameraPitch=-35, cameraTargetPosition=cam_pos)

    def _add_reference_line(self):
        '''
        Add some reference lines to the pybullet simulation
        :return: None
        '''
        self.p.addUserDebugLine(lineFromXYZ=[-100,-100,0], lineToXYZ=[100,100,0], lineColorRGB=[0.5,0,0], lineWidth=1)
        self.p.addUserDebugLine(lineFromXYZ=[-100,100,0], lineToXYZ=[100,-100,0], lineColorRGB=[0.5,0,0], lineWidth=1)
        self.p.addUserDebugLine(lineFromXYZ=[-100,-173.2,0], lineToXYZ=[100,173.2,0], lineColorRGB=[0,0,0.5], lineWidth=1)
        self.p.addUserDebugLine(lineFromXYZ=[-100,173.2,0], lineToXYZ=[100,-173.2,0], lineColorRGB=[0,0,0.5], lineWidth=1)
        self.p.addUserDebugLine(lineFromXYZ=[-173.2,-100,0], lineToXYZ=[173.2,100,0], lineColorRGB=[0,0,0.5], lineWidth=1)
        self.p.addUserDebugLine(lineFromXYZ=[-173.2,100,0], lineToXYZ=[173.2,-100,0], lineColorRGB=[0,0,0.5], lineWidth=1)
        self.p.addUserDebugLine(lineFromXYZ=[-100,0,0], lineToXYZ=[100,0,0], lineColorRGB=[0,0,0], lineWidth=1)
        self.p.addUserDebugLine(lineFromXYZ=[0,-100,0], lineToXYZ=[0,100,0], lineColorRGB=[0,0,0], lineWidth=1)
    
    def get_robot_config(self):
        '''
        Get the robot joint configuration
        :return robot_config: robot joint configuration
        '''
        if self.real_robot_control:
            robot_config = self.group_feedback.position
        else:
            robot_config = bullet2hebi(np.array([self.p.getJointState(self.YunaID, i)[0] for i in self.actuator]))

        return robot_config
    
    def get_robot_joint_velocity(self):
        '''
        Get the robot joint velocity
        :return robot_joint_velocity: robot joint velocity
        '''
        robot_joint_velocity = bullet2hebi(np.array([self.p.getJointState(self.YunaID, i)[1] for i in self.actuator]))
        return robot_joint_velocity

    def get_foot_position_in_body_frame(self):
        '''
        Get the foot position in the body frame
        :return foot_pos: foot position in the body frame
        '''
        foot_pos = solveFK(self.get_robot_config()) 
        return foot_pos
    
    def get_foot_position_in_CoM_frame(self):
        '''
        Get the foot position in the CoM frame, for the CoM does not coincide with the origin of the body frame
        :return foot_pos: foot position in the inertia frame
        '''
        CoM_offset = np.array([-0.0028085, -0.0003448, 0.0270705])
        foot_pos = self.get_foot_position_in_body_frame()
        for i in range(3):
            foot_pos[i,:] -= CoM_offset[i]
        return foot_pos
    
    def get_foot_contact(self):
        '''
        Get the contact foot, 1 means contact, 0 means no contact
        :return contact_foot: contact foot
        '''
        contact_points = self.p.getContactPoints(self.groundID, self.YunaID)
        foot_contact = np.zeros((6,))
        for i in contact_points:
            if np.isin(i[4], self.footID):
                foot_contact[np.where(self.footID==i[4])[0][0]] = 1
        return foot_contact.astype(int)

    def _get_foot_id(self):
        '''
        Get the foot link urdf ID in the pybullet simulation
        :return: None
        '''
        link_name = ['leg1__LAST_LINK',
                    'leg2__LAST_LINK',
                    'leg3__LAST_LINK',
                    'leg4__LAST_LINK',
                    'leg5__LAST_LINK',
                    'leg6__LAST_LINK']
        leg_counter = 0
        self.footID = np.zeros((6,))
        while leg_counter < 6:
            for i in range(self.p.getNumJoints(self.YunaID)):
                if self.p.getJointInfo(self.YunaID, i)[12].decode('UTF-8') == link_name[leg_counter]:
                    self.footID[leg_counter] = i
                    leg_counter += 1
                    break

    def reset_velocity_estimator(self):
        '''
        Reset the velocity estimator
        :return: None
        '''
        self._velocity_filter_x = WindowFilter()
        self._velocity_filter_y = WindowFilter()
        self._velocity_filter_z = WindowFilter()
        self._com_velocity_world_frame = np.zeros((3,))
        self._com_velocity_body_frame = np.zeros((3,))

    def update_filtered_velocity(self):
        '''
        Update the velocity estimator
        :return: None
        '''
        velocity = self.get_robot_linear_velocity()
        # velocity in world frame
        vx = self._velocity_filter_x.calculate_average(velocity[0])
        vy = self._velocity_filter_y.calculate_average(velocity[1])
        vz = self._velocity_filter_z.calculate_average(velocity[2])
        self._com_velocity_world_frame = np.array([vx, vy, vz])
        # velocity in body frame
        pos, orn = self.p.getBasePositionAndOrientation(self.YunaID)
        _, inv_rotation_mat= p.invertTransform((0,0,0), orn)
        self._com_velocity_body_frame, _ = self.p.multiplyTransforms((0,0,0), inv_rotation_mat, self._com_velocity_world_frame, (0,0,0,1))
        self._com_velocity_body_frame = np.asarray(self._com_velocity_body_frame)
    # get robot states in body frame
    def get_robot_linear_velocity(self):
        return self.p.getBaseVelocity(self.YunaID)[0]

    def get_robot_linear_velocity_filtered(self):
        return self._com_velocity_body_frame
    
    def get_robot_angular_velocity(self):
        orn_world = self.p.getBasePositionAndOrientation(self.YunaID)[1]
        orn_inv = self.p.invertTransform((0,0,0), orn_world)[1]
        omega_world = self.p.getBaseVelocity(self.YunaID)[1]
        omega_body = self.p.multiplyTransforms((0,0,0), orn_inv, omega_world, (0,0,0,1))[0]
        return np.asarray(omega_body)

    def get_robot_position(self): # only care about the CoM height in world frame
        return np.array((0., 0., self.get_robot_height()))
    
    def get_robot_orientation(self): # in Euler angles
        orn = self.p.getBasePositionAndOrientation(self.YunaID)[1]
        return np.asarray(self.p.getEulerFromQuaternion(orn))
    
    def get_robot_height(self):
        contact = self.get_foot_contact()
        if np.sum(contact) == 0:
            return self.h # can't estimate when all feet are in the air
        else:
            # get base orientation
            pos, orn = self.p.getBasePositionAndOrientation(self.YunaID)
            rot_mat = self.p.getMatrixFromQuaternion(orn)
            rot_mat = np.array(rot_mat).reshape((3, 3))
            # get foot position in body frame
            foot_pos_body = self.get_foot_position_in_body_frame()
            # get foot position in world frame
            foot_pos_world = rot_mat @ foot_pos_body
            # get average height of feet in contact with ground
            useful_heights = contact * (-foot_pos_world[2, :])
            return np.sum(useful_heights) / np.sum(contact)

    def get_jacobian(self):
        '''
        Get the Jacobian matrix of the robot
        :return: Jacobian matrix
        '''
        return self.xmk.getLegJacobians(np.array([self.get_robot_config()]))

    @property
    def time_since_initalisation(self):
        return self.time_steps * self.dt
    
    ### robot parameters ###
    @property
    def robot_body_mass(self):
        mass =1.8476+0.564+(1.2804+1.4679+1.58985+0.229433)*6 # the value is 29.817098
        return -mass * 1 #TODO

    @property
    def robot_body_inertia(self):
        return np.diag([0.017, 0.057, 0.064]) * 8 #TODO
    ########################

class WindowFilter:
    def __init__(self, window_size=20):
        self.window_size = window_size
        self._value_deque= collections.deque(maxlen=self.window_size) # use deque for fast append and pop, list can also be used but slower
        self._sum = 0
        self._correction = 0

    def _neumaier_sum(self, value):
        """
        Update the moving window sum using Neumaier's algorithm.
        For more details please refer to:
        https://en.wikipedia.org/wiki/Kahan_summation_algorithm
        :param value: the new value to be added to the window
        :return: None
        """
        new_sum = self._sum + value
        if abs(self._sum) >= abs(value):
            self._correction += (self._sum - new_sum) + value # If self._sum is bigger, low-order digits of value are lost.
        else:
            self._correction += (value - new_sum) + self._sum # low-order digits of sum are lost
        self._sum = new_sum

    def calculate_average(self, value):
        '''
        Calculate the moving window average
        :param value: the new value to be added to the window
        :return: the moving window average
        '''
        if len(self._value_deque) < self.window_size:
            pass
        else:
            self._neumaier_sum(-self._value_deque[0]) # subtract the oldest value
        
        self._neumaier_sum(value)
        self._value_deque.append(value)
        return (self._sum + self._correction) / self.window_size

if __name__=='__main__':
    # test code
    yunaenv = YunaEnv(real_robot_control=0)
    time.sleep(3)
    yunaenv.close()
