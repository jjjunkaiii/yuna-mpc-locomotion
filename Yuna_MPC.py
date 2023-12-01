from Yuna_Env import YunaEnv
import numpy as np
import quadprog
from functions import skew_matrix, rot, solveIK, hebi2bullet

class YunaMPC():
    def __init__(self, env):
        self.env = env
        self.gait_scheduler = Gait_scheduler(self.env)
        self.swing_leg_controller = Swing_leg_controller(self.env, self.gait_scheduler)
        self.stance_leg_controller = Stance_leg_controller(self.env, self.gait_scheduler)
        self.last_action = None #np.array([self.env.p.getJointState(self.env.YunaID, i)[3] for i in self.env.actuator])

    def get_action(self):
        control_mode = self.gait_scheduler.joint_control_mode
        joint_stance_torques = self.stance_leg_controller.get_action()
        joint_swing_angles = self.swing_leg_controller.get_action()
        action = np.vstack((control_mode, joint_stance_torques, joint_swing_angles))
                
        return action

    def get_command(self, vel_l, vel_a):
        self.swing_leg_controller.get_command(vel_l, vel_a)
        self.stance_leg_controller.get_command(vel_l, vel_a)

    def update(self):
        self.env.update_filtered_velocity()
        self.gait_scheduler.update()
        self.swing_leg_controller.update()


class Swing_leg_controller():
    def __init__(self, env, gait_scheduler):
        self.env = env
        self.gait_scheduler = gait_scheduler
        self.max_step_length = 0.2
        self.max_turn_angle = 20 / 180 * np.pi
        # speed correction coefficients TODO: tune this later
        self.rho = 0.8 # can't be too small, or robot will be too sensitive to the outside disturbance, leave it big to ensure the stability
        self.des_vel_l = np.array((0., 0., 0.))
        self.des_vel_a = np.array((0.,))
        self._des_leg_state = self.gait_scheduler.desired_leg_state # record previous desired leg state
        self.init_pos = self.env.get_foot_position_in_body_frame()
    
    def get_action(self):
        # self._compute_step_param()
        end_pos = self._get_end_pos()
        init_pos = self.init_pos.copy()
        phase = self.gait_scheduler.sub_phase
        current_pos = init_pos.copy()
        for leg_id in range(6):
            if self.des_leg_state[leg_id] == self.gait_scheduler.SWING: #TODO： coordinate with stance leg controller
                current_pos[:, leg_id] = self._compute_trajectory(init_pos[:, leg_id], end_pos[:, leg_id], phase[leg_id])
        joint_angles, _ = solveIK(current_pos) # joint command is in bullet order
        #joint_torques = self.convert_angle2torque(joint_angles)
        return joint_angles#joint_torques
    
    def convert_angle2torque(self, angles):
        observed_angle = hebi2bullet(self.env.get_robot_config())
        command_angle = angles
        observed_velocity = hebi2bullet(self.env.get_robot_joint_velocity())
        command_velocity = np.zeros(18)
        kp = 30
        kd = 0.1
        torque = kp * (command_angle - observed_angle) + kd * (command_velocity - observed_velocity)
        return torque

    def get_command(self, vel_l, vel_a):
        self.des_vel_l = np.array(vel_l, dtype=float) # linear velocity
        self.des_vel_a = float(vel_a) # angular (yaw specifically) velocity

    def _compute_step_param(self):
        # get actual CoM linear and angular velocity
        vel_l = self.env.get_robot_linear_velocity_filtered()
        vel_a = self.env.get_robot_angular_velocity()[2]
        # get desired CoM linear and angular velocity
        des_vel_l = self.des_vel_l
        des_vel_a = self.des_vel_a
        # combine actual and desired CoM linear and angular velocity
        command_vel_l = vel_l * (1 - self.rho) + des_vel_l * self.rho
        command_vel_a = vel_a * (1 - self.rho) + des_vel_a * self.rho
        # compute step length, course and rotation
        disp_l = command_vel_l * self.gait_scheduler.swing_duration * self.gait_scheduler.duty_factor
        disp_l[2] = 0
        disp_a = command_vel_a * self.gait_scheduler.swing_duration * self.gait_scheduler.duty_factor
        disp_l = np.clip(disp_l, -self.max_step_length, self.max_step_length)
        disp_a = np.clip(disp_a, -self.max_turn_angle, self.max_turn_angle)
        return disp_l, disp_a
    
    def update(self):
        # update init_pos and des_leg_state
        self.des_leg_state = self.gait_scheduler.desired_leg_state
        for leg_id, state in enumerate(self.des_leg_state):
            _state = self._des_leg_state[leg_id]
            if state == self.gait_scheduler.SWING and _state != self.gait_scheduler.SWING:
                # this is the phase changing timestep
                self.init_pos[:, leg_id] = self.env.get_foot_position_in_body_frame()[:, leg_id]
        self._des_leg_state = self.des_leg_state.copy()

    def _get_end_pos(self):
        neutral_pos = self.env.eePos.copy()
        end_pos = np.copy(neutral_pos)
        disp_l, disp_a = self._compute_step_param()
        leg_state = self.gait_scheduler.leg_state
        for leg_id in range(6):
            neutral_pos_leg = neutral_pos[:, leg_id]
            end_pos[:, leg_id] = rot(pos=neutral_pos_leg+disp_l, angle=disp_a, pivot=disp_l)#TODO: check if it's minus or plus in disp_l and disp_a
        return end_pos
                
    def _compute_trajectory(self, init_pos, end_pos, phase):
        # We augment the swing speed using the below formula. For the first half of
        # the swing cycle, the swing leg moves faster and finishes 80% of the full
        # swing trajectory. The rest 20% of trajectory takes another half swing
        # cycle. Intuitely, we want to move the swing foot quickly to the target
        # landing location and stay above the ground, in this way the control is more
        # robust to perturbations to the body that may cause the swing foot to drop
        # onto the ground earlier than expected. This is a common practice similar
        # to the MIT cheetah and Marc Raibert's original controllers.
        if phase <= 0.5:
            phase = 0.8 * np.sin(phase * np.pi)
        else:
            phase = 0.8 + (phase - 0.5) * 0.4 #TODO: phase should have a limit, or robot will reach kinematic limit
        x = (1 - phase) * init_pos[0] + phase * end_pos[0]
        y = (1 - phase) * init_pos[1] + phase * end_pos[1]
        max_clearance = self.env.h # set the highest point the same as robot body height
        mid = max(end_pos[2], init_pos[2]) + max_clearance
        z = self._gen_parabola(phase, init_pos[2], mid, end_pos[2])
        traj= np.array([x, y, z])
        return traj
    
    def _gen_parabola(self, phase, start, mid, end):
        """Gets a point on a parabola y = a x^2 + b x + c.

        The Parabola is determined by three points (0, start), (0.5, mid), (1, end) in
        the plane.

        Args:
            phase: Normalized to [0, 1]. A point on the x-axis of the parabola.
            start: The y value at x == 0.
            mid: The y value at x == 0.5.
            end: The y value at x == 1.

        Returns:
            The y value at x == phase.
        """
        mid_phase = 0.5
        delta_1 = mid - start
        delta_2 = end - start
        delta_3 = mid_phase**2 - mid_phase
        coef_a = (delta_1 - delta_2 * mid_phase) / delta_3
        coef_b = (delta_2 * mid_phase**2 - delta_1) / delta_3
        coef_c = start

        return coef_a * phase**2 + coef_b * phase + coef_c


class Stance_leg_controller():
    def __init__(self, env, gait_scheduler):
        self.env = env
        self.gait_scheduler = gait_scheduler
        self.KP = np.array((0., 0., 100., 100., 100., 0.))*0.1 # TODO
        self.KD = np.array((40., 30., 10., 10., 10., 30.))*0.1 # TODO
        self.MAX_DDQ = np.array((10., 10., 10., 20., 20., 20.))
        self.MIN_DDQ = -self.MAX_DDQ

    def get_action(self):
        des_acc = self._get_desired_acc()
        contact_force = self._compute_contact_force(des_acc)
        joint_torques = self._map_force_to_torque(contact_force)
        return joint_torques
    
    def get_command(self, vel_l, vel_a):
        self.des_vel_l = vel_l
        self.des_vel_a = vel_a

    def _get_desired_acc(self):
        # get actual CoM position and velocity
        pos_l = self.env.get_robot_position()
        vel_l = self.env.get_robot_linear_velocity_filtered()
        pos_a = self.env.get_robot_orientation()
        vel_a = self.env.get_robot_angular_velocity()
        q = np.hstack((pos_l, pos_a))
        q_dot = np.hstack((vel_l, vel_a))
        # get desired CoM position and velocity
        des_pos_l = np.array([0., 0., self.env.h])
        des_vel_l = np.array([self.des_vel_l[0], self.des_vel_l[1], 0.])
        des_pos_a = np.array([0., 0., 0.])
        des_vel_a = np.array([0., 0., self.des_vel_a])
        des_q = np.hstack((des_pos_l, des_pos_a))
        des_q_dot = np.hstack((des_vel_l, des_vel_a))
        # compute desired acceleration
        des_q_ddot = self.KP * (des_q - q) + self.KD * (des_q_dot - q_dot)
        des_q_ddot = np.clip(des_q_ddot, self.MIN_DDQ, self.MAX_DDQ)
        return des_q_ddot

    def _compute_mass_matrix(self):
        rot_z = np.eye(3) # mass matrix is in body frame so no rotation about z axis
        body_mass = self.env.robot_body_mass
        body_inertia = self.env.robot_body_inertia
        foot_pos = self.env.get_foot_position_in_CoM_frame()

        inv_mass = np.eye(3) / body_mass
        inv_inertia = np.linalg.inv(body_inertia)

        # compute mass matrix
        mass_mat = np.zeros((6, 18))
        for leg_id in range(6):
            # first 3 rows are mass
            mass_mat[:3, leg_id * 3:leg_id * 3 + 3] = inv_mass
            # last 3 rows are inertia
            foot_pos_skew = skew_matrix(foot_pos[:, leg_id])
            mass_mat[3:6, leg_id * 3:leg_id * 3 + 3] = rot_z.T.dot(inv_inertia).dot(foot_pos_skew)
        return mass_mat
    
    def _compute_objective_matrix(self, desired_acc):
        Q = np.diag(np.array([1., 1., 1., 10., 10, 1.]))#TODO: tune this
        R = np.ones(18) * 1e-4
        M = self._compute_mass_matrix()
        q_ddot = desired_acc
        g = np.array([0., 0., self.env.gravity, 0., 0., 0.]) # -9.81 in pybullet
        # Compute G and a
        G = M.T @ Q @ M + R
        a = (g - q_ddot).T @ Q @ M #in quadprog, the objective function is 1/2 x^T G x - a^T x
        return G, a

    def _compute_constraint_matrix(self, contacts):
        mu = self.env.friction * 0.5 # lower the possibility of slipping TODO: tune this coefficient
        f_min = 1e-3* self.env.robot_body_mass * self.env.gravity #TODO: modify the ratio
        f_max = 1 * self.env.robot_body_mass * self.env.gravity #TODO: check the max GRF
        C = np.zeros((36, 18))
        b = np.zeros(36)
        # ground reaction force constraints in first 12 rows
        # -f_min <= f_z <= f_max if in contact
        # -1e-7 <= f_z <= 1e-7 if not in contact
        for leg_id in range(6):
            C[leg_id * 2, leg_id * 3 + 2] = 1
            C[leg_id * 2 + 1, leg_id * 3 + 2] = -1
            if contacts[leg_id]:
                b[leg_id * 2], b[leg_id * 2 + 1] = f_min, -f_max
            else:
                b[leg_id * 2] = -1e-7
                b[leg_id * 2 + 1] = -1e-7
        # friction constraints in last 24 rows
        # -mu*f_z <= f_x <= mu*f_z
        # -mu*f_z <= f_y <= mu*f_z
        for leg_id in range(6):
            row_id = 12 + leg_id * 4
            col_id = leg_id * 3
            C[row_id + 0, col_id:col_id + 3] = np.array([1, 0, mu])
            C[row_id + 1, col_id:col_id + 3] = np.array([-1, 0, mu])
            C[row_id + 2, col_id:col_id + 3] = np.array([0, 1, mu])
            C[row_id + 3, col_id:col_id + 3] = np.array([0, -1, mu])
            # b already initialized to 0 so no need to reassign 0 to b
        C = C.T # transpose to match the format of qp solver
        return C, b
    
    def _compute_contact_force(self, desired_acc):
        # contacts = self.env.get_foot_contact()
        # contacts = np.array(
        # [leg_state == self.gait_scheduler.STANCE
        #  for leg_state in self.gait_scheduler.desired_leg_state],
        # dtype=np.int32)
        desired_contacts = self.gait_scheduler.desired_leg_state
        actual_contacts = self.env.get_foot_contact()
        contacts = np.logical_and(desired_contacts, actual_contacts)
        G, a = self._compute_objective_matrix(desired_acc)
        G += np.eye(18) * 1e-8 # TODO: check if this is necessary
        C, b = self._compute_constraint_matrix(contacts)
        # solve for contact forces
        contact_force = quadprog.solve_qp(G, a, C, b)
        contact_force = -contact_force[0].reshape((6, 3)) # Newton's third law, the reactive force
        #contact_force[:, 0:2] = -contact_force[:, 0:2]
        return contact_force # order in bullet frame
    
    def _map_force_to_torque(self, contact_force):
        motor_torque = np.zeros(18)
        Jv = self.env.get_jacobian()# np.zeros([6, 3, 6])
        for leg_id in range(6):
            J_leg = Jv[:, :, leg_id]
            F_leg = np.append(contact_force[leg_id, :], np.zeros(3,)).T # F_leg is in bullet order already
            motor_torque[leg_id * 3:leg_id * 3 + 3] = J_leg.T @ F_leg
        motor_torque = hebi2bullet(motor_torque)
        return motor_torque # order in bullet frame


class Gait_scheduler():
    def __init__(self, env):
        self.env = env
        self.contact_detection_phase_threshold = 0.1 # no detection at the beginning phase
        self.foot_clearance = 0.1
        self.duty_factor = 0.5
        self.stance_duration = 0.15
        self.swing_duration = self.stance_duration / self.duty_factor - self.stance_duration
        self.cycle_duration = self.stance_duration + self.swing_duration
        # leg status
        self.SWING = 0
        self.STANCE = 1
        self.EARLY_CONTACT = 2
        self.LOSE_CONTACT = 3
        self.init_leg_state = (self.SWING, self.STANCE, self.STANCE, self.SWING, self.SWING, self.STANCE)
        self.next_leg_state = []
        self.phase_change_threshold = []
        for state in self.init_leg_state:
            if state == self.SWING:
                self.next_leg_state.append(self.STANCE)
                self.phase_change_threshold.append(1 - self.duty_factor)
            else:
                self.next_leg_state.append(self.SWING)
                self.phase_change_threshold.append(self.duty_factor)
        self.reset()

    def reset(self):
        self.sub_phase = np.zeros(6)
        self.leg_state = list(self.init_leg_state)
        self.desired_leg_state = list(self.init_leg_state)

    def update(self):
        # update self.leg_state and self.desired_leg_state
        simulation_time = self.env.time_since_initalisation
        phase = np.fmod(simulation_time, self.cycle_duration) / self.cycle_duration # phase in [0, 1]
        contact = self.env.get_foot_contact()
        for leg_id in range(6):
            threshold = self.phase_change_threshold[leg_id]
            if phase < threshold:
                self.sub_phase[leg_id] = phase / threshold
                self.desired_leg_state[leg_id] = self.init_leg_state[leg_id]
            else:
                self.sub_phase[leg_id] = (phase - threshold) / (1 - threshold)
                self.desired_leg_state[leg_id] = self.next_leg_state[leg_id]
            self.leg_state[leg_id] = self.desired_leg_state[leg_id] # update current leg state
            # detect early contact and lose contact, no detection at the beginning of sub phase
            if self.sub_phase[leg_id] > self.contact_detection_phase_threshold:
                if self.leg_state[leg_id] == self.SWING and contact[leg_id]:
                    self.leg_state[leg_id] = self.EARLY_CONTACT
                if self.leg_state[leg_id] == self.STANCE and not contact[leg_id]:
                    self.leg_state[leg_id] = self.LOSE_CONTACT

    @property
    def joint_control_mode(self):
        joint_control_mode = np.zeros(18)
        for leg_id in range(6):
            if self.desired_leg_state[leg_id] in (self.SWING, self.LOSE_CONTACT):
                joint_control_mode[leg_id * 3:leg_id * 3 + 3] = self.env.p.POSITION_CONTROL
            else:
                joint_control_mode[leg_id * 3:leg_id * 3 + 3] = self.env.p.TORQUE_CONTROL
        joint_control_mode = hebi2bullet(joint_control_mode)

        #joint_control_mode = np.ones(18)
        return joint_control_mode