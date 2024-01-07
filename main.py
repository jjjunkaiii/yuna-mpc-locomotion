from Yuna_Env import YunaEnv
from Yuna_MPC import YunaMPC

import numpy as np
import time
# command
comm_lin_vel = np.array([0.0, 0.0, 0.0])
comm_ang_vel = 0.0

def main():
    env = YunaEnv(camerafollow=False, complex_terrain=False)
    yuna = YunaMPC(env)
    time_start = time.time()
    # time.sleep(10) # for video recording
    while (time.time() - time_start < 1e3):
        yuna.get_command(comm_lin_vel, comm_ang_vel)
        yuna.update()
        action = yuna.get_action()
        env.step(action, sleep='auto') #sleep=0.1

if __name__ == '__main__':
    main()
