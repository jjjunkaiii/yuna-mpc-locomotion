from Yuna_Env import YunaEnv
from Yuna_MPC import YunaMPC

import numpy as np
import time
# command
comm_lin_vel = np.array([0.2, 0.0, 0.0])
comm_ang_vel = 0.0

def main():
    env = YunaEnv(camerafollow=False)
    yuna = YunaMPC(env)
    time_start = time.time()
    time.sleep(1)
    while (time.time() - time_start < 1e3):
        yuna.get_command(comm_lin_vel, comm_ang_vel)
        yuna.update()
        action = np.zeros((3,18))
        action[0] = np.ones(1)
        action[1,1]=50
        env.step(action)

if __name__ == '__main__':
    main()


time.sleep(1e3)
