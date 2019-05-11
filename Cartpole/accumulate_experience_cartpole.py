import gym
from model import CartPoleDQNTarget
from utils.Memory import Supervised_ExperienceReplay
from configs import CartpoleConfig as dense_config
import numpy as np

def accumulate_experience_cartpole(teacher: CartPoleDQNTarget,
                                   exp_replay: Supervised_ExperienceReplay, config=dense_config):
    env = gym.make('CartPole-v0')
    steps = 0
    while 1:
        state = env.reset()
        state = np.asarray(state)
        done = False
        while not done:
            steps += 1
            teacher_q_value = teacher.get_q(state=np.expand_dims(state, axis=0))
            action = teacher.select_action(teacher_q_value)
            next_state, reward, done, _ = env.step(action)
            exp_replay.add_memory(state, teacher_q_value, action)  # feeding the experience replay
            state = next_state
        if steps > config.OBSERVE:  # we have OBSERVE  number of exp in exp_replay
            break

