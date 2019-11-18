import gym
import numpy as np
from utils.Memory import Supervised_ExperienceReplay
from configs import StudentPongConfig as student_config
from Pacman.processor import process_observation, process_state_batch, init_state, append_frame
from copy import deepcopy


def accumulate_experience(teacher, exp_replay: Supervised_ExperienceReplay, config=student_config):
    """
    teacher feeds the Experience replay with new experiences
    :param teacher: teacher net, knows how to solve the problem
    :param exp_replay: the experience replay where the teacher saves its experiences
    :param config : holds customer variables such as OBSERVE
    :return: an experience replay filled with new experiences
    """
    env = gym.make('MsPacmanDeterministic-v4')
    steps = 0
    while 1:
        state = init_state()
        observation = env.reset()
        observation = process_observation(observation)
        state = np.asarray(state)
        done = False
        while not done:
            steps += 1
            state = append_frame(state, observation)
            teacher_q_value = teacher.get_q(state=np.expand_dims(state, axis=0))
            action = teacher.select_action(teacher_q_value)
            next_observation, reward, done, _ = env.step(action)
            next_observation = process_observation(next_observation)
            observation = next_observation
            exp_replay.add_memory(state, teacher_q_value, action)  # feeding the experience replay
        if steps > config.OBSERVE:  # we have OBSERVE  number of exp in exp_replay
            try:
                del env
            except ImportError:
                pass
            break
