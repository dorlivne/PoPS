from utils.wrappers import wrap_deepmind
import gym
import numpy as np
from utils.Memory import Supervised_ExperienceReplay
from configs import StudentPongConfig as student_config


def accumulate_experience(teacher, exp_replay: Supervised_ExperienceReplay, config=student_config):
    """
    teacher feeds the Experience replay with new experiences
    :param teacher: teacher net, knows how to solve the problem
    :param exp_replay: the experience replay where the teacher saves its experiences
    :param config : holds customer variables such as OBSERVE
    :return: an experience replay filled with new experiences
    """

    env = gym.make("PongNoFrameskip-v4")
    env = wrap_deepmind(env, frame_stack=True)
    steps = 0
    while 1:
        state = env.reset()
        state = np.asarray(state)
        done = False
        while not done:
            steps += 1
            teacher_q_value = teacher.get_q(state=np.reshape(state, (1, state.shape[0], state.shape[1], state.shape[2])))
            action = teacher.select_action(teacher_q_value)
            next_state, reward, done, _ = env.step(action + 1)
            next_state = np.asarray(next_state)
            exp_replay.add_memory(state, teacher_q_value, action)  # feeding the experience replay
            state = next_state
        if steps > config.OBSERVE:  # we have OBSERVE  number of exp in exp_replay
            try:
                del env
            except ImportError:
                pass
            break
