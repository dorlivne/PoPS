from configs import LunarLanderConfig as dense_config
from model import ActorLunarlander
import gym
import numpy as np
from Lunarlander.train_lunarlander import preprocess_state
def accumulate_experience_lunarlander(teacher: ActorLunarlander,
                                      exp_replay, config=dense_config):
    env = gym.make('LunarLander-v2')
    steps = 0
    while 1:
        state = env.reset()
        state = np.asarray(state)
        done = False
        while not done:
            steps += 1
            teacher_q_value, action_dist = teacher.get_before_softmax(state=preprocess_state(state))
            action = teacher.select_action(action_dist)
            next_state, reward, done, _ = env.step(action)
            exp_replay.add_memory(state, teacher_q_value, action)  # feeding the experience replay
            state = next_state
        if steps > config.OBSERVE:  # we have OBSERVE  number of exp in exp_replay
            break
