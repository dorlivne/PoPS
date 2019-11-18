import numpy as np
from PIL import Image
from copy import deepcopy
INPUT_SHAPE = (84, 84)


def init_state():
    # return np.zeros((84, 84, 4))
    return np.zeros((4, 84, 84))

def append_frame(state, frame):
    # new_state = deepcopy(state)
    # new_state[:, :, :-1] = state[:, :, 1:]
    # new_state[:, :, -1] = frame
    new_state = deepcopy(state)
    new_state[:-1, :, :, ] = state[1:, :, :]
    new_state[-1, :, :] = frame
    del state
    return new_state


def process_observation(observation):
    assert observation.ndim == 3
    img = Image.fromarray(observation)
    img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
    processed_observation = np.array(img)
    assert processed_observation.shape == INPUT_SHAPE
    return processed_observation.astype('float32') / 255.  # saves storage in experience memory


def process_state_batch(batch):
        return np.asarray(batch).astype('float32') / 255.


def clip_rewards(reward):
    return np.clip(reward, -1, 1)
