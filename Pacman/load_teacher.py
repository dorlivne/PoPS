from model import DQNPacman
from configs import DensePacmanAgentConfig as dense_config


def load_teacher():
    """
    load ready-to-go teacher from "https://towardsdatascience.com/advanced-dqns-playing-pac-man-with-deep-reinforcement-learning-3ffbd99e0814"
    :return: a trained teacher model trained with double dueling dqn with prioritized ER
    """
    dqn = DQNPacman(input_size=dense_config.input_size, output_size=dense_config.output_size,
                    model_path=dense_config.model_path, scope=dense_config.scope,
                    epsilon_stop=dense_config.final_epsilon, epsilon=dense_config.initial_epsilon)
    dqn.load_model()
    return dqn



if __name__ == '__main__':
    load_teacher()