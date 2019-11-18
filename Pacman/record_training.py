import gym
from model import DQNPacman, DQNAgent,StudentPacman, PacmanTargetNet
from configs import DensePacmanAgentConfig as dense_config
from configs import StudentPacmanConfig as student_config
from configs import PrunePacmanAgentConfig as prune_config
import numpy as np
from argparse import ArgumentParser
from PIL import Image
from copy import deepcopy
from Pacman.processor import process_observation, process_state_batch, init_state, append_frame
from Pacman.accumulate_experience_Pacman import accumulate_experience
from utils.Memory import Supervised_ExperienceReplay
from Pacman.evaluate import evaluate
from utils.logger_utils import get_logger
from train import train_student


def main():
    student = StudentPacman(input_size=student_config.input_size,
                                       output_size=student_config.output_size,
                                       model_path=r"/media/dorliv/50D03BD3D03BBE52/Study/Masters/PoPS/Pacman/PoPS_Iterative/policy_step_1",
                                       tau=student_config.tau,
                                       redundancy=[0.989990234375, 0.989990234375, 0.989990234375, 0.9900001992984694, 0.9900390625],
                                       pruning_freq=prune_config.pruning_freq,
                                       sparsity_end=prune_config.sparsity_end,
                                       target_sparsity=prune_config.target_sparsity,
                                       prune_till_death=True)
    student.load_model()
    teacher = PacmanTargetNet(input_size=dense_config.input_size, output_size=dense_config.output_size)
    teacher.load_model(path=dense_config.ready_path)  # load teacher
    ER = Supervised_ExperienceReplay(student_config.memory_size)
    learning_rate_dynamic_multiplier = 1.0
    n_epochs = 10000
    plus = True
    last_score = 0
    score = 150
    score_history = np.zeros(10)
    logger = get_logger("record_training/record")
    logger.info("recording training session of Pacman model with {} parameters using Policy Distillation"
                .format(student.print_num_of_params()))
    for train_num in range(n_epochs):
        if score > (last_score + 100):
            _ = evaluate(agent=student, n_epoch=1, record=True, video_path="/media/dorliv/50D03BD3D03BBE52/Study/Masters/PoPS/Pacman/videos/epoch_last")  # evaluating the agent
            last_score = score
        accumulate_experience(teacher=teacher, exp_replay=ER, config=student_config)
        logger.info("Session number {} : finished accumulating experience, training the student...".format(train_num))
        print("EPOCH NUM {}/{}: training the student".format(train_num, n_epochs))
        train_student(logger=logger, student=student, exp_replay=ER,
                      lr=student_config.learning_rate_schedule(train_num, arch_type=1) * learning_rate_dynamic_multiplier,
                      epoch=train_num)
        print("EPOCH NUM {}/{}: evaluating the student".format(train_num, n_epochs))
        logger.info("student evaluation number {}".format(train_num))
        score = evaluate(agent=student, n_epoch=10, render=False)  # evaluating the agent
        logger.info("student finished training {},  with mean score of {}".format(train_num, score))
        if train_num > 50 and score < dense_config.LOWER_BOUND:
            if not plus:
                learning_rate_dynamic_multiplier = 0.1
                plus = True
            else:
                learning_rate_dynamic_multiplier = 10.0
                plus = False
        score_history[train_num % 10] = score
        if np.mean(score_history) > dense_config.OBJECTIVE_SCORE:
            print("finished recording")
            logger.info("finished recording")
            break

if __name__ == '__main__':
    main()