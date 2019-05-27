from utils.Memory import Supervised_ExperienceReplay, Supervised_Prioritzed_ExperienceReplay
from PONG.accumulate_experience_Pong import accumulate_experience
from configs import StudentPongConfig as student_config
from model import DQNAgent
import numpy as np
from PONG.evaluate import evaluate


def policy_distilliation_batch_train(exp_replay, student: DQNAgent, learning_rate=1.0e-4,
                                     config=student_config, use_per=False, e=None):
    """
    train the student on a batch of experiences
    :param student: student to be trained on a batch of experiences
    :param exp_replay: the Experience replay
    :param learning_rate: learning rate for the SGD
    :param config: config where batch size and the output size is described
    :param use_per: if True use Prioritized supervised experience replay
    :return: loss fn value
    """
    if not use_per:
        mini_batch = exp_replay.getMiniBatch(batch_size=config.batch_size)
        weights, indexes = np.ones(np.shape(mini_batch)[0], config.output_size), None
    else:
        mini_batch, weights, indexes = exp_replay.getMiniBatch(batch_size=config.batch_size,
                                                               beta=config.beta_schedule(beta0=config.BETA0_PER,
                                                                                         e=e, n_epoch=config.n_epochs))
    state = [exp.state for exp in mini_batch]
    target = [exp.label for exp in mini_batch]
    target = np.squeeze(target)
    loss, td_errors = student.learn(target_batch=target, input=state, learning_rate=learning_rate,
                                    weights=weights)

    if use_per:
        td_errors_maximum = np.max(td_errors)
        td_errors *= np.ones_like(td_errors) * (td_errors_maximum ** -1)
        action_batch = [exp.action for exp in mini_batch]
        new_priority = np.abs(td_errors) + config.EPS_PER  # we add epsilon so that every transaction has a chance
        new_priority = [priority[action_batch[i]] for i, priority in enumerate(new_priority)]
        exp_replay.update_priorities(indexes=indexes, priorities=new_priority)

    return loss



def train_student(logger, student, exp_replay: Supervised_ExperienceReplay,
                  prune=False, best_path=student_config.prune_best, lr=1.0e-4, stop_prune_arg=True,
                  use_per=False, epoch=None, num_of_iteration=20000,
                  objective_score=18.0, lower_bound=0.0, config=student_config, evaluate_fn=evaluate):
    """
    trains the student based on the experience in the Experience replay
    :param logger: logger to log progress information
    :param student: student net
    :param epoch:the agent epoch of training useful only when using PER
    :param exp_replay: exp_replay
    :param prune: if prune is false then we don't prune else we prune
    :param best_path: the location to save the best pruned agent which scores over 18.0
    :param lr: the learning rate to teach the student
    :param stop_prune_arg: argument which is relevant only when pruning, allows to stop pruning for a period of time
                           until the agent is back to winning ways
    :param use_per: if True use Prioritized supervised experience replay
    :param num_of_iteration  to train the model
    :param objective_score of the model
    :param lower_bound of the model performance
    :param config details hyper parameters and special properties
    :return: information regarding the current learning session and the stop prune arg for later iterations to use,
             relevant only when pruning the agent
    """
    NNZ_params_list = []
    score_list = []
    stop_prune = stop_prune_arg
    m = 0
    logger.info("learning rate is {}".format(lr))
    last_sparsity_measure = -1
    for iter in range(num_of_iteration + 1):  # training on experience replay for 100k mini_batchs
        if prune and not stop_prune:
            student.prune()
        _ = policy_distilliation_batch_train(student=student, exp_replay=exp_replay, learning_rate=lr, e=epoch,
                                             use_per=use_per, config=config)  # train student
        if iter % 1000 == 0:

            # ----- PRUNE sector -----
            if prune:
                score = evaluate_fn(agent=student, n_epoch=config.eval_prune, render=False)
                sparsity = student.get_model_sparsity()
                logger.info("Evaluation of student {}/{} : sparsity is {}, {} episode mean score is {}"
                            .format(iter, num_of_iteration, sparsity, config.eval_prune, score))
                print("Evaluation of student {}/{} : sparsity is {}, {} episode mean score is {}"
                      .format(iter, num_of_iteration, sparsity, config.eval_prune, score))
                if sparsity > last_sparsity_measure:
                    NNZ = student.get_number_of_nnz_params()
                    score_list.append(score)
                    NNZ_params_list.append(NNZ)
                    last_sparsity_measure = sparsity
                elif score > score_list[-1]:
                    score_list[-1] = score
                # pruning w.r.t to the accuracy
                if score > objective_score:
                    if stop_prune:
                        freeze_global_step = student.unfreeze_global_step()
                        logger.info("agent got back on the horse and managed to score 18 plus,"
                                    " continue pruning with global step {}".format(freeze_global_step))
                        stop_prune = False

                    logger.info("saving model with sparsity {}, in path {}".format(sparsity, best_path))
                    student.save_model(path=best_path)
                if score < lower_bound and not stop_prune:
                    stop_prune = True
                    freeze_global_step = student.freeze_global_step()
                    logger.info("stopped pruning due to low results, global step is {}".format(freeze_global_step))

                if score < lower_bound:  # added to avoid wasting time
                    m += 1
                    if m == 5:
                        break
                else:
                    m = 0

    return score_list, NNZ_params_list, stop_prune


def fit_supervised(logger, teacher, student, n_epochs, arch_type=0, config=student_config,
                   use_per=False, objective_score=18.0, lower_score_bound=10.0, accumulate_experience_fn=accumulate_experience,
                   evaluate_fn=evaluate):
    """
    the framework for training an agent with Policy distillation
    :param teacher: the teacher net, should be a bigger net then the student net
    :param student: the student net, same as the teacher with smaller architecture
    :param n_epochs: number of epoch to train the student via policy distilliation with the teacher
    :param config: holds information about certain customer variables
    :param use_per: use Prioritized Experience Replay flag
    :param objective_score: the objective score for the environment
    :param lower_score_bound: the lower_bound, if crossed after epoch 50, the learning rate is altered
    :param accumulate_experience_fn: a function in which the teacher interacts with the environment and feeds the ER
    :param evaluate_fn: a function to evaluate the agent
    """
    logger.info("Start :  training agent ")
    if not use_per:
         exp_replay = Supervised_ExperienceReplay(size=config.memory_size)
    else:
        exp_replay = Supervised_Prioritzed_ExperienceReplay(size=config.memory_size,
                                                            alpha=config.ALPHA_PER)
    i = 0
    score = 0
    plus = True
    learning_rate_dynamic_multiplier = 1.0
    for train_num in range(n_epochs):
        accumulate_experience_fn(teacher=teacher, exp_replay=exp_replay, config=config)
        logger.info("Session number {} : finished accumulating experience, training the student...".format(train_num))
        print("EPOCH NUM {}/{}: training the student".format(train_num, n_epochs))
        train_student(logger=logger, student=student, exp_replay=exp_replay,
                      lr=config.learning_rate_schedule(train_num, arch_type=arch_type) * learning_rate_dynamic_multiplier,
                      epoch=train_num, use_per=use_per)
        print("EPOCH NUM {}/{}: evaluating the student".format(train_num, n_epochs))
        logger.info("student evaluation number {}".format(train_num))
        score = evaluate_fn(agent=student, n_epoch=10, render=False)  # evaluating the agent
        logger.info("student finished training {},  with mean score of {}".format(train_num, score))
        if score > objective_score:
            i += 1
            if i % 5 == 0:
                student.save_model()
                logger.info("student finished training, objective achieved with mean score of {}".format(score))
                return score
            if i > 2:
                student.save_model()
        else:
            i = 0
        if train_num > 50 and score < lower_score_bound:
            if not plus:
                learning_rate_dynamic_multiplier = 0.1
                plus = True
            else:
                learning_rate_dynamic_multiplier = 10.0
                plus = False
    student.save_model()
    return score

