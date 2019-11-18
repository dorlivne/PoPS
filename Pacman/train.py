from configs import DensePacmanAgentConfig as dense_config
import numpy as np
USE_PER = 1


def train_on_batch(agent, target_dqn, exp_replay, e, config=dense_config):
    batch_size = config.batch_size
    batch_size = batch_size * config.steps_per_train
    # Multi step variation,
    # the batch size should match the delta of steps between training
    if not USE_PER:
        mini_batch = exp_replay.getMiniBatch(batch_size=batch_size)
        weights, indexes = np.ones(np.shape(mini_batch)[0], config.output_size), None
    else:
        n_epoch = config.n_epoch
        mini_batch, weights, indexes = exp_replay.getMiniBatch(batch_size=batch_size,
                                             beta=config.beta_schedule(beta0=config.BETA0_PER,
                                                                             e=e, n_epoch=n_epoch))
        weights = np.expand_dims(weights, axis=-1)
        weights = weights * np.ones(1, config.output_size)
    state_batch, action_batch, reward_batch, dones_batch, next_state_batch = [], [], [], [], []
    for exp in mini_batch:
        state_batch.append(exp.state)
        action_batch.append(exp.action)
        reward_batch.append(exp.reward)
        dones_batch.append(exp.done)
        if dones_batch[-1]:
            next_state_batch.append(exp.state)   # this is just to prevent nope, the terminal states are masked anyway
        else:
            next_state_batch.append(exp.next_state)
    target_batch = target_dqn.get_q(state_batch)  # target.get_q(state_batch) --> (batch_size,action_dim)
    next_qvalues_batch = target_dqn.get_q(state=next_state_batch)  # target.get_q(next_state) --> (batch_size,action_dim)
    for i, reward in enumerate(reward_batch):   # iteration over batch_size
            target = np.clip(reward, -1., 1.) + (agent.gamma * np.max(next_qvalues_batch[i])) * (1 - dones_batch[i]) # create target_value --> scalar
            target_batch[i][action_batch[i]] = target  # target_batch (target.get_q(state_batch))[i][action[i]) = target
            # calculating the TD_error for each experience
    _, td_errors = agent.learn(target_batch=target_batch, learning_rate=config.learning_rate_schedule(e, 1),
                            input=state_batch, weights=weights)
    if USE_PER:
        new_priority = np.abs(td_errors) + config.EPS_PER  # we add epsilon so that every transaction has a chance
        new_priority = [priority[action_batch[i]] for i, priority in enumerate(new_priority)]
        exp_replay.update_priorities(indexes=indexes, priorities=new_priority)


def train_on_batch_with_benchmark(agent, target_dqn, exp_replay, e, config=dense_config):
    batch_size = config.batch_size
    batch_size = batch_size * config.steps_per_train
    # Multi step variation,
    # the batch size should match the delta of steps between training
    mini_batch = exp_replay.getMiniBatch(batch_size=batch_size)
    state_batch, action_batch, reward_batch, dones_batch, next_state_batch = [], [], [], [], []
    for exp in mini_batch:
        state_batch.append(exp.state)
        action_batch.append(exp.action)
        reward_batch.append(exp.reward)
        dones_batch.append(exp.done)
        if dones_batch[-1]:
            next_state_batch.append(exp.state)   # this is just to prevent nope, the terminal states are masked anyway
        else:
            next_state_batch.append(exp.next_state)
    target_batch_original = target_dqn.get_q(state_batch)  # target.get_q(state_batch) --> (batch_size,action_dim)
    target_batch = target_batch_original
    next_qvalues_batch = target_dqn.get_q(state=next_state_batch)  # target.get_q(next_state) --> (batch_size,action_dim)
    for i, reward in enumerate(reward_batch):   # iteration over batch_size
            target = np.clip(reward, -1., 1.) + (agent.gamma * np.max(next_qvalues_batch[i])) * (1 - dones_batch[i]) # create target_value --> scalar
            target_batch[i][action_batch[i]] = target  # target_batch (target.get_q(state_batch))[i][action[i]) = target
            # calculating the TD_error for each experience
    agent.learn_with_benchmark(target_batch=target_batch, learning_rate=config.learning_rate_schedule(e),
                               input=state_batch, teacher_batch=target_batch_original)

