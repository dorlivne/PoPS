from utils.tensorflow_utils import _build_conv_layer, _build_fc_layer
import tensorflow as tf
import numpy as np
from tensorflow.contrib.model_pruning.python import pruning
from abc import ABCMeta, abstractmethod

import random
import os


class BaseNetwork:

    def __init__(self, model_path):
        self.graph = tf.Graph()
        self.model_path = model_path

    @property
    def sess(self):
        if not hasattr(self, '_sess'):
            config = tf.ConfigProto()
            # to save GPU resources
            config.gpu_options.allow_growth = True
            self._sess = tf.Session(config=config, graph=self.graph)
        return self._sess

    def initialize(self):
        with self.graph.as_default():
            self.init_variables(tf.global_variables())

    def init_variables(self, var_list):
        self.sess.run(tf.variables_initializer(var_list))

    def number_of_parameters(self, var_list):
        return sum(np.prod(v.get_shape().as_list()) for v in var_list)

    def print_num_of_params(self):
        with self.graph.as_default():
            print('Number of parameters (four bytes == 1 parameter): {}.\n'.format(
                int(self.number_of_parameters(tf.trainable_variables()))))
            return int(self.number_of_parameters(tf.trainable_variables()))

    # need to create a saver after the graph has been established to use these functions
    def save_model(self, path=None, sess=None, global_step=None):
        save_dir = path or self.model_path
        os.makedirs(save_dir, exist_ok=True)
        self.saver.save(sess or self.sess,
                        os.path.join(save_dir, 'model.ckpt'),
                        global_step=global_step)
        return self

    def load_model(self, path=None, sess=None, verbose=True):
        path = path or self.model_path
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt is None:
            raise FileNotFoundError('Can`t load a model. Checkpoint does not exist.')
        restore_path = ckpt.model_checkpoint_path
        self.saver.restore(sess or self.sess, restore_path)

        return self

    def _variable_on_cpu(self, name, shape, initializer):
        """Helper to create a Variable stored on CPU memory.

        Args:
          name: name of the variable
          shape: list of ints
          initializer: initializer for Variable

        Returns:
          Variable Tensor
        """
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer)
        return var

    def _variable_with_weight_decay(self, name, shape, wd, initialization):
        """Helper to create an initialized Variable with weight decay.

        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.

        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.

        Returns:
          Variable Tensor
        """
        var = self._variable_on_cpu(
            name,
            shape,
            initialization)
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

class DQNAgent(BaseNetwork):  # abstract class
    __metaclass__ = ABCMeta
    """
    an interface for DQN agents 
    this interface comes with a graph and a session for each net,
    every net that implements this interface will have to fill the absract methods and you will receive a fully 
    operational DQN agent
    """
    def __init__(self, input_size,
                 output_size,
                 model_path,
                 scope='DenseAgent',
                 gamma=0.99,
                 epsilon=1,
                 epsilon_stop=0.05,
                 epsilon_decay=0.995):
        super(DQNAgent, self).__init__(model_path=model_path)
        self.input_size = input_size
        self.output_size = output_size
        self.epsilon = epsilon
        self.epsilon_stop = epsilon_stop
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.scope = scope
        self.freeze_global_step_var = 0
        with self.graph.as_default():
            self.global_step = tf.Variable(0, trainable=False)
            self._build_placeholders()
            self.logits = self._build_logits()
            self.weights_matrices = pruning.get_masked_weights()
            self.loss = self._build_loss()
            self.train_op = self._build_train_op()
            self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)
            self.init_variables(tf.global_variables())

    def _build_placeholders(self):
        self.input = tf.placeholder(dtype=tf.float32, shape=self.input_size, name='input')
        self.target = tf.placeholder(dtype=tf.float32, shape=self.output_size, name='target')
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=None, name='learning_rate')
        self.error_weights = tf.placeholder(dtype=tf.float32, shape=None, name='td_errors_weight')

    def get_flat_weights(self):
        with self.graph.as_default():
            weights_matrices = self.sess.run(self.weights_matrices)
            flatten_matrices = []
            for matrix in weights_matrices:
                flatten_matrices.append(np.ndarray.flatten(matrix))
            return flatten_matrices

    def get_weights(self):
        with self.graph.as_default():
            tensor_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            return self.sess.run(tensor_weights)

    def copy_weights(self, weights):
        with self.graph.as_default():
            tensor_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            for i, weight in enumerate(weights):
                self.sess.run(tf.assign(tensor_weights[i], weight))

    def get_number_of_nnz_params(self):
        flatten_matrices = self.get_flat_weights()
        weights = []
        for w in flatten_matrices:
            weights.extend(list(w.ravel()))
        weights_array = [w for w in weights if w != 0]
        return len(weights_array)

    def get_number_of_nnz_params_per_layer(self):
        flatten_matrices = self.get_flat_weights()
        nnz_at_each_layer = []
        for matrix in flatten_matrices:
            nnz_at_each_layer.append(len([w for w in matrix.ravel() if w != 0]))
        return nnz_at_each_layer

    def get_number_of_params(self):
        flatten_matrices = self.get_flat_weights()
        weights = []
        for w in flatten_matrices:
            weights.extend(list(w.ravel()))
        weights_array = [w for w in weights]
        return len(weights_array)

    @abstractmethod
    def _build_logits(self):
        pass

    @abstractmethod
    def _build_loss(self):
        self.td_errors = None
        pass

    @abstractmethod
    def _build_train_op(self):
        pass

    # global step job is to determine how much weights are going to get pruned in the next pruning phase, it is not
    # relevant when not pruning
    def freeze_global_step(self):
        self.freeze_global_step_var = self.print_global_step()
        return self.freeze_global_step_var

    def reset_global_step(self):
        self.init_variables(var_list=[self.global_step])

    def set_global_step(self, value=None):
        global_step = self.freeze_global_step_var if value is None else value
        self.sess.run(self.global_step.assign(global_step))

    def unfreeze_global_step(self):
        self.set_global_step(value=self.freeze_global_step_var)
        return self.print_global_step()

    def print_global_step(self):
        return self.sess.run(self.global_step)

    def get_q(self, state):
        return self.sess.run(self.logits, feed_dict={self.input: state})

    def lower_epsilon(self):
        self.epsilon = max(self.epsilon_stop, self.epsilon * self.epsilon_decay)

    def select_action(self, qValues, explore=True):
        rand = random.random()
        if rand < self.epsilon and explore:
            action = np.random.randint(0, self.output_size[-1])  # selecting from action_space
        else:
            action = np.argmax(qValues)
        return action

    def test_mode(self, model_path=None):
        model_path = self.model_path if not model_path else model_path
        self.epsilon = 0
        self.load_model(path=model_path)

    def learn(self, target_batch, learning_rate, input, weights):
        _, loss, td_errors = self.sess.run([self.train_op, self.loss, self.td_errors],
                                     feed_dict={self.input: input,
                                                self.target: target_batch,
                                                self.learning_rate: learning_rate,
                                                self.error_weights: weights})
        return loss, td_errors


class DQNPong(DQNAgent):
    """
    an agent which implements the interface, can successfully learn PONG with high results
    with a  pruning option to observe the redundancy in the net
    """
    def __init__(self, input_size,
                 output_size,
                 model_path,
                 scope='DenseAgent',
                 gamma=0.99,
                 pruning_start=0,
                 pruning_end=-1,
                 pruning_freq=int(10),
                 sparsity_start=0,
                 sparsity_end=int(10e5),
                 target_sparsity=0.9,
                 initial_sparsity=0,
                 epsilon=1.0,
                 epsilon_stop=0.05,
                 prune_till_death=False
                 ):
        super(DQNPong, self).__init__(input_size=input_size, output_size=output_size,
                                      model_path=model_path, scope=scope,
                                      gamma=gamma, epsilon_stop=epsilon_stop, epsilon=epsilon)
        with self.graph.as_default():
            self.sparsity = pruning.get_weight_sparsity()
            self.hparams = pruning.get_pruning_hparams() \
                .parse('name={}, begin_pruning_step={}, end_pruning_step={}, target_sparsity={},'
                       ' sparsity_function_begin_step={},sparsity_function_end_step={},'
                       'pruning_frequency={},initial_sparsity={},'
                       ' sparsity_function_exponent={}'.format(scope,
                                                               pruning_start,
                                                               pruning_end,
                                                               target_sparsity,
                                                               sparsity_start,
                                                               sparsity_end,
                                                               pruning_freq,
                                                               initial_sparsity,
                                                               3))
            # note that the global step plays an important part in the pruning mechanism,
            # the higher the global step the closer the sparsity is to sparsity end
            self.pruning_obj = pruning.Pruning(self.hparams, global_step=self.global_step)
            if not prune_till_death:
                weight_sparsity_map = ["conv_1/weights:0.1", 'conv_2/weights:0.1', 'conv_3/weights:0.6',
                                       "fc_4/weights:0.999", "logits/weights:0.1"]
                self.pruning_obj._spec.weight_sparsity_map = weight_sparsity_map
                self.pruning_obj._weight_sparsity_map = self.pruning_obj._get_weight_sparsity_map()
            self.mask_update_op = self.pruning_obj.conditional_mask_update_op()
            # the pruning objects defines the pruning mechanism, via the mask_update_op the model gets pruned
            # the pruning takes place at each training epoch and it objective to achieve the sparsity end HP
            self.init_variables(tf.global_variables())  # initialize variables in graph

    def get_model_sparsity(self):
        sparsity = self.sess.run(self.sparsity)
        return np.mean(sparsity)

    def prune(self):
        self.sess.run([self.mask_update_op])

    def _build_logits(self):  # original size net
            self.input = tf.math.divide(self.input, 256)
            with tf.variable_scope('conv_1') as scope:
                conv_1 =_build_conv_layer(self, inputs=self.input, filter_height=8, filter_width=8,
                                          channel_in=self.input_size[-1], channel_out=32,
                                          strides=[1, 4, 4, 1], scope=scope,
                                          weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
                conv_1 = tf.nn.relu(conv_1, name=scope.name)

            with tf.variable_scope('conv_2') as scope:
                conv_2 = _build_conv_layer(self, inputs=conv_1, filter_height=4, filter_width=4,
                                          channel_in=32, channel_out=64,
                                          strides=[1, 2, 2, 1], scope=scope,
                                           weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
                conv_2 = tf.nn.relu(conv_2, name=scope.name)

            with tf.variable_scope('conv_3') as scope:
                conv_3 = _build_conv_layer(self, inputs=conv_2, filter_height=3, filter_width=3,
                                          channel_in=64, channel_out=64,
                                          strides=[1, 1, 1, 1], scope=scope,
                                           weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
                conv_3 = tf.nn.relu(conv_3, name=scope.name)
            flatten_conv = tf.layers.flatten(conv_3)
            with tf.variable_scope('fc_4') as scope:
                fc_4 = _build_fc_layer(self, inputs=flatten_conv, scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(),
                                       shape=(3136, 512), activation=tf.nn.relu)

            with tf.variable_scope('logits') as scope:
                logits = _build_fc_layer(self, inputs=fc_4, scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(),
                                       shape=(512, self.output_size[-1]))
            self.output_node_name = logits.name
            return logits


    def _build_loss(self):
        self.td_errors = abs(self.target - self.logits)
        mse = tf.losses.mean_squared_error(labels=self.target, predictions=self.logits, weights=self.error_weights) # if not per then this error weights are 1
        mse = tf.reduce_mean(mse)
        # loss function is MSE( Q(s,a) - (reward + max(Q(next_state))),
        # note that this is a vectorized MSE, with batch_size dim
        return mse

    def _build_train_op(self):
            return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                     global_step=self.global_step)

    def lower_epsilon(self):
        self.epsilon = max(self.epsilon_stop, self.epsilon - self.degradation)

    def set_degradation(self, degradation):
        self.degradation = degradation


class PongTargetNet(DQNPong):
    """
    a target net for the above DQN agent, allows fast and stable training session
    """
    def __init__(self, input_size,
                 output_size, epsilon=0.05):
        self.input_size = input_size
        self.output_size = output_size
        self.graph = tf.Graph()
        self.epsilon = epsilon
        with self.graph.as_default():
            self._build_placeholders()
            self.logits = self._build_logits()
            self.saver = tf.train.Saver(var_list=tf.global_variables())
            self.init_variables(tf.global_variables())

    def sync(self, agent_path):
        """
        loads a new set of weights
        :param agent_path: the path of the agent we load the weights from
        """
        self.load_model(agent_path)


class StudentPong(DQNPong):
    """
    an agent which is similar to DQNPong, the only difference is that this agent is implemented in a way that fits
    the policy distillation algorithm, and size can be dynamically shift according to the redundancy measure
    """
    def __init__(self, input_size,
                 output_size,
                 model_path,
                 scope='Student',
                 gamma=0.99,
                 pruning_start=0,
                 pruning_end=-1,
                 pruning_freq=int(10),
                 sparsity_start=0,
                 sparsity_end=int(10e5),
                 target_sparsity=0.99,
                 initial_sparsity=0,
                 tau=0.01,
                 epsilon=0.0,
                 prune_till_death=False,
                 redundancy=None,
                 last_measure=10e6):
        self.redundancy = redundancy
        self.tau = tau
        self.last_measure = last_measure
        super(StudentPong, self).__init__(input_size,
                                          output_size,
                                          model_path,
                                          scope,
                                          gamma,
                                          pruning_start,
                                          pruning_end,
                                          pruning_freq,
                                          sparsity_start,
                                          sparsity_end,
                                          target_sparsity,
                                          initial_sparsity,
                                          epsilon=epsilon,
                                          prune_till_death=prune_till_death)

    def _calculate_sizes_according_to_redundancy(self):
        """
        the redundancy states what is the percentage of redundancy there is, at each layer
        :return: new sizes for each layer parameter!
        """
        assert self.redundancy is not None
        initial_sizes = [32, 64, 64, 512]
        new_sizes = np.zeros(4)
        total_number_of_parameters_initial = [self.filter_1 * self.filter_1 * self.input_size[-1] * initial_sizes[0],
                                              self.filter_2 * self.filter_2 * initial_sizes[0] * initial_sizes[1],
                                              self.filter_3 * self.filter_3 * initial_sizes[1] * initial_sizes[2],
                                              49 * initial_sizes[2] * initial_sizes[3],
                                              self.output_size[-1] * initial_sizes[3]]

        total_number_of_parameters_next_iteration = []
        for i, initial_num_of_params_at_layer in enumerate(total_number_of_parameters_initial):
            total_number_of_parameters_next_iteration.append(
                int(initial_num_of_params_at_layer * (1 - self.redundancy[i])))  # 1 - whats not important = important

        # total_number_of_parameters_next_iteration now holds the amount of parameters we want at each layer
        # we shall now use it to find the value of the layer parameters
        # conv_1_channel_out'
        new_sizes[3] = (total_number_of_parameters_next_iteration[4] / (self.output_size[-1]))  # fc_out
        new_sizes[0] = total_number_of_parameters_next_iteration[0] / (self.filter_1 * self.filter_1 * self.input_size[-1])  # ch_1_out
        if new_sizes[0] < 2:  # minimum of first channel
            new_sizes[0] = 2
            self.filter_1 = int(np.sqrt(total_number_of_parameters_next_iteration[0] / (self.input_size[-1] * new_sizes[0])))  # change filters to work it out
            new_sizes[1] = total_number_of_parameters_next_iteration[1] / (self.filter_2 * self.filter_2 * new_sizes[0])
            if new_sizes[1] < 4:   # minimum of first channel
                new_sizes[1] = 4
            if new_sizes[3] < 16:  # minimum of last channel
                new_sizes[3] = 16
            new_sizes[2] = (total_number_of_parameters_next_iteration[2] / (self.filter_3 * self.filter_3 * new_sizes[1]))  # ch_3_out
            new_sizes[1] = (total_number_of_parameters_next_iteration[1] + total_number_of_parameters_next_iteration[2]) / (self.filter_2 * self.filter_2 * new_sizes[0] + self.filter_3 * self.filter_3 * new_sizes[2])  # ch_2_out
            tmp = (new_sizes[2] * 49 * new_sizes[3] / total_number_of_parameters_initial[3])
            while tmp > (1 - self.redundancy[3]):
                new_sizes[2] -= 1

        for i, size in enumerate(new_sizes):
            new_sizes[i] = int(size)
        return new_sizes

    def _build_logits(self):   # Dynamic archi
            self.filter_1 = 8
            self.filter_2 = 4
            self.filter_3 = 3
            if self.redundancy is None:
                channel_out_conv1 = 32
                channel_out_conv2 = 64
                channel_out_conv3 = 64
                fc_4_out = 512
            else:
                size_according_to_redundancy = self._calculate_sizes_according_to_redundancy()
                channel_out_conv1 = size_according_to_redundancy[0]
                channel_out_conv2 = size_according_to_redundancy[1]
                channel_out_conv3 = size_according_to_redundancy[2]
                fc_4_out = size_according_to_redundancy[3]
            self.input = tf.math.divide(self.input, 256)
            with tf.variable_scope('conv_1') as scope:
                conv_1 =_build_conv_layer(self, inputs=self.input, filter_height=self.filter_1, filter_width=self.filter_1,
                                          channel_in=self.input_size[-1], channel_out=channel_out_conv1,
                                          strides=[1, 4, 4, 1], scope=scope,
                                          weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
                conv_1 = tf.nn.relu(conv_1, name=scope.name)
            with tf.variable_scope('conv_2') as scope:
                conv_2 = _build_conv_layer(self, inputs=conv_1, filter_height=self.filter_2, filter_width=self.filter_2,
                                          channel_in=channel_out_conv1, channel_out=channel_out_conv2,
                                          strides=[1, 2, 2, 1], scope=scope,
                                           weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
                conv_2 = tf.nn.relu(conv_2, name=scope.name)
            with tf.variable_scope('conv_3') as scope:
                conv_3 = _build_conv_layer(self, inputs=conv_2, filter_height=self.filter_3, filter_width=self.filter_3,
                                          channel_in=channel_out_conv2, channel_out=channel_out_conv3,
                                          strides=[1, 1, 1, 1], scope=scope,
                                           weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
                conv_3 = tf.nn.relu(conv_3, name=scope.name)
            flatten_conv = tf.layers.flatten(conv_3)
            with tf.variable_scope('fc_4') as scope:
                fc_4 = _build_fc_layer(self, inputs=flatten_conv, scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(),
                                       shape=(channel_out_conv3 * 49, fc_4_out), activation=tf.nn.relu)
                # the input shape is good as long as the filters dont change
            with tf.variable_scope('logits') as scope:
                logits = _build_fc_layer(self, inputs=fc_4, scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(),
                                       shape=(fc_4_out, self.output_size[-1]))

            self.output_node_name = logits.name
            return logits

    def _build_loss(self):
        """
        KLL  policy distillation loss function we want to minimize, according to Policy distillation
        """
        self.td_errors = abs(self.logits - self.target)
        # the td_error is crucial for Prioritized Experience repla
        eps = 0.00001
        teacher_sharpend_dist = tf.nn.softmax(self.target / self.tau, dim=1) + eps
        teacher_sharpend_dist = tf.squeeze(teacher_sharpend_dist)
        student_dist = tf.nn.softmax(self.logits, dim=1) + eps
        return tf.reduce_sum(teacher_sharpend_dist * tf.log(teacher_sharpend_dist / student_dist))

    def change_loss_function(self):
        """
        :return: a loss that is compatible with vanilla DQN , not suited for Policy distillation
        """
        with self.graph.as_default():
            self.loss = super(StudentPong, self)._build_loss()

    def change_train_op(self):
        """
        :return: a train_op that is compatible with vanilla DQN , not suited for Policy distillation
        """
        with self.graph.as_default():
            self.train_op = super(StudentPong, self)._build_train_op()

    def _build_train_op(self):
        """
        the suggested optimizer according to the Policy distillation paper
        """
        return tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                 global_step=self.global_step)

    """
    all the commented out arch are tested architecture trained with Policy distillation
    these architectures were hand picked until i found the bound, it was a tedious task to do manually
    """
    """
    def _build_logits(self):  # 0.8% of original_network - works
        self.input = tf.math.divide(self.input, 256)
        with tf.variable_scope('conv_1') as scope:
            conv_1 = _build_conv_layer(self, inputs=self.input, filter_height=6, filter_width=6,
                                       channel_in=self.input_size[-1], channel_out=8,
                                       strides=[1, 4, 4, 1], scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
            conv_1 = tf.nn.relu(conv_1, name=scope.name)

        with tf.variable_scope('conv_2') as scope:
            conv_2 = _build_conv_layer(self, inputs=conv_1, filter_height=4, filter_width=4,
                                       channel_in=8, channel_out=8,
                                       strides=[1, 2, 2, 1], scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
            conv_2 = tf.nn.relu(conv_2, name=scope.name)

        with tf.variable_scope('conv_3') as scope:
            conv_3 = _build_conv_layer(self, inputs=conv_2, filter_height=3, filter_width=3,
                                       channel_in=8, channel_out=7,
                                       strides=[1, 1, 1, 1], scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
            conv_3 = tf.nn.relu(conv_3, name=scope.name)
        flatten_conv = tf.layers.flatten(conv_3)
        with tf.variable_scope('fc_4') as scope:
            fc_4 = _build_fc_layer(self, inputs=flatten_conv, scope=scope,
                                   weight_init=tf.keras.initializers.glorot_uniform(),
                                   shape=(343, 32), activation=tf.nn.relu)

        with tf.variable_scope('logits') as scope:
            logits = _build_fc_layer(self, inputs=fc_4, scope=scope,
                                     weight_init=tf.keras.initializers.glorot_uniform(),
                                     shape=(32, self.output_size[-1]))
        self.output_node_name = logits.name
        return logits
    """
    """
    def _build_logits(self):  # 1% of original_network - no channel pruning works
        self.input = tf.math.divide(self.input, 256)
        with tf.variable_scope('conv_1') as scope:
            conv_1 = _build_conv_layer(self, inputs=self.input, filter_height=3, filter_width=3,
                                       channel_in=self.input_size[-1], channel_out=16,
                                       strides=[1, 4, 4, 1], scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
            conv_1 = tf.nn.relu(conv_1, name=scope.name)

        with tf.variable_scope('conv_2') as scope:
            conv_2 = _build_conv_layer(self, inputs=conv_1, filter_height=3, filter_width=3,
                                       channel_in=16, channel_out=16,
                                       strides=[1, 2, 2, 1], scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
            conv_2 = tf.nn.relu(conv_2, name=scope.name)

        with tf.variable_scope('conv_3') as scope:
            conv_3 = _build_conv_layer(self, inputs=conv_2, filter_height=2, filter_width=2,
                                       channel_in=16, channel_out=8,
                                       strides=[1, 1, 1, 1], scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
            conv_3 = tf.nn.relu(conv_3, name=scope.name)
        flatten_conv = tf.layers.flatten(conv_3)
        with tf.variable_scope('fc_4') as scope:
            fc_4 = _build_fc_layer(self, inputs=flatten_conv, scope=scope,
                                   weight_init=tf.keras.initializers.glorot_uniform(),
                                   shape=(648, 32), activation=tf.nn.relu)

        with tf.variable_scope('logits') as scope:
            logits = _build_fc_layer(self, inputs=fc_4, scope=scope,
                                     weight_init=tf.keras.initializers.glorot_uniform(),
                                     shape=(32, self.output_size[-1]))
        self.output_node_name = logits.name
        return logits
    """
    """
    def _build_logits(self):  # 0.6% of original_network
        self.input = tf.math.divide(self.input, 256)
        with tf.variable_scope('conv_1') as scope:
            conv_1 = _build_conv_layer(self, inputs=self.input, filter_height=4, filter_width=4,
                                       channel_in=self.input_size[-1], channel_out=4,
                                       strides=[1, 4, 4, 1], scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
            conv_1 = tf.nn.relu(conv_1, name=scope.name)

        with tf.variable_scope('conv_2') as scope:
            conv_2 = _build_conv_layer(self, inputs=conv_1, filter_height=4, filter_width=4,
                                       channel_in=4, channel_out=6,
                                       strides=[1, 2, 2, 1], scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
            conv_2 = tf.nn.relu(conv_2, name=scope.name)

        with tf.variable_scope('conv_3') as scope:
            conv_3 = _build_conv_layer(self, inputs=conv_2, filter_height=3, filter_width=3,
                                       channel_in=6, channel_out=6,
                                       strides=[1, 1, 1, 1], scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
            conv_3 = tf.nn.relu(conv_3, name=scope.name)
        flatten_conv = tf.layers.flatten(conv_3)
        with tf.variable_scope('fc_4') as scope:
            fc_4 = _build_fc_layer(self, inputs=flatten_conv, scope=scope,
                                   weight_init=tf.keras.initializers.glorot_uniform(),
                                   shape=(294, 32), activation=tf.nn.relu)

        with tf.variable_scope('logits') as scope:
            logits = _build_fc_layer(self, inputs=fc_4, scope=scope,
                                     weight_init=tf.keras.initializers.glorot_uniform(),
                                     shape=(32, self.output_size[-1]))
        self.output_node_name = logits.name
        return logits
    """
    """
    def _build_logits(self):  # 1% of original_network - works
        self.input = tf.math.divide(self.input, 256)
        with tf.variable_scope('conv_1') as scope:
            conv_1 = _build_conv_layer(self, inputs=self.input, filter_height=8, filter_width=8,
                                       channel_in=self.input_size[-1], channel_out=8,
                                       strides=[1, 4, 4, 1], scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
            conv_1 = tf.nn.relu(conv_1, name=scope.name)

        with tf.variable_scope('conv_2') as scope:
            conv_2 = _build_conv_layer(self, inputs=conv_1, filter_height=4, filter_width=4,
                                       channel_in=8, channel_out=8,
                                       strides=[1, 2, 2, 1], scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
            conv_2 = tf.nn.relu(conv_2, name=scope.name)

        with tf.variable_scope('conv_3') as scope:
            conv_3 = _build_conv_layer(self, inputs=conv_2, filter_height=3, filter_width=3,
                                       channel_in=8, channel_out=8,
                                       strides=[1, 1, 1, 1], scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
            conv_3 = tf.nn.relu(conv_3, name=scope.name)
        flatten_conv = tf.layers.flatten(conv_3)
        with tf.variable_scope('fc_4') as scope:
            fc_4 = _build_fc_layer(self, inputs=flatten_conv, scope=scope,
                                   weight_init=tf.keras.initializers.glorot_uniform(),
                                   shape=(392, 32), activation=tf.nn.relu)

        with tf.variable_scope('logits') as scope:
            logits = _build_fc_layer(self, inputs=fc_4, scope=scope,
                                     weight_init=tf.keras.initializers.glorot_uniform(),
                                     shape=(32, self.output_size[-1]))
        self.output_node_name = logits.name
        return logits
    """
    """
    def _build_logits(self):  # 1.8% of original_network - works
        self.input = tf.math.divide(self.input, 256)
        with tf.variable_scope('conv_1') as scope:
            conv_1 = _build_conv_layer(self, inputs=self.input, filter_height=8, filter_width=8,
                                       channel_in=self.input_size[-1], channel_out=8,
                                       strides=[1, 4, 4, 1], scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
            conv_1 = tf.nn.relu(conv_1, name=scope.name)

        with tf.variable_scope('conv_2') as scope:
            conv_2 = _build_conv_layer(self, inputs=conv_1, filter_height=4, filter_width=4,
                                       channel_in=8, channel_out=8,
                                       strides=[1, 2, 2, 1], scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
            conv_2 = tf.nn.relu(conv_2, name=scope.name)

        with tf.variable_scope('conv_3') as scope:
            conv_3 = _build_conv_layer(self, inputs=conv_2, filter_height=3, filter_width=3,
                                       channel_in=8, channel_out=8,
                                       strides=[1, 1, 1, 1], scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
            conv_3 = tf.nn.relu(conv_3, name=scope.name)
        flatten_conv = tf.layers.flatten(conv_3)
        with tf.variable_scope('fc_4') as scope:
            fc_4 = _build_fc_layer(self, inputs=flatten_conv, scope=scope,
                                   weight_init=tf.keras.initializers.glorot_uniform(),
                                   shape=(392, 64), activation=tf.nn.relu)

        with tf.variable_scope('logits') as scope:
            logits = _build_fc_layer(self, inputs=fc_4, scope=scope,
                                     weight_init=tf.keras.initializers.glorot_uniform(),
                                     shape=(64, self.output_size[-1]))
        self.output_node_name = logits.name
        return logits
     """
    """
    def _build_logits(self):  # 3% of original_network - works
        self.input = tf.math.divide(self.input, 256)
        with tf.variable_scope('conv_1') as scope:
            conv_1 = _build_conv_layer(self, inputs=self.input, filter_height=8, filter_width=8,
                                       channel_in=self.input_size[-1], channel_out=8,
                                       strides=[1, 4, 4, 1], scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
            conv_1 = tf.nn.relu(conv_1, name=scope.name)

        with tf.variable_scope('conv_2') as scope:
            conv_2 = _build_conv_layer(self, inputs=conv_1, filter_height=4, filter_width=4,
                                       channel_in=8, channel_out=8,
                                       strides=[1, 2, 2, 1], scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
            conv_2 = tf.nn.relu(conv_2, name=scope.name)

        with tf.variable_scope('conv_3') as scope:
            conv_3 = _build_conv_layer(self, inputs=conv_2, filter_height=3, filter_width=3,
                                       channel_in=8, channel_out=8,
                                       strides=[1, 1, 1, 1], scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
            conv_3 = tf.nn.relu(conv_3, name=scope.name)
        flatten_conv = tf.layers.flatten(conv_3)
        with tf.variable_scope('fc_4') as scope:
            fc_4 = _build_fc_layer(self, inputs=flatten_conv, scope=scope,
                                   weight_init=tf.keras.initializers.glorot_uniform(),
                                   shape=(392, 128), activation=tf.nn.relu)

        with tf.variable_scope('logits') as scope:
            logits = _build_fc_layer(self, inputs=fc_4, scope=scope,
                                     weight_init=tf.keras.initializers.glorot_uniform(),
                                     shape=(128, self.output_size[-1]))
        self.output_node_name = logits.name
        return logits
    """
    """
    def _build_logits(self):  # 0.4269% from the original net--- doesnt work
        self.input = tf.math.divide(self.input, 256)
        with tf.variable_scope('conv_1') as scope:
            conv_1 = _build_conv_layer(self, inputs=self.input, filter_height=8, filter_width=8,
                                       channel_in=self.input_size[-1], channel_out=2,
                                       strides=[1, 4, 4, 1], scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
            conv_1 = tf.nn.relu(conv_1, name=scope.name)

        with tf.variable_scope('conv_2') as scope:
            conv_2 = _build_conv_layer(self, inputs=conv_1, filter_height=4, filter_width=4,
                                       channel_in=2, channel_out=4,
                                       strides=[1, 2, 2, 1], scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
            conv_2 = tf.nn.relu(conv_2, name=scope.name)

        with tf.variable_scope('conv_3') as scope:
            conv_3 = _build_conv_layer(self, inputs=conv_2, filter_height=3, filter_width=3,
                                       channel_in=4, channel_out=4,
                                       strides=[1, 1, 1, 1], scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
            conv_3 = tf.nn.relu(conv_3, name=scope.name)
        flatten_conv = tf.layers.flatten(conv_3)
        with tf.variable_scope('fc_4') as scope:
            fc_4 = _build_fc_layer(self, inputs=flatten_conv, scope=scope,
                                   weight_init=tf.keras.initializers.glorot_uniform(),
                                   shape=(196, 32), activation=tf.nn.relu)

        with tf.variable_scope('logits') as scope:
            logits = _build_fc_layer(self, inputs=fc_4, scope=scope,
                                     weight_init=tf.keras.initializers.glorot_uniform(),
                                     shape=(32, self.output_size[-1]))
        self.output_node_name = logits.name
        return logits

    """
    """
    def _build_logits(self):  # 6% of original net
        self.input = tf.math.divide(self.input, 256)
        with tf.variable_scope('conv_1') as scope:
            conv_1 = _build_conv_layer(self, inputs=self.input, filter_height=8, filter_width=8,
                                       channel_in=self.input_size[-1], channel_out=8,
                                       strides=[1, 4, 4, 1], scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
            conv_1 = tf.nn.relu(conv_1, name=scope.name)

        with tf.variable_scope('conv_2') as scope:
            conv_2 = _build_conv_layer(self, inputs=conv_1, filter_height=4, filter_width=4,
                                       channel_in=8, channel_out=16,# 16
                                       strides=[1, 2, 2, 1], scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
            conv_2 = tf.nn.relu(conv_2, name=scope.name)

        with tf.variable_scope('conv_3') as scope:
            conv_3 = _build_conv_layer(self, inputs=conv_2, filter_height=3, filter_width=3,
                                       channel_in=16, channel_out=16, # 16
                                       strides=[1, 1, 1, 1], scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
            conv_3 = tf.nn.relu(conv_3, name=scope.name)
        flatten_conv = tf.layers.flatten(conv_3)
        with tf.variable_scope('fc_4') as scope:
            fc_4 = _build_fc_layer(self, inputs=flatten_conv, scope=scope,
                                   weight_init=tf.keras.initializers.glorot_uniform(),
                                   shape=(784, 128), activation=tf.nn.relu) # 392

        with tf.variable_scope('logits') as scope:
            logits = _build_fc_layer(self, inputs=fc_4, scope=scope,
                                     weight_init=tf.keras.initializers.glorot_uniform(),
                                     shape=(128, self.output_size[-1])) # 128
        self.output_node_name = logits.name
        return logits
    """
    """
    def _build_logits(self):  # half the size of teacher net
        self.input = tf.math.divide(self.input, 256)
        with tf.variable_scope('conv_1') as scope:
            conv_1 = _build_conv_layer(self, inputs=self.input, filter_height=8, filter_width=8,
                                       channel_in=self.input_size[-1], channel_out=16,
                                       strides=[1, 4, 4, 1], scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
            conv_1 = tf.nn.relu(conv_1, name=scope.name)

        with tf.variable_scope('conv_2') as scope:
            conv_2 = _build_conv_layer(self, inputs=conv_1, filter_height=4, filter_width=4,
                                       channel_in=16, channel_out=32,# 16
                                       strides=[1, 2, 2, 1], scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
            conv_2 = tf.nn.relu(conv_2, name=scope.name)

        with tf.variable_scope('conv_3') as scope:
            conv_3 = _build_conv_layer(self, inputs=conv_2, filter_height=3, filter_width=3,
                                       channel_in=32, channel_out=32, # 16
                                       strides=[1, 1, 1, 1], scope=scope,
                                       weight_init=tf.keras.initializers.glorot_uniform(), padding='VALID')
            conv_3 = tf.nn.relu(conv_3, name=scope.name)
        flatten_conv = tf.layers.flatten(conv_3)
        with tf.variable_scope('fc_4') as scope:
            fc_4 = _build_fc_layer(self, inputs=flatten_conv, scope=scope,
                                   weight_init=tf.keras.initializers.glorot_uniform(),
                                   shape=(1568, 256), activation=tf.nn.relu) # 392

        with tf.variable_scope('logits') as scope:
            logits = _build_fc_layer(self, inputs=fc_4, scope=scope,
                                     weight_init=tf.keras.initializers.glorot_uniform(),
                                     shape=(256, self.output_size[-1])) # 128
        self.output_node_name = logits.name
        return logits
    """


class CartPoleDQN(DQNPong):
    """
    DNN for the Cart-pole environment
    """
    def __init__(self, input_size,
                 output_size,
                 model_path,
                 scope='DenseAgent',
                 gamma=0.99,
                 pruning_start=0,
                 pruning_end=-1,
                 pruning_freq=int(10),
                 sparsity_start=0,
                 sparsity_end=int(10e5),
                 target_sparsity=0.9,
                 initial_sparsity=0,
                 epsilon=1.0,
                 epsilon_stop=0.05
                 ):
        super(CartPoleDQN, self).__init__(input_size=input_size, output_size=output_size,
                                          model_path=model_path, scope=scope, gamma=gamma,
                                          epsilon=epsilon, epsilon_stop=epsilon_stop)
        with self.graph.as_default():
            self.sparsity = pruning.get_weight_sparsity()
            self.hparams = pruning.get_pruning_hparams() \
                .parse('name={}, begin_pruning_step={}, end_pruning_step={}, target_sparsity={},'
                       ' sparsity_function_begin_step={},sparsity_function_end_step={},'
                       'pruning_frequency={},initial_sparsity={},'
                       ' sparsity_function_exponent={}'.format(scope,
                                                               pruning_start,
                                                               pruning_end,
                                                               target_sparsity,
                                                               sparsity_start,
                                                               sparsity_end,
                                                               pruning_freq,
                                                               initial_sparsity,
                                                               3))
            # note that the global step plays an important part in the pruning mechanism,
            # the higher the global step the closer the sparsity is to sparsity end
            self.pruning_obj = pruning.Pruning(self.hparams, global_step=self.global_step)
            self.mask_update_op = self.pruning_obj.conditional_mask_update_op()
            # the pruning objects defines the pruning mechanism, via the mask_update_op the model gets pruned
            # the pruning takes place at each training epoch and it objective to achieve the sparsity end HP
            self.init_variables(tf.global_variables())  # initialize variables in graph


    def _build_logits(self):
        with tf.variable_scope("fc_1") as scope:
            fc_1 = _build_fc_layer(self=self, inputs=self.input, scope=scope, shape=[self.input_size[-1], 256],
                                   activation=tf.nn.relu, weight_init=tf.keras.initializers.glorot_uniform())
        with tf.variable_scope("fc_2") as scope:
            fc_2 = _build_fc_layer(self=self, inputs=fc_1, scope=scope, shape=[256, 256],
                                   activation=tf.nn.relu, weight_init=tf.keras.initializers.glorot_uniform())
        with tf.variable_scope("fc_3") as scope:
            fc_3 = _build_fc_layer(self=self, inputs=fc_2, scope=scope, shape=[256, 128],
                                   activation=tf.nn.relu, weight_init=tf.keras.initializers.glorot_uniform())
        with tf.variable_scope("logits") as scope:
            logits = _build_fc_layer(self=self, inputs=fc_3, scope=scope, shape=[128, self.output_size[-1]],
                                   activation=tf.nn.relu, weight_init=tf.keras.initializers.glorot_uniform())

        return logits

    def _get_initial_size(self):
        return [self.input_size[-1] * 256, 256 * 256, 256 * 128, 128 * self.output_size[-1]]


class CartPoleDQNTarget(CartPoleDQN):
    """
    a target net for the above DQN agent, allows fast and stable training session
    """
    def __init__(self, input_size,
                 output_size, epsilon=0.00):
        self.input_size = input_size
        self.output_size = output_size
        self.graph = tf.Graph()
        self.epsilon = epsilon
        with self.graph.as_default():
            self._build_placeholders()
            self.logits = self._build_logits()
            self.saver = tf.train.Saver(var_list=tf.global_variables())
            self.init_variables(tf.global_variables())

    def sync(self, agent_path):
        """
        loads a new set of weights
        :param agent_path: the path of the agent we load the weights from
        """
        self.load_model(agent_path)


class StudentCartpole(CartPoleDQN):
    """
    an agent which is similar to DQNCartpole, the only difference is that this agent is implemented in a way that fits
    the policy distillation algorithm, and size can be dynamically shift according to the redundancy measure
    """
    def __init__(self, input_size,
                 output_size,
                 model_path,
                 scope='StudentCartpole',
                 gamma=0.99,
                 pruning_start=0,
                 pruning_end=-1,
                 pruning_freq=int(10),
                 sparsity_start=0,
                 sparsity_end=int(10e5),
                 target_sparsity=0.9,
                 initial_sparsity=0,
                 epsilon=0.0,
                 epsilon_stop=0.05,
                 tau=0.01,
                 redundancy=None,
                 last_measure=10e4):
        self.last_measure=last_measure
        self.tau = tau
        self.redundancy = redundancy
        super(StudentCartpole, self).__init__(input_size=input_size,
                                              output_size=output_size,
                                              model_path=model_path,
                                              scope=scope,
                                              gamma=gamma,
                                              pruning_start=pruning_start,
                                              pruning_end=pruning_end,
                                              pruning_freq=pruning_freq,
                                              sparsity_start=sparsity_start,
                                              sparsity_end=sparsity_end,
                                              target_sparsity=target_sparsity,
                                              initial_sparsity=initial_sparsity,
                                              epsilon=epsilon,
                                              epsilon_stop=epsilon_stop)

    def _calculate_sizes_according_to_redundancy(self):
        """
                the redundancy states what is the percentage of redundancy there is, at each layer
                :return: new sizes for each layer parameter!
        """
        assert self.redundancy is not None
        initial_sizes = self._get_initial_size()
        total_number_of_parameters_for_next_iteration = []
        for i, initial_num_of_params_at_layer in enumerate(initial_sizes):
            total_number_of_parameters_for_next_iteration.append(
                int(initial_num_of_params_at_layer * (1 - self.redundancy[i])))  # 1 - whats not important = important
        new_size_parameters = np.zeros(3)
        #  middle-out: finding the parameters from the inner layers because of the small fixed sizes of the input and output
        new_size_parameters[1] = (total_number_of_parameters_for_next_iteration[1] +
                                  total_number_of_parameters_for_next_iteration[2]) / (256 + 128)
        new_size_parameters[0] = total_number_of_parameters_for_next_iteration[0] / self.input_size[-1]
        new_size_parameters[2] = total_number_of_parameters_for_next_iteration[3] / self.output_size[-1]
        new_size_parameters[1] = (new_size_parameters[1] + (
                    total_number_of_parameters_for_next_iteration[1] + total_number_of_parameters_for_next_iteration[
                2]) / (new_size_parameters[2] + new_size_parameters[0])) * 0.5
        new_size_parameters[0] = (total_number_of_parameters_for_next_iteration[0] + total_number_of_parameters_for_next_iteration[1]) / (self.input_size[-1] + new_size_parameters[1])
        new_size_parameters[2] = (total_number_of_parameters_for_next_iteration[3] + total_number_of_parameters_for_next_iteration[2]) / (self.output_size[-1] + new_size_parameters[1])
        current_size = new_size_parameters[0] * self.input_size[-1] + new_size_parameters[1] * new_size_parameters[0] + new_size_parameters[2] * new_size_parameters[1] + \
                           self.output_size[-1] * new_size_parameters[2]
        # to ensure that the size is monotonically decreasing
        while current_size > self.last_measure:
                i = np.argmax(new_size_parameters)
                new_size_parameters[i] -= 1
                current_size = new_size_parameters[0] * self.input_size[-1] + new_size_parameters[1] * new_size_parameters[
                    0] + new_size_parameters[2] * new_size_parameters[1] + \
                               self.output_size[-1] * new_size_parameters[2]
        i = np.argmin(new_size_parameters)
        # avoiding zero parameter
        if new_size_parameters[i] < 1:
            new_size_parameters[i] = 3
            if i == 1:
                new_size_parameters[0] = (total_number_of_parameters_for_next_iteration[0] +
                                              total_number_of_parameters_for_next_iteration[1]) / (
                                                         self.input_size[-1] + new_size_parameters[1])
                new_size_parameters[2] = (total_number_of_parameters_for_next_iteration[3] +
                                              total_number_of_parameters_for_next_iteration[2]) / (
                                                         self.output_size[-1] + new_size_parameters[1])
            else:
                new_size_parameters[1] = (new_size_parameters[1] + (total_number_of_parameters_for_next_iteration[1] +
                                         total_number_of_parameters_for_next_iteration[2]) / (new_size_parameters[2] + new_size_parameters[0])) * 0.5





        for i, size in enumerate(new_size_parameters):
            new_size_parameters[i] = int(size)

        return new_size_parameters

    def _build_logits(self):
        if self.redundancy is None:
            fc_1_dim = 256
            fc_2_dim = 256
            fc_3_dim = 128
        else:
            new_size_parameters = self._calculate_sizes_according_to_redundancy()
            fc_1_dim = new_size_parameters[0]
            fc_2_dim = new_size_parameters[1]
            fc_3_dim = new_size_parameters[2]
        with tf.variable_scope("fc_1") as scope:
            fc_1 = _build_fc_layer(self=self, inputs=self.input, scope=scope,
                                   shape=[self.input_size[-1], fc_1_dim],
                                   activation=tf.nn.relu, weight_init=tf.keras.initializers.glorot_uniform())
        with tf.variable_scope("fc_2") as scope:
            fc_2 = _build_fc_layer(self=self, inputs=fc_1, scope=scope, shape=[fc_1_dim, fc_2_dim],
                                   activation=tf.nn.relu, weight_init=tf.keras.initializers.glorot_uniform())
        with tf.variable_scope("fc_3") as scope:
            fc_3 = _build_fc_layer(self=self, inputs=fc_2, scope=scope, shape=[fc_2_dim, fc_3_dim],
                                   activation=tf.nn.relu, weight_init=tf.keras.initializers.glorot_uniform())
        with tf.variable_scope("logits") as scope:
            logits = _build_fc_layer(self=self, inputs=fc_3, scope=scope, shape=[fc_3_dim, self.output_size[-1]],
                                     activation=tf.nn.relu, weight_init=tf.keras.initializers.glorot_uniform())

        return logits

    def _build_loss(self):
        """
        KLL  policy distillation loss function we want to minimize, according to Policy distillation
        """
        self.td_errors = abs(self.logits - self.target)
        # the td_error is crucial for Prioritized Experience replay, not operational for Policy distillation yet
        eps = 0.00001
        teacher_sharpend_dist = tf.nn.softmax(self.target / self.tau, dim=1) + eps
        teacher_sharpend_dist = tf.squeeze(teacher_sharpend_dist)
        student_dist = tf.nn.softmax(self.logits, dim=1) + eps
        return tf.reduce_sum(teacher_sharpend_dist * tf.log(teacher_sharpend_dist / student_dist))

    def change_loss_function(self):
        """
        :return: a loss that is compatible with vanilla DQN , not suited for Policy distillation
        """
        with self.graph.as_default():
            self.loss = super(CartPoleDQN, self)._build_loss()

    def change_train_op(self):
        """
        :return: a train_op that is compatible with vanilla DQN , not suited for Policy distillation
        """
        with self.graph.as_default():
            self.train_op = super(CartPoleDQN, self)._build_train_op()

    def _build_train_op(self):
        """
        the suggested optimizer according to Policy distillation
        """
        return tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                    global_step=self.global_step)

class Actor(DQNAgent):

    def __init__(self, input_size,
                 output_size,
                 model_path,
                 scope='Actor',
                 gamma=0.99,
                 pruning_start=0,
                 pruning_end=-1,
                 pruning_freq=int(10),
                 sparsity_start=0,
                 sparsity_end=int(10e5),
                 target_sparsity=0.9,
                 initial_sparsity=0):
        super(Actor, self).__init__(input_size=input_size,output_size=output_size,model_path=model_path, scope=scope,
                                    gamma=gamma)
        with self.graph.as_default():
            self.sparsity = pruning.get_weight_sparsity()
            self.hparams = pruning.get_pruning_hparams() \
                .parse('name={}, begin_pruning_step={}, end_pruning_step={}, target_sparsity={},'
                       ' sparsity_function_begin_step={},sparsity_function_end_step={},'
                       'pruning_frequency={},initial_sparsity={},'
                       ' sparsity_function_exponent={}'.format(scope,
                                                               pruning_start,
                                                               pruning_end,
                                                               target_sparsity,
                                                               sparsity_start,
                                                               sparsity_end,
                                                               pruning_freq,
                                                               initial_sparsity,
                                                               3))
            # note that the global step plays an important part in the pruning mechanism,
            # the higher the global step the closer the sparsity is to sparsity end
            self.pruning_obj = pruning.Pruning(self.hparams, global_step=self.global_step)
            self.mask_update_op = self.pruning_obj.conditional_mask_update_op()
            # the pruning objects defines the pruning mechanism, via the mask_update_op the model gets pruned
            # the pruning takes place at each training epoch and it objective to achieve the sparsity end HP
            self.init_variables(tf.global_variables())  # initialize variables in graph

    def prune(self):
        self.sess.run([self.mask_update_op])

    def get_model_sparsity(self):
        sparsity = self.sess.run(self.sparsity)
        return np.mean(sparsity)

    def _build_loss(self):
        """
               return:
               categorical_crossentropy is
               loss = - [one_hot_vector(action_chosen)] * log( action_dist)
        """

        advantage = self.target
        return tf.keras.losses.categorical_crossentropy(y_pred=self.logits, y_true=advantage)

    def _build_train_op(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='optimizer').minimize(self.loss)

    def select_action(self, qValues, explore=True):
        return np.random.choice(np.arange(self.output_size[-1]), p=qValues.ravel())

    def learn(self, target_batch, learning_rate, input, weights=None):  # weights are for compatibility
        _,loss = self.sess.run([self.train_op, self.loss], feed_dict={self.input: input,
                                                   self.target: target_batch,
                                                   self.learning_rate: learning_rate})
        return loss, 1  # for compatibaility


class ActorLunarlander(Actor):
    def __init__(self, input_size,
                 output_size,
                 model_path,
                 scope='ActorLunarLander',
                 gamma=0.99,
                 pruning_start=0,
                 pruning_end=-1,
                 pruning_freq=int(10),
                 sparsity_start=0,
                 sparsity_end=int(10e5),
                 target_sparsity=0.9):
        super(ActorLunarlander, self).__init__(input_size=input_size, output_size=output_size, model_path=model_path, scope=scope,
                                               gamma=gamma, pruning_freq=pruning_freq, pruning_start=pruning_start,
                                               pruning_end=pruning_end, sparsity_end=sparsity_end, sparsity_start=sparsity_start,
                                               target_sparsity=target_sparsity)

    def _build_logits(self):
            with tf.variable_scope("fc_1") as scope:
                fc_1 = _build_fc_layer(self=self, inputs=self.input, scope=scope, shape=[self.input_size[-1], 64],
                                       activation=tf.nn.relu, weight_init=tf.keras.initializers.glorot_uniform())
            with tf.variable_scope("fc_2") as scope:
                fc_2 = _build_fc_layer(self=self, inputs=fc_1, scope=scope, shape=[64, 64],
                                       activation=tf.nn.relu, weight_init=tf.keras.initializers.glorot_uniform())
            with tf.variable_scope("fc_3") as scope:
                fc_3 = _build_fc_layer(self=self, inputs=fc_2, scope=scope, shape=[64, 64],
                                       activation=tf.nn.relu, weight_init=tf.keras.initializers.glorot_uniform())
            with tf.variable_scope("logits") as scope:
                self.before_softmax = _build_fc_layer(self=self, inputs=fc_3, scope=scope, shape=[64, self.output_size[-1]],
                                                      weight_init=tf.keras.initializers.glorot_uniform())

                action_dist = tf.nn.softmax(self.before_softmax)
            return action_dist

    def get_before_softmax(self,state):
        return self.sess.run([self.before_softmax, self.logits], feed_dict={self.input: state})

    def _get_initial_size(self):
        return [8 * 64, 64 * 64, 64 * 64, 64 * 4]


class StudentActorLunarlander(ActorLunarlander):
    
    def __init__(self, input_size,
                 output_size,
                 model_path,
                 scope='StudentActorLunarLander',
                 gamma=0.99,
                 pruning_start=0,
                 pruning_end=-1,
                 pruning_freq=int(10),
                 sparsity_start=0,
                 sparsity_end=int(10e5),
                 target_sparsity=0.9, tau=0.01,
                 redundancy=None,
                 last_measure=10e4):
        self.tau = tau
        self.last_measure=last_measure
        self.redundancy = redundancy
        super(StudentActorLunarlander, self).__init__(input_size=input_size, output_size=output_size, model_path=model_path, scope=scope,
                                               gamma=gamma, pruning_freq=pruning_freq, pruning_start=pruning_start,
                                               pruning_end=pruning_end, sparsity_end=sparsity_end, sparsity_start=sparsity_start,
                                               target_sparsity=target_sparsity)



    def _calculate_sizes_according_to_redundancy(self):
        assert self.redundancy is not None
        initial_sizes = self._get_initial_size()
        total_number_of_parameters_for_next_iteration = []
        for i, initial_num_of_params_at_layer in enumerate(initial_sizes):
            total_number_of_parameters_for_next_iteration.append(
                int(initial_num_of_params_at_layer * (1 - self.redundancy[i])))  # 1 - whats not important = important
        new_size_parameters = np.zeros(3)
        new_size_parameters[1] = (total_number_of_parameters_for_next_iteration[1] +
                                  total_number_of_parameters_for_next_iteration[2]) / (64 + 64)
        new_size_parameters[0] = total_number_of_parameters_for_next_iteration[0] / self.input_size[-1]
        new_size_parameters[2] = total_number_of_parameters_for_next_iteration[3] / self.output_size[-1]
        new_size_parameters[1] = (new_size_parameters[1] + (
                    total_number_of_parameters_for_next_iteration[1] + total_number_of_parameters_for_next_iteration[
                2]) / (new_size_parameters[2] + new_size_parameters[0])) * 0.5

        current_size = new_size_parameters[0] * self.input_size[-1] + new_size_parameters[1] * new_size_parameters[0] + \
                       new_size_parameters[2] * new_size_parameters[1] + \
                       self.output_size[-1] * new_size_parameters[2]

        # to ensure that the size is monotonically decreasing
        while current_size > self.last_measure:
            i = np.argmax(new_size_parameters)
            new_size_parameters[i] -= 1
            current_size = new_size_parameters[0] * self.input_size[-1] + new_size_parameters[1] * new_size_parameters[
                0] + new_size_parameters[2] * new_size_parameters[1] + \
                           self.output_size[-1] * new_size_parameters[2]
        # to ensure no zero parameters
        i = np.argmin(new_size_parameters)
        while new_size_parameters[i] < 1:
            new_size_parameters[i] = 4
            if i == 1:
                new_size_parameters[0] = (total_number_of_parameters_for_next_iteration[0] +
                                          total_number_of_parameters_for_next_iteration[1]) / (
                                                 self.input_size[-1] + new_size_parameters[1])
                new_size_parameters[2] = (total_number_of_parameters_for_next_iteration[3] +
                                          total_number_of_parameters_for_next_iteration[2]) / (
                                                 self.output_size[-1] + new_size_parameters[1])
            else:
                new_size_parameters[1] = (new_size_parameters[1] + (total_number_of_parameters_for_next_iteration[1] +
                                                                    total_number_of_parameters_for_next_iteration[
                                                                        2]) / (
                                                      new_size_parameters[2] + new_size_parameters[0])) * 0.5
            i = np.argmin(new_size_parameters)

        for i, size in enumerate(new_size_parameters):
            new_size_parameters[i] = int(size)

        return new_size_parameters

    def _build_logits(self):
        if  self.redundancy is None:
            fc_1_dim = 64
            fc_2_dim = 64
            fc_3_dim = 64
        else:
            new_size_parameters = self._calculate_sizes_according_to_redundancy()
            fc_1_dim = new_size_parameters[0]
            fc_2_dim = new_size_parameters[1]
            fc_3_dim = new_size_parameters[2]
        with tf.variable_scope("fc_1") as scope:
            fc_1 = _build_fc_layer(self=self, inputs=self.input, scope=scope, shape=[self.input_size[-1], fc_1_dim],
                                   activation=tf.nn.relu, weight_init=tf.keras.initializers.glorot_uniform())
        with tf.variable_scope("fc_2") as scope:
            fc_2 = _build_fc_layer(self=self, inputs=fc_1, scope=scope, shape=[fc_1_dim, fc_2_dim],
                                   activation=tf.nn.relu, weight_init=tf.keras.initializers.glorot_uniform())
        with tf.variable_scope("fc_3") as scope:
            fc_3 = _build_fc_layer(self=self, inputs=fc_2, scope=scope, shape=[fc_2_dim, fc_3_dim],
                                   activation=tf.nn.relu, weight_init=tf.keras.initializers.glorot_uniform())
        with tf.variable_scope("logits") as scope:
            action_dist = _build_fc_layer(self=self, inputs=fc_3, scope=scope, shape=[fc_3_dim, self.output_size[-1]],
                                               activation=tf.nn.softmax, weight_init=tf.keras.initializers.glorot_uniform())
        return action_dist


    def _build_loss(self):
        """
        KLL  policy distillation loss function we want to minimize, according to Policy distillation
        """
        eps = 0.00001
        teacher_sharpend_dist = tf.nn.softmax(self.target / self.tau, dim=1) + eps
        teacher_sharpend_dist = tf.squeeze(teacher_sharpend_dist)
        student_dist = self.logits + eps  # logits is already after softmax for actor models
        return tf.reduce_sum(teacher_sharpend_dist * tf.log(teacher_sharpend_dist / student_dist))

    def change_loss_function(self):
        """
        :return: a loss that is compatible with vanilla DQN , not suited for Policy distillation
        """
        with self.graph.as_default():
            self.loss = super(ActorLunarlander, self)._build_loss()

    def change_train_op(self):
        """
        :return: a train_op that is compatible with vanilla DQN , not suited for Policy distillation
        """
        with self.graph.as_default():
            self.train_op = super(ActorLunarlander, self)._build_train_op()

    def _build_train_op(self):
        """
        the suggested optimizer according to Policy distillation
        """
        return tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


    

class CriticLunarLander(DQNAgent):

    def __init__(self, input_size,
                 output_size,
                 model_path,
                 scope='CriticLunarLander',
                 gamma=0.99):
        super(CriticLunarLander, self).__init__(input_size=input_size, output_size=output_size,
                                                model_path=model_path, scope=scope, gamma=gamma)
        
    def _build_logits(self):
        with tf.variable_scope("fc_1") as scope:
            fc_1 = _build_fc_layer(self=self, inputs=self.input, scope=scope, shape=[self.input_size[-1], 64],
                                   activation=tf.nn.relu, weight_init=tf.keras.initializers.glorot_uniform())
        with tf.variable_scope("fc_2") as scope:
            fc_2 = _build_fc_layer(self=self, inputs=fc_1, scope=scope, shape=[64, 64],
                                   activation=tf.nn.relu, weight_init=tf.keras.initializers.glorot_uniform())
        with tf.variable_scope("fc_3") as scope:
            fc_3 = _build_fc_layer(self=self, inputs=fc_2, scope=scope, shape=[64, 64],
                                   activation=tf.nn.relu, weight_init=tf.keras.initializers.glorot_uniform())
        with tf.variable_scope("logits") as scope:
            logits = _build_fc_layer(self=self, inputs=fc_3, scope=scope, shape=[64, self.output_size[-1]],
                                          weight_init=tf.keras.initializers.glorot_uniform())
        return logits

    def _build_loss(self):
        return tf.keras.losses.mean_squared_error(y_true=self.target, y_pred=self.logits)

    def _build_train_op(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def learn(self, target_batch, learning_rate, input):
        _ = self.sess.run(self.train_op, feed_dict={self.input: input,
                                                   self.target: target_batch,
                                                   self.learning_rate: learning_rate})

class CriticLunarLanderTarget(CriticLunarLander):
    """
    a target net for the above agent, allows fast and stable training session
    """
    def __init__(self, input_size,
                 output_size, epsilon=0.00):
        self.input_size = input_size
        self.output_size = output_size
        self.graph = tf.Graph()
        self.epsilon = epsilon
        with self.graph.as_default():
            self._build_placeholders()
            self.logits = self._build_logits()
            self.saver = tf.train.Saver(var_list=tf.global_variables())
            self.init_variables(tf.global_variables())

    def sync(self, agent_path):
        """
        loads a new set of weights
        :param agent_path: the path of the agent we load the weights from
        """
        self.load_model(agent_path)
