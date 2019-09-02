import tensorflow as tf
import numpy as np

import utils


class ModelBasedPolicy(object):

    def __init__(self,
                 env,
                 init_dataset,
                 horizon=15,
                 num_random_action_selection=4096,
                 nn_layers=1):
        self._cost_fn = env.cost_fn
        self._state_dim = env.observation_space.shape[0]
        self._action_dim = env.action_space.shape[0]
        self._action_space_low = env.action_space.low
        self._action_space_high = env.action_space.high
        self._init_dataset = init_dataset
        self._horizon = horizon
        self._num_random_action_selection = num_random_action_selection
        self._nn_layers = nn_layers
        self._learning_rate = 1e-3

        self._sess, self._state_ph, self._action_ph, self._next_state_ph,\
            self._next_state_pred, self._loss, self._optimizer, self._best_action = self._setup_graph()

    def _setup_placeholders(self):
        """
            Creates the placeholders used for training, prediction, and action selection

            returns:
                state_ph: current state
                action_ph: current_action
                next_state_ph: next state

            implementation details:
                (a) the placeholders should have 2 dimensions,
                    in which the 1st dimension is variable length (i.e., None)
        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        state_ph = tf.placeholder(tf.float32, shape=(None, self._state_dim))
        action_ph = tf.placeholder(tf.float32, shape=(None, self._action_dim))
        next_state_ph = tf.placeholder(tf.float32, shape=(None, self._state_dim))
        # raise NotImplementedError

        return state_ph, action_ph, next_state_ph

    def _dynamics_func(self, state, action, reuse):
        """
            Takes as input a state and action, and predicts the next state

            returns:
                next_state_pred: predicted next state

            implementation details (in order):
                (a) Normalize both the state and action by using the statistics of self._init_dataset and
                    the utils.normalize function
                (b) Concatenate the normalized state and action
                (c) Pass the concatenated, normalized state-action tensor through a neural network with
                    self._nn_layers number of layers using the function utils.build_mlp. The resulting output
                    is the normalized predicted difference between the next state and the current state
                (d) Unnormalize the delta state prediction, and add it to the current state in order to produce
                    the predicted next state

        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        d = self._init_dataset
        state_norm = utils.normalize(state, d.state_mean, d.state_std)
        action_norm = utils.normalize(action, d.action_mean, d.action_std)
        input_layer=tf.concat([state_norm, action_norm],axis=1)
        delta = utils.build_mlp(
            input_layer,
            output_dim=self._state_dim,
            scope = 'dynamic_pred',
            n_layers=self._nn_layers,
            reuse=reuse
        )
        delta_ = utils.unnormalize(delta, d.delta_state_mean, d.delta_state_std)
        next_state_pred = state + delta_
        # raise NotImplementedError
        return next_state_pred

    def _setup_training(self, state_ph, next_state_ph, next_state_pred):
        """
            Takes as input the current state, next state, and predicted next state, and returns
            the loss and optimizer for training the dynamics model

            returns:
                loss: Scalar loss tensor
                optimizer: Operation used to perform gradient descent

            implementation details (in order):
                (a) Compute both the actual state difference and the predicted state difference
                (b) Normalize both of these state differences by using the statistics of self._init_dataset and
                    the utils.normalize function
                (c) The loss function is the mean-squared-error between the normalized state difference and
                    normalized predicted state difference
                (d) Create the optimizer by minimizing the loss using the Adam optimizer with self._learning_rate

        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        d = self._init_dataset
        act_state_diff = utils.normalize(next_state_ph-state_ph, d.delta_state_mean, d.delta_state_std) 
        pred_state_diff = utils.normalize(next_state_pred-state_ph, d.delta_state_mean, d.delta_state_std)
        loss = tf.losses.mean_squared_error(act_state_diff, pred_state_diff)
        optimizer = tf.train.AdamOptimizer(learning_rate = self._learning_rate)
        optimizer = optimizer.minimize(loss)
        # raise NotImplementedError

        return loss, optimizer

    def _setup_action_selection(self, state_ph):
        """
            Computes the best action from the current state by using randomly sampled action sequences
            to predict future states, evaluating these predictions according to a cost function,
            selecting the action sequence with the lowest cost, and returning the first action in that sequence

            returns:
                best_action: the action that minimizes the cost function (tensor with shape [self._action_dim])

            implementation details (in order):
                (a) We will assume state_ph has a batch size of 1 whenever action selection is performed
                (b) Randomly sample uniformly self._num_random_action_selection number of action sequences,
                    each of length self._horizon
                (c) Starting from the input state, unroll each action sequence using your neural network
                    dynamics model
                (d) While unrolling the action sequences, keep track of the cost of each action sequence
                    using self._cost_fn
                (e) Find the action sequence with the lowest cost, and return the first action in that sequence

            Hints:
                (i) self._cost_fn takes three arguments: states, actions, and next states. These arguments are
                    2-dimensional tensors, where the 1st dimension is the batch size and the 2nd dimension is the
                    state or action size
                (ii) You should call self._dynamics_func and self._cost_fn a total of self._horizon times
                (iii) Use tf.random_uniform(...) to generate the random action sequences

        """
        ### PROBLEM 2
        ### YOUR CODE HERE
        sample_actions = tf.random_uniform(
            shape = [self._num_random_action_selection, self._horizon, self._action_dim],
            minval = self._action_space_low,
            maxval = self._action_space_high
        )
        # sample_actions = sample_uniform*(self._action_space_high-self._action_space_low) + self._action_space_low


        state_batch = tf.tile(state_ph, [self._num_random_action_selection, 1])
        costs = tf.zeros(shape=[self._num_random_action_selection])

        for i in range(self._horizon):
            action_batch = sample_actions[:,i,:]    # num_selec * act_dim
            state_preds = self._dynamics_func(state_batch, action_batch, reuse=True)
            costs += self._cost_fn(state_batch, action_batch, state_preds)
            state_batch = state_preds
        best_index = tf.argmin(costs)
        # return First Action
        best_action = sample_actions[best_index][0]    # act_dim
        # raise NotImplementedError
        return best_action

    def _setup_graph(self):
        """
        Sets up the tensorflow computation graph for training, prediction, and action selection

        The variables returned will be set as class attributes (see __init__)
        """
        sess = tf.Session()

        ### PROBLEM 1
        ### YOUR CODE HERE
        state_ph, action_ph, next_state_ph = self._setup_placeholders()
        next_state_pred = self._dynamics_func(state_ph, action_ph, reuse=False)
        loss, optimizer = self._setup_training(state_ph, next_state_ph, next_state_pred)
        ### PROBLEM 2
        ### YOUR CODE HERE
        best_action = self._setup_action_selection(state_ph)

        sess.run(tf.global_variables_initializer())

        return sess, state_ph, action_ph, next_state_ph, \
                next_state_pred, loss, optimizer, best_action

    def train_step(self, states, actions, next_states):
        """
        Performs one step of gradient descent

        returns:
            loss: the loss from performing gradient descent
        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        loss, _ = self._sess.run([self._loss, self._optimizer], feed_dict={self._state_ph: states, 
                                        self._action_ph: actions,
                                        self._next_state_ph: next_states})
        # raise NotImplementedError
        return loss

    def predict(self, state, action):
        """
        Predicts the next state given the current state and action

        returns:
            next_state_pred: predicted next state

        implementation detils:
            (i) The state and action arguments are 1-dimensional vectors (NO batch dimension)
        """
        assert np.shape(state) == (self._state_dim,)
        assert np.shape(action) == (self._action_dim,)

        ### PROBLEM 1
        ### YOUR CODE HERE
        next_state_pred = self._sess.run(self._next_state_pred, feed_dict = {self._state_ph: [state],
                                                            self._action_ph: [action]})[0]
        # raise NotImplementedError

        assert np.shape(next_state_pred) == (self._state_dim,)
        return next_state_pred

    def get_action(self, state):
        """
        Computes the action that minimizes the cost function given the current state

        returns:
            best_action: the best action
        """
        assert np.shape(state) == (self._state_dim,)

        ### PROBLEM 2
        ### YOUR CODE HERE
        best_action = self._sess.run(self._best_action, feed_dict={self._state_ph: [state]})
        # raise NotImplementedError

        assert np.shape(best_action) == (self._action_dim,)
        return best_action
