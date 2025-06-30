# -*- coding: utf-8 -*-
"""
loss_functions


"""

import tensorflow as tf
from tensorflow import keras
import argparse
#from keras import backend as K
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow as tf

# # Define a custom loss function
# def custom_loss(y_true, y_pred):
#   # Example custom loss: Mean Squared Error
#   return tf.reduce_mean(tf.square(y_true - y_pred))

# Define a custom loss function
def custom_loss(y_true, y_pred):
  # Convert y_true to one-hot encoding
  y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=10)
  # Example custom loss: Mean Squared Error
  return tf.reduce_mean(tf.square(y_true - y_pred))

# Define a noise-aware loss function
def noise_aware_loss(y_true, y_pred, noise_rate=0.1):
  """
  Calculates a noise-aware loss function that considers potential label noise.

  Args:
    y_true: True labels.
    y_pred: Predicted probabilities.
    noise_rate: Estimated noise rate in the labels.

  Returns:
    The noise-aware loss value.
  """
  y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=10)
  # Calculate the standard cross-entropy loss
  cross_entropy_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

  # Consider the possibility of label noise
  noise_aware_term = (1 - noise_rate) * cross_entropy_loss + \
                     noise_rate * tf.keras.losses.categorical_crossentropy(1 - y_true, y_pred)

  return noise_aware_term




def crossentropy_reed_wrap(_beta):
    def crossentropy_reed_core(y_true, y_pred):
        """
        This loss function is proposed in:
        Reed et al. "Training Deep Neural Networks on Noisy Labels with Bootstrapping", 2014

        :param y_true:
        :param y_pred:
        :return:
        """

        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=10)
        # hyper param
        print(_beta)
        y_pred = K.clip(y_pred, K.epsilon(), 1)

        # (1) dynamically update the targets based on the current state of the model: bootstrapped target tensor
        # use predicted class proba directly to generate regression targets
        y_true_update = _beta * y_true + (1 - _beta) * y_pred

        # (2) compute loss as always
        _loss = -K.sum(y_true_update * K.log(y_pred), axis=-1)

        return _loss
    return crossentropy_reed_core

def symmetric_cross_entropy(alpha, beta):
  """
  Calculates the Symmetric Cross Entropy loss function.

  This loss function is designed to be robust to noisy labels.
  It combines Cross Entropy with Reverse Cross Entropy.

  Args:
    alpha: Weighting parameter for the Cross Entropy term.
    beta: Weighting parameter for the Reverse Cross Entropy term.

  Returns:
    A function that calculates the Symmetric Cross Entropy loss.
  """
  def loss(y_true, y_pred):
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=10)
    y_true_1 = y_true
    y_pred_1 = y_pred

    y_true_2 = y_pred
    y_pred_2 = y_true

    ce_loss = tf.keras.losses.categorical_crossentropy(y_true_1, y_pred_1)
    rce_loss = tf.keras.losses.categorical_crossentropy(y_true_2, y_pred_2)

    return alpha * ce_loss + beta * rce_loss
  return loss

def lq_loss_wrap(_q):
    def lq_loss_core(y_true, y_pred):
        """
        This loss function is proposed in:
         Zhilu Zhang and Mert R. Sabuncu, "Generalized Cross Entropy Loss for Training Deep Neural Networks with
         Noisy Labels", 2018
        https://arxiv.org/pdf/1805.07836.pdf
        :param y_true:
        :param y_pred:
        :return:
        """

        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=10)
        # hyper param
        print(_q)

        _tmp = y_pred * y_true
        _loss = K.max(_tmp, axis=-1)

        # compute the Lq loss between the one-hot encoded label and the prediction
        _loss = (1 - (_loss + 10 ** (-8)) ** _q) / _q

        return _loss
    return lq_loss_core



def mae_loss_multi_class(y_true, y_pred):
    """
    Mean Absolute Error (MAE) loss for multi-class classification.

    Args:
        y_true: True labels (one-hot encoded or integer).
        y_pred: Predicted logits (raw scores before softmax).

    Returns:
        tf.Tensor: MAE loss.
    """
    # If y_true is not one-hot encoded, convert it
    if len(y_true.shape) == len(y_pred.shape) - 1:
        y_true = tf.one_hot(tf.cast(y_true, dtype=tf.int32), depth=tf.shape(y_pred)[-1])

    # Convert logits to probabilities (softmax)
    probs = tf.nn.softmax(y_pred, axis=-1)  # Use axis=-1 for Keras

    # Compute MAE loss
    loss = tf.reduce_mean(tf.abs(probs - y_true), axis=-1)  # Mean over classes

    # Apply reduction (mean by default)
    return tf.reduce_mean(loss)

# class ActivePassiveLoss(nn.Module):
#     def __init__(self, lambda_=0.5, reduction='mean'):
#         super(ActivePassiveLoss, self).__init__()
#         self.lambda_ = lambda_  # Weight for Active vs. Passive Loss
#         self.reduction = reduction

    # def forward(self, logits, targets):
    #     # Active Loss (Standard Cross-Entropy)
    #     active_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

    #     # Passive Loss (Reverse Cross-Entropy: encourages predictions to avoid wrong labels)
    #     passive_loss = F.binary_cross_entropy_with_logits(logits, 1 - targets, reduction='none')

    #     # Combine losses
    #     loss = self.lambda_ * active_loss + (1 - self.lambda_) * passive_loss

    #     if self.reduction == 'mean':
    #         return loss.mean()
    #     elif self.reduction == 'sum':
    #         return loss.sum()
    #     else:
    #         return loss

    # def forward(self, logits, targets):
    #   probs = torch.sigmoid(logits)
    #   confidence = torch.abs(probs - 0.5) * 2  # Measure confidence (0 to 1)
    #   lambda_ = confidence.mean()  # Higher confidence → rely more on Active Loss

    #   active_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    #   passive_loss = F.binary_cross_entropy_with_logits(logits, 1 - targets, reduction='none')
    #   loss = lambda_ * active_loss + (1 - lambda_) * passive_loss

    #   return loss.mean() if self.reduction == 'mean' else loss


def bi_tempered_logistic_loss(t1=1.0, t2=1.0):
    """
    Bi-Tempered Logistic Loss for Keras, with t1 and t2 as top-level parameters.

    Args:
        t1 (float): Temperature parameter 1.
        t2 (float): Temperature parameter 2.

    Returns:
        function: A loss function that takes y_true, y_pred, and num_iters.
    """
    def bi_tempered_loss_fn(y_true, y_pred, num_iters=5):
        """
        Inner bi-tempered logistic loss function.

        Args:
            y_true (tf.Tensor): True labels (one-hot encoded or integer).
            y_pred (tf.Tensor): Predicted logits (raw scores before softmax).
            num_iters (int, optional): Number of iterations for numerical stability. Defaults to 5.

        Returns:
            tf.Tensor: Bi-tempered loss.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if len(y_true.shape) == len(y_pred.shape) - 1:
            y_true = tf.one_hot(tf.cast(y_true, dtype=tf.int32), depth=tf.shape(y_pred)[-1])
            y_true = tf.cast(y_true, tf.float32)

        # These helper functions can be nested further or defined outside
        # but within the scope of bi_tempered_logistic_loss_nested for clarity
        def log_t(u, t):
            if t == 1.0:
                return tf.math.log(tf.clip_by_value(u, 1e-8, 1.0))
            else:
                return (tf.pow(u, 1.0 - t) - 1.0) / (1.0 - t)

        def exp_t(u, t):
            if t == 1.0:
                return tf.math.exp(tf.clip_by_value(u, -100.0, 100.0))
            else:
                return tf.clip_by_value(1.0 + (1.0 - t) * u, 1e-6, 1e6)**(1.0 / (1.0 - t))

        probs = tf.nn.softmax(y_pred)
        probs_clipped = tf.clip_by_value(probs, 1e-8, 1.0)

        # Calculate loss components using the outer t1 and t2
        loss1 = (1.0 - tf.reduce_sum(y_true * tf.pow(probs_clipped, (1.0 - t1)), axis=-1)) / (1.0 - t1)
        loss2 = (1.0 - tf.reduce_sum(y_true * tf.pow(probs_clipped, (1.0 - t2)), axis=-1)) / (1.0 - t2)

        return tf.reduce_mean(0.5 * (loss1 + loss2))

    return bi_tempered_loss_fn


def zero_one_loss(y_true, y_pred):
  """
  Calculates the 0-1 loss (also known as classification error).

  Args:
    y_true: A NumPy array of true labels.
    y_pred: A NumPy array of predicted labels.

  Returns:
    The 0-1 loss, which is the fraction of incorrectly classified samples.
  """
  if y_true.shape != y_pred.shape:
    raise ValueError("Shape of true labels and predicted labels must match.")

  incorrect_predictions = np.sum(y_true != y_pred)
  total_samples = len(y_true)
  loss = incorrect_predictions / total_samples
  return loss



def crossentropy_outlier_wrap(_l):
    def crossentropy_outlier_core(y_true, y_pred):

        # hyper param
        print(_l)
        y_pred = K.clip(y_pred, K.epsilon(), 1)

        # compute loss for every data point
        _loss = -K.sum(y_true * K.log(y_pred), axis=-1)

        def _get_real_median(_v):
            """
            given a tensor with shape (batch_size,), compute and return the median

            :param v:
            :return:
            """
            _val = tf.nn.top_k(_v, 33).values
            return 0.5 * (_val[-1] + _val[-2])

        _mean_loss, _var_loss = tf.nn.moments(_loss, axes=[0])
        _median_loss = _get_real_median(_loss)
        _std_loss = tf.sqrt(_var_loss)

        # threshold
        t_l = _median_loss + _l*_std_loss
        _mask_l = 1 - (K.cast(K.greater(_loss, t_l), 'float32'))
        _loss = _loss * _mask_l

        return _loss
    return crossentropy_outlier_core


import tensorflow as tf

def agce_loss_fn(num_classes=10, a=1, q=2, scale=1.):
    """
    Returns a Keras-compatible AGCELoss function.

    Args:
        num_classes (int): The total number of classes.
        a (int or float): Hyperparameter 'a' for the AGCELoss.
        q (int or float): Hyperparameter 'q' for the AGCELoss.
        scale (float): Scaling factor for the loss.

    Returns:
        A callable loss function that takes (y_true, y_pred) as arguments.
    """
    # Cast a and q to float32 at the beginning to ensure consistent types
    a_float = tf.cast(a, dtype=tf.float32)
    q_float = tf.cast(q, dtype=tf.float32)

    def loss(y_true, y_pred):
        # Ensure y_pred is float32 for calculations
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        # Apply softmax to predictions
        pred_softmax = tf.nn.softmax(y_pred, axis=-1)

        # Convert labels to one-hot encoding
        # y_true might come in different shapes; ensure it's compatible for one_hot
        # Assuming y_true is integer labels:
        if len(y_true.shape) == 2 and y_true.shape[1] == num_classes: # Already one-hot
            label_one_hot = y_true
        else: # Assuming integer labels
            label_one_hot = tf.one_hot(tf.cast(y_true, dtype=tf.int32), num_classes, dtype=tf.float32)

        # Calculate the term inside the power function
        term_to_power = tf.reduce_sum(label_one_hot * pred_softmax, axis=-1)

        # Calculate the loss
        # Ensure both terms in subtraction are float32
        loss_val = (tf.cast(tf.pow(a_float + 1., q_float), dtype=tf.float32) - tf.pow(a_float + term_to_power, q_float)) / q_float

        # Apply scaling and return the mean loss
        return tf.reduce_mean(loss_val) * scale
    return loss


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

def nfl_and_mae_loss(alpha=1., beta=1., num_classes=10, gamma=0.5):
    """
    Returns a Keras loss function that combines Normalized Focal Loss and MAE Loss.

    Args:
        alpha (float): Scale factor for Normalized Focal Loss.
        beta (float): Scale factor for MAE Loss.
        num_classes (int): The total number of classes.
        gamma (float): Gamma parameter for Focal Loss.

    Returns:
        A callable Keras loss function.
    """

    def normalized_focal_loss(y_true, y_pred):
        """
        Calculates the Normalized Focal Loss.
        This is a nested function, closing over 'gamma', 'num_classes', and 'alpha'.
        """
        # Ensure y_true is integer and correct shape for one_hot
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.squeeze(y_true) # Remove any extra dimensions, e.g., if (batch_size, 1)

        logpt = tf.nn.log_softmax(y_pred, axis=-1)

        # Calculate normalizor, mimicking PyTorch's .data.exp() with tf.stop_gradient
        normalizor = tf.reduce_sum(-1 * tf.pow(1.0 - tf.exp(tf.stop_gradient(logpt)), gamma) * tf.stop_gradient(logpt), axis=-1)

        # Gather logpt for the true class using one-hot encoding
        y_true_one_hot = tf.one_hot(y_true, depth=num_classes, dtype=tf.float32)
        logpt_true_class = tf.reduce_sum(logpt * y_true_one_hot, axis=-1)

        pt = tf.exp(logpt_true_class) # Equivalent to logpt.data.exp() for pt

        # Loss calculation: -1 * (1-pt)**gamma * logpt
        loss = -1.0 * tf.pow(1.0 - pt, gamma) * logpt_true_class

        # Scale and normalize the loss
        # Added K.epsilon() for numerical stability
        loss = alpha * loss / (normalizor + K.epsilon())

        # size_average is True by default in the original, so we return the mean
        return tf.reduce_mean(loss)

    def mae_loss(y_true, y_pred):
        """
        Calculates the Mean Absolute Error Loss.
        This is a nested function, closing over 'num_classes' and 'beta'.
        """
        # Ensure y_true is integer and correct shape for one_hot
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.squeeze(y_true)

        pred_softmax = tf.nn.softmax(y_pred, axis=-1)
        label_one_hot = tf.one_hot(y_true, depth=num_classes, dtype=tf.float32)

        # Loss: 1. - sum(label_one_hot * pred, dim=1)
        loss = 1.0 - tf.reduce_sum(label_one_hot * pred_softmax, axis=-1)

        return beta * tf.reduce_mean(loss)

    # Combine the two losses
    def combined_loss(y_true, y_pred):
        return normalized_focal_loss(y_true, y_pred) + mae_loss(y_true, y_pred)

    return combined_loss

nflmae_combined_loss = nfl_and_mae_loss(alpha=1., beta=1., num_classes=10, gamma=0.5)


#########################################################################
# from here on we distinguish data points in the batch, based on its origin
# we only apply robustness measures to the data points coming from the noisy subset
# Therefore, the next functions are used only when training with the entire train set
#########################################################################


def crossentropy_reed_origin_wrap(_beta):
    def crossentropy_reed_origin_core(y_true, y_pred):
        # hyper param
        print(_beta)

        # 1) determine the origin of the patch, as a boolean vector in y_true_flag
        # (True = patch from noisy subset)
        _y_true_flag = K.greater(K.sum(y_true, axis=-1), 90)

        # 2) convert the input y_true (with flags inside) into a valid y_true one-hot-vector format
        # attenuating factor for data points that need it (those that came with a one-hot of 100)
        _mask_reduce = K.cast(_y_true_flag, 'float32') * 0.01

        # identity factor for standard one-hot vectors
        _mask_keep = K.cast(K.equal(_y_true_flag, False), 'float32')

        # combine 2 masks
        _mask = _mask_reduce + _mask_keep

        _y_true_shape = K.shape(y_true)
        _mask = K.reshape(_mask, (_y_true_shape[0], 1))

        # applying mask to have a valid y_true that we can use as always
        y_true = y_true * _mask

        y_true = K.clip(y_true, K.epsilon(), 1)
        y_pred = K.clip(y_pred, K.epsilon(), 1)

        # (1) dynamically update the targets based on the current state of the model: bootstrapped target tensor
        # use predicted class proba directly to generate regression targets
        y_true_bootstrapped = _beta * y_true + (1 - _beta) * y_pred

        # at this point we have 2 versions of y_true
        # decide which target label to use for each datapoint
        _mask_noisy = K.cast(_y_true_flag, 'float32')                   # only allows patches from noisy set
        _mask_clean = K.cast(K.equal(_y_true_flag, False), 'float32')   # only allows patches from clean set
        _mask_noisy = K.reshape(_mask_noisy, (_y_true_shape[0], 1))
        _mask_clean = K.reshape(_mask_clean, (_y_true_shape[0], 1))

        # points coming from clean set use the standard true one-hot vector. dim is (batch_size, 1)
        # points coming from noisy set use the Reed bootstrapped target tensor
        y_true_final = y_true * _mask_clean + y_true_bootstrapped * _mask_noisy

        # (2) compute loss as always
        _loss = -K.sum(y_true_final * K.log(y_pred), axis=-1)

        return _loss
    return crossentropy_reed_origin_core

def lq_loss_origin_wrap(_q):
    def lq_loss_origin_core(y_true, y_pred):

        # hyper param
        print(_q)

        # 1) determine the origin of the patch, as a boolean vector in y_true_flag
        # (True = patch from noisy subset)
        _y_true_flag = K.greater(K.sum(y_true, axis=-1), 90)

        # 2) convert the input y_true (with flags inside) into a valid y_true one-hot-vector format
        # attenuating factor for data points that need it (those that came with a one-hot of 100)
        _mask_reduce = K.cast(_y_true_flag, 'float32') * 0.01

        # identity factor for standard one-hot vectors
        _mask_keep = K.cast(K.equal(_y_true_flag, False), 'float32')

        # combine 2 masks
        _mask = _mask_reduce + _mask_keep

        _y_true_shape = K.shape(y_true)
        _mask = K.reshape(_mask, (_y_true_shape[0], 1))

        # applying mask to have a valid y_true that we can use as always
        y_true = y_true * _mask

        y_true = K.clip(y_true, K.epsilon(), 1)
        y_pred = K.clip(y_pred, K.epsilon(), 1)

        # compute two types of losses, for all the data points
        # (1) compute CCE loss for every data point
        _loss_CCE = -K.sum(y_true * K.log(y_pred), axis=-1)

        # (2) compute lq_loss for every data point
        _tmp = y_pred * y_true
        _loss_tmp = K.max(_tmp, axis=-1)
        # compute the Lq loss between the one-hot encoded label and the predictions
        _loss_q = (1 - (_loss_tmp + 10 ** (-8)) ** _q) / _q

        # decide which loss to take for each datapoint
        _mask_noisy = K.cast(_y_true_flag, 'float32')                   # only allows patches from noisy set
        _mask_clean = K.cast(K.equal(_y_true_flag, False), 'float32')   # only allows patches from clean set

        # points coming from clean set contribute with CCE loss
        # points coming from noisy set contribute with lq_loss
        _loss_final = _loss_CCE * _mask_clean + _loss_q * _mask_noisy

        return _loss_final
    return lq_loss_origin_core

def crossentropy_max_origin_wrap(_m):
    def crossentropy_max_origin_core(y_true, y_pred):

        # hyper param
        print(_m)

        # 1) determine the origin of the patch, as a boolean vector y_true_flag
        # (True = patch from noisy subset)
        _y_true_flag = K.greater(K.sum(y_true, axis=-1), 90)

        # 2) convert the input y_true (with flags inside) into a valid y_true one-hot-vector format
        # attenuating factor for data points that need it (those that came with a one-hot of 100)
        _mask_reduce = K.cast(_y_true_flag, 'float32') * 0.01

        # identity factor for standard one-hot vectors
        _mask_keep = K.cast(K.equal(_y_true_flag, False), 'float32')

        # combine 2 masks
        _mask = _mask_reduce + _mask_keep

        _y_true_shape = K.shape(y_true)
        _mask = K.reshape(_mask, (_y_true_shape[0], 1))

        # applying mask to have a valid y_true that we can use as always
        y_true = y_true * _mask

        y_true = K.clip(y_true, K.epsilon(), 1)
        y_pred = K.clip(y_pred, K.epsilon(), 1)

        # compute loss for every data point
        _loss = -K.sum(y_true * K.log(y_pred), axis=-1)

        # threshold m
        t_m = K.max(_loss) * _m

        _mask_m = 1 - (K.cast(K.greater(_loss, t_m), 'float32') * K.cast(_y_true_flag, 'float32'))
        _loss = _loss * _mask_m

        return _loss
    return crossentropy_max_origin_core

def crossentropy_outlier_origin_wrap(_l):
    def crossentropy_outlier_origin_core(y_true, y_pred):

        # hyper param
        print(_l)

        # 1) determine the origin of the patch, as a boolean vector y_true_flag
        # (True = patch from noisy subset)
        _y_true_flag = K.greater(K.sum(y_true, axis=-1), 90)

        # 2) convert the input y_true (with flags inside) into a valid y_true one-hot-vector format
        # attenuating factor for data points that need it (those that came with a one-hot of 100)
        _mask_reduce = K.cast(_y_true_flag, 'float32') * 0.01

        # identity factor for standard one-hot vectors
        _mask_keep = K.cast(K.equal(_y_true_flag, False), 'float32')

        # combine 2 masks
        _mask = _mask_reduce + _mask_keep

        _y_true_shape = K.shape(y_true)
        _mask = K.reshape(_mask, (_y_true_shape[0], 1))

        # applying mask to have a valid y_true that we can use as always
        y_true = y_true * _mask

        y_true = K.clip(y_true, K.epsilon(), 1)
        y_pred = K.clip(y_pred, K.epsilon(), 1)

        # compute loss for every data point
        _loss = -K.sum(y_true * K.log(y_pred), axis=-1)

        def _get_real_median(_v):
            """
            given a tensor with shape (batch_size,), compute and return the median

            :param v:
            :return:
            """
            _val = tf.nn.top_k(_v, 33).values
            return 0.5 * (_val[-1] + _val[-2])

        _mean_loss, _var_loss = tf.nn.moments(_loss, axes=[0])
        _median_loss = _get_real_median(_loss)
        _std_loss = tf.sqrt(_var_loss)

        # threshold
        t_l = _median_loss + _l*_std_loss

        _mask_l = 1 - (K.cast(K.greater(_loss, t_l), 'float32') * K.cast(_y_true_flag, 'float32'))
        _loss = _loss * _mask_l

        return _loss
    return crossentropy_outlier_origin_core