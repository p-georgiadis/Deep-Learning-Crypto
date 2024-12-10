import tensorflow as tf

# Global variable to control direction weighting. Initially set high, then reduced via callback.
DIRECTION_WEIGHT_FACTOR = tf.Variable(3.0, trainable=False, dtype=tf.float32)

def directional_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true_cur = y_true[:,0]
    y_true_prev = y_true[:,1]
    y_pred_cur = tf.reshape(y_pred, [-1])
    direction_real = tf.sign(y_true_cur - y_true_prev)
    direction_pred = tf.sign(y_pred_cur - y_true_prev)
    correct = tf.cast(tf.equal(direction_pred, direction_real), tf.float32)
    return tf.reduce_mean(correct)

def di_mse_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    DI-MSE loss that uses DIRECTION_WEIGHT_FACTOR to penalize wrong directions more in early training.
    """
    y_true_cur = y_true[:,0]
    y_true_prev = y_true[:,1]
    pred = tf.squeeze(y_pred, axis=1)
    N = tf.cast(tf.shape(y_true_cur)[0], tf.float32)

    direction_real = tf.sign(y_true_cur - y_true_prev)
    direction_pred = tf.sign(pred - y_true_prev)

    correct_mask = tf.equal(direction_pred, direction_real)
    correct_mask_f = tf.cast(correct_mask, tf.float32)

    up_mask = direction_real > 0
    down_mask = tf.logical_not(up_mask)

    ups = tf.reduce_sum(tf.cast(up_mask, tf.float32))
    downs = tf.reduce_sum(tf.cast(down_mask, tf.float32))
    total_dir = ups + downs

    diff = pred - y_true_cur
    sq_err = diff * diff

    def fallback_mse():
        return tf.reduce_mean(sq_err)

    def compute_di_mse():
        eps = tf.constant(1e-7, tf.float32)
        W_up = tf.where(tf.greater(total_dir, eps), ups/(total_dir+eps), 1.0)
        W_down = tf.where(tf.greater(total_dir, eps), downs/(total_dir+eps), 1.0)

        upward_correct = tf.logical_and(correct_mask, up_mask)
        downward_correct = tf.logical_and(correct_mask, down_mask)

        W_up_mask = tf.cast(upward_correct, tf.float32)*W_up
        W_down_mask = tf.cast(downward_correct, tf.float32)*W_down

        W = W_up_mask + W_down_mask
        correct_loss = W * sq_err

        # Wrong directions: penalize using DIRECTION_WEIGHT_FACTOR
        wrong_mask = tf.logical_not(correct_mask)
        wrong_loss = tf.cast(wrong_mask, tf.float32)*DIRECTION_WEIGHT_FACTOR

        di_mse = (tf.reduce_sum(correct_loss) + tf.reduce_sum(wrong_loss))/N
        return di_mse

    return tf.cond(tf.less(total_dir, 1e-7), fallback_mse, compute_di_mse)
