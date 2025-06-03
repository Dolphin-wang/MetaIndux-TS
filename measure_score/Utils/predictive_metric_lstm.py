# Necessary Packages
import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf1
tf.compat.v1.disable_eager_execution()
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from measure_score.Utils.metric_utils import extract_time
from tqdm.auto import tqdm

def predictive_score_metrics_lstm(ori_data, generated_data):
    """Report the performance of Post-hoc LSTM one-step ahead prediction.
    
    Args:
        - ori_data: original data
        - generated_data: generated synthetic data
        
    Returns:
        - predictive_score_mae: MAE of the predictions on the original data
        - predictive_score_rmse: RMSE of the predictions on the original data
        - Y_mb: True labels
        - y_pred: Predicted labels
    """
    # Initialization on the Graph
    tf1.reset_default_graph()

    # Basic Parameters
    no, seq_len, dim = ori_data.shape
    
    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])
    
    ## Build a post-hoc LSTM predictive network 
    # Network parameters
    hidden_dim = int(dim * 2)
    iterations = 5000
    batch_size = 128

    # Input placeholders
    X = tf1.placeholder(tf.float32, [None, max_seq_len, dim - 1], name="myinput_x")
    T = tf1.placeholder(tf.int32, [None], name="myinput_t")
    Y = tf1.placeholder(tf.float32, [None, 1, 1], name="myinput_y")
    
    # Predictor function
    def predictor(x, t):
        """Simple predictor function.
        
        Args:
          - x: time-series data
          - t: time information
          
        Returns:
          - y_hat: prediction
          - p_vars: predictor variables
        """
        with tf1.variable_scope("predictor", reuse=tf1.AUTO_REUSE) as vs:
            p_cell = tf1.nn.rnn_cell.LSTMCell(num_units=hidden_dim, activation=tf.nn.tanh, name='p_cell')
            p_outputs, _ = tf1.nn.dynamic_rnn(p_cell, x, dtype=tf.float32, sequence_length=t)
            
            # Get the last time step's output
            last_output_index = tf.stack([tf.range(tf.shape(p_outputs)[0]), t - 1], axis=1)
            last_output = tf.gather_nd(p_outputs, last_output_index)
            
            # Predict the last feature value
            y_hat_logit = tf1.layers.dense(last_output, 1, activation=None)
            y_hat = tf.expand_dims(y_hat_logit, axis=-1)  # Expand dimensions to (None, 1, 1)
            p_vars = [v for v in tf1.all_variables() if v.name.startswith(vs.name)]

        return y_hat, p_vars

    y_pred, p_vars = predictor(X, T)

    # Loss for the predictor
    p_loss = tf1.losses.absolute_difference(Y, y_pred)
    
    # Optimizer
    p_solver = tf1.train.AdamOptimizer().minimize(p_loss, var_list=p_vars)

    ## Training    
    # Session start
    sess = tf1.Session()
    sess.run(tf1.global_variables_initializer())

    # Training using Synthetic dataset
    for itt in tqdm(range(iterations), desc='training', total=iterations):
        # Set mini-batch
        idx = np.random.permutation(len(generated_data))
        train_idx = idx[:batch_size]     
        X_mb = list(generated_data[i][:, :(dim - 1)] for i in train_idx)
        T_mb = list(generated_time[i] - 1 for i in train_idx)
        Y_mb = list(np.reshape(generated_data[i][-1, -1], [1, 1]) for i in train_idx)        
              
        # Train predictor
        _, step_p_loss = sess.run([p_solver, p_loss], feed_dict={X: X_mb, T: T_mb, Y: Y_mb})
    
    ## Test the trained model on the original data
    idx = np.random.permutation(len(ori_data))
    train_idx = idx[:no]

    X_mb = list(ori_data[i][:, :(dim - 1)] for i in train_idx)
    T_mb = list(ori_time[i] - 1 for i in train_idx)
    Y_mb = list(np.reshape(ori_data[i][-1, -1], [1, 1]) for i in train_idx)
    
    # Prediction
    pred_Y_curr = sess.run(y_pred, feed_dict={X: X_mb, T: T_mb})
    
    # Compute the performance in terms of MAE
    MAE_temp = 0
    for i in range(no):
        MAE_temp = MAE_temp + mean_absolute_error(Y_mb[i], pred_Y_curr[i, :, :])
    
    predictive_score_mae = MAE_temp / no

    RMSE_temp = 0
    for i in range(no):
        RMSE_temp += np.sqrt(mean_squared_error(Y_mb[i], pred_Y_curr[i, :, :]))  
    predictive_score_rmse = RMSE_temp / no
    
    return predictive_score_mae, predictive_score_rmse

