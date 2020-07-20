# metric for training CNN
import tensorflow as tf
from tensorflow.keras import metrics
from sklearn.metrics import roc_auc_score

class customized_metrics_bacc(metrics.Metric):
    @tf.function
    def call(self, y_true_local, y_pred_local, fp=metrics.FalsePositives(), fn=metrics.FalseNegatives(),
             tp=metrics.TruePositives(), tn=metrics.TrueNegatives()):
        return 0.5 * tn / (tn + fn) + (tp / (tp + fp))
