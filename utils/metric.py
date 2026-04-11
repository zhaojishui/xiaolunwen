
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from sklearn.metrics import average_precision_score
import torch
import torch.nn.functional as F
import torch.utils.data
__all__ = ['MetricsTop']


class MetricsTop():
    def __init__(self, train_mode):
        if train_mode == "regression":
            self.metrics_dict = {
                'MOSI': self.__eval_mosi_regression,
                'MOSEI': self.__eval_mosei_regression,
                'SIMS': self.__eval_sims_regression
            }
        else:
            raise AssertionError

    def __multiclass_acc(self, y_pred, y_true):
        """
        Compute the multiclass accuracy w.r.t. groundtruth
        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))  # np.sum可以统计出有多少个元素是True。

    def __eval_mosei_regression(self, y_pred, y_true, exclude_zero=False):
        test_preds = y_pred.view(-1).cpu().detach().numpy()  # view(-1)，把y_pred这样的一个二维数组（总预测个数，1）展成一行。
        test_truth = y_true.view(-1).cpu().detach().numpy()

        test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)  # 小于-3的数变成-3，大于3的数变成3，中间的数保持不变。
        test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
        test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)  # -2到2，正好是5个数，对应5分类准确率。
        test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)


        mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths，没有指定维度，所以展平成一行求平均值。
        corr = np.corrcoef(test_preds, test_truth)[0][1]  # 返回相关系数矩阵，形状是（2，2），用[0][1]取到相关性，反正矩阵是一个对称矩阵。
        mult_a7 = self.__multiclass_acc(test_preds_a7, test_truth_a7)  # 这三个都是用预测正确的个数除以总的预测数。
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)


        non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])  # 取出test_truth不是0的索引
        non_zeros_binary_truth = (test_truth[non_zeros] > 0)  # 得到的是一行bool值，test_truth>0的是true，否则是false
        non_zeros_binary_preds = (test_preds[non_zeros] > 0)

        non_zeros_acc2 = accuracy_score(non_zeros_binary_preds, non_zeros_binary_truth)  # 计算2分类准确率，0的不考虑。
        non_zeros_f1_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average='weighted')  # 计算f1

        binary_truth = (test_truth >= 0)  # 大于等于0是true，小于0就是false
        binary_preds = (test_preds >= 0)
        acc2 = accuracy_score(binary_preds, binary_truth)
        f_score = f1_score(binary_truth, binary_preds, average='weighted')  # 这里0的考虑到了。

        eval_results = {
            "Has0_acc_2": round(acc2, 4),  # acc2保留4位小数
            "Has0_F1_score": round(f_score, 4),
            "Non0_acc_2": round(non_zeros_acc2, 4),
            "Non0_F1_score": round(non_zeros_f1_score, 4),
            "Mult_acc_5": round(mult_a5, 4),
            "Mult_acc_7": round(mult_a7, 4),
            "MAE": round(mae, 4),
            "Corr": round(corr,4)
        }
        return eval_results  # 返回一个字典。

    def __eval_mosi_regression(self, y_pred, y_true):
        return self.__eval_mosei_regression(y_pred, y_true)

    def __eval_sims_regression(self, y_pred, y_true):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()
        test_preds = np.clip(test_preds, a_min=-1., a_max=1.)  # sims数据集的回归分类标签范围就是[-1,1]
        test_truth = np.clip(test_truth, a_min=-1., a_max=1.)

        # two classes{[-1.0, 0.0], (0.0, 1.0]}
        ms_2 = [-1.01, 0.0, 1.01]
        test_preds_a2 = test_preds.copy()
        test_truth_a2 = test_truth.copy()
        for i in range(2):  # i取0，1.把-1<test_preds <=0的地方赋值成0，把0<test_preds <=1的地方赋值成1
            test_preds_a2[np.logical_and(test_preds > ms_2[i], test_preds <= ms_2[i + 1])] = i
        for i in range(2):  # 0表示负样本，1表示正样本。
            test_truth_a2[np.logical_and(test_truth > ms_2[i], test_truth <= ms_2[i + 1])] = i

        # three classes{[-1.0, -0.1], (-0.1, 0.1], (0.1, 1.0]}
        ms_3 = [-1.01, -0.1, 0.1, 1.01]
        test_preds_a3 = test_preds.copy()
        test_truth_a3 = test_truth.copy()
        for i in range(3):  # i取0，1，2，这是三分类。
            test_preds_a3[np.logical_and(test_preds > ms_3[i], test_preds <= ms_3[i + 1])] = i
        for i in range(3):
            test_truth_a3[np.logical_and(test_truth > ms_3[i], test_truth <= ms_3[i + 1])] = i

        # five classes{[-1.0, -0.7], (-0.7, -0.1], (-0.1, 0.1], (0.1, 0.7], (0.7, 1.0]}
        ms_5 = [-1.01, -0.7, -0.1, 0.1, 0.7, 1.01]
        test_preds_a5 = test_preds.copy()
        test_truth_a5 = test_truth.copy()
        for i in range(5):  # 五分类。
            test_preds_a5[np.logical_and(test_preds > ms_5[i], test_preds <= ms_5[i + 1])] = i
        for i in range(5):
            test_truth_a5[np.logical_and(test_truth > ms_5[i], test_truth <= ms_5[i + 1])] = i

        mae = np.mean(
            np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths#MAE，平均绝对误差计算公式。
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a2 = self.__multiclass_acc(test_preds_a2, test_truth_a2)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        f_score = f1_score(test_truth_a2, test_preds_a2, average='weighted')

        eval_results = {
            "Mult_acc_2": mult_a2,
            "Mult_acc_3": mult_a3,
            "Mult_acc_5": mult_a5,
            "F1_score": f_score,
            "MAE": mae,
            "Corr": corr,  # Correlation Coefficient
        }
        return eval_results


    def getMetics(self, datasetName):
        return self.metrics_dict[datasetName.upper()]

def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key])
    return dst_str


def to_numpy(array):
  if isinstance(array, np.ndarray):
    return array
  if isinstance(array, torch.autograd.Variable):
    array = array.data
  if array.is_cuda:
    array = array.cpu()

  return array.numpy()


def squeeze(array):
  if not isinstance(array, list) or len(array) > 1:
    return array
  else:  # len(array) == 1:
    return array[0]


def unsqueeze(array):
  if isinstance(array, list):
    return array
  else:
    return [array]


def is_due(*args):
  """Determines whether to perform an action or not, depending on the epoch.
     Used for logging, saving, learning rate decay, etc.

  Args:
    *args: epoch, due_at (due at epoch due_at) epoch, num_epochs,
          due_every (due every due_every epochs)
          step, due_every (due every due_every steps)
  Returns:
    due: boolean: perform action or not
  """
  if len(args) == 2 and isinstance(args[1], list):
    epoch, due_at = args
    due = epoch in due_at
  elif len(args) == 3:
    epoch, num_epochs, due_every = args
    due = (due_every >= 0) and (epoch % due_every == 0 or epoch == num_epochs)
  else:
    step, due_every = args
    due = (due_every > 0) and (step % due_every == 0)

  return due


def softmax(w, t=1.0, axis=None):
  w = np.array(w) / t#t：温度参数（控制“平滑程度”）
  e = np.exp(w - np.amax(w, axis=axis, keepdims=True))#技巧：减去最大值，防止np.exp(1000)  # 会溢出（inf）这种情况发生。
  dist = e / np.sum(e, axis=axis, keepdims=True)
  return dist


def min_cosine(student, teacher, option, weights=None):
  cosine = torch.nn.CosineEmbeddingLoss()
  dists = cosine(student, teacher.detach(), torch.tensor([-1]).cuda())
  if weights is None:
    dist = dists.mean()
  else:
    dist = (dists * weights).mean()

  return dist



def distance_metric(student, teacher, option, weights=None):
  """Distance metric to calculate the imitation loss.

  Args:
    student: batch_size x n_classes
    teacher: batch_size x n_classes
    option: one of [cosine, l2, l2, kl]
    weights: batch_size or float

  Returns:
    The computed distance metric.
  """
  if option == 'cosine':
    dists = 1 - F.cosine_similarity(student, teacher.detach(), dim=1)
    # dists = 1 - F.cosine_similarity(student, teacher, dim=1)
  elif option == 'l2':
    dists = (student-teacher.detach()).pow(2).sum(1)
  elif option == 'l1':
    dists = torch.abs(student-teacher.detach()).sum(1)
  elif option == 'kl':
    assert weights is None
    T = 8
    # averaged for each minibatch
    dist = F.kl_div(
        F.log_softmax(student / T), F.softmax(teacher.detach() / T)) * (
            T * T)
    return dist
  else:
    raise NotImplementedError

  if weights is None:
    dist = dists.mean()
  else:
    dist = (dists * weights).mean()

  return dist


def get_segments(input, timestep):
  """Split entire input into segments of length timestep.

  Args:
    input: 1 x total_length x n_frames x ...
    timestep: the timestamp.

  Returns:
    input: concatenated video segments
    start_indices: indices of the segments
  """
  assert input.size(0) == 1, 'Test time, batch_size must be 1'

  input.squeeze_(dim=0)
  # Find overlapping segments
  length = input.size()[0]
  step = timestep // 2
  num_segments = (length - timestep) // step + 1
  start_indices = (np.arange(num_segments) * step).tolist()
  if length % step > 0:
    start_indices.append(length - timestep)

  # Get the segments
  segments = []
  for s in start_indices:
    segment = input[s: (s + timestep)].unsqueeze(0)
    segments.append(segment)
  input = torch.cat(segments, dim=0)
  return input, start_indices

def get_stats(logit, label):
  '''
  Calculate the accuracy.
  '''
  logit = to_numpy(logit)
  label = to_numpy(label)

  pred = np.argmax(logit, 1)
  acc = np.sum(pred == label)/label.shape[0]

  return acc, pred, label


def get_stats_detection(logit, label, n_classes=52):
  '''
  Calculate the accuracy and average precisions.
  '''
  logit = to_numpy(logit)
  label = to_numpy(label)
  scores = softmax(logit, axis=1)

  pred = np.argmax(logit, 1)
  length = label.shape[0]
  acc = np.sum(pred == label)/length

  keep_bg = label == 0
  acc_bg = np.sum(pred[keep_bg] == label[keep_bg])/label[keep_bg].shape[0]
  ratio_bg = np.sum(keep_bg)/length

  keep_action = label != 0
  acc_action = np.sum(
      pred[keep_action] == label[keep_action]) / label[keep_action].shape[0]

  # Average precision
  y_true = np.zeros((len(label), n_classes))
  y_true[np.arange(len(label)), label] = 1
  acc = np.sum(pred == label)/label.shape[0]
  aps = average_precision_score(y_true, scores, average=None)
  aps = list(filter(lambda x: not np.isnan(x), aps))
  ap = np.mean(aps)

  return ap, acc, acc_bg, acc_action, ratio_bg, pred, label


def info(text):
  print('\033[94m' + text + '\033[0m')


def warn(text):
  print('\033[93m' + text + '\033[0m')


def err(text):
  print('\033[91m' + text + '\033[0m')
