import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import sampler


def truncated_normal(mean=0.0, stddev=1.0, m=1):
    '''
    The generated values follow a normal distribution with specified 
    mean and standard deviation, except that values whose magnitude is 
    more than 2 standard deviations from the mean are dropped and 
    re-picked. Returns a vector of length m
    '''
    samples = []
    for i in range(m):
        while True:
            sample = np.random.normal(mean, stddev)
            if np.abs(sample) <= 2 * stddev:
                break
        samples.append(sample)
    assert len(samples) == m, "something wrong"
    if m == 1:
        return samples[0]
    else:
        return np.array(samples)


# --- PyTorch helpers ---

def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def pred_batch(x, model):
    """
    batch prediction helper
    """
    y_pred = np.argmax(model(to_var(x)).data.cpu().numpy(), axis=1)
    return torch.from_numpy(y_pred)


def test(model, loader, blackbox=False, hold_out_size=None):
    """
    Check model accuracy on model based on loader (train or test)
    """
    model.eval()

    num_correct, num_samples = 0, len(loader.dataset)

    if blackbox:
        num_samples -= hold_out_size

    for x, y in loader:
        x = x.cuda()
        x_var = to_var(x, volatile=True)
        scores,attention = model(x_var)
        #scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct)/float(num_samples)
    print('Got %d/%d correct (%.2f%%) on the clean data' 
        % (num_correct, num_samples, 100 * acc))

    return acc


def attack_over_test_data(model, adversary, param, loader_test, oracle=None):
    """
    Given target model computes accuracy on perturbed data
    """
    total_correct = 0
    total_samples = len(loader_test.dataset)

    # For black-box
    if oracle is not None:
        total_samples -= param['hold_out_size']

    for t, (X, y) in enumerate(loader_test):
        y_pred = pred_batch(X, model)
        X_adv = adversary.perturb(X.numpy(), y_pred)
        X_adv = torch.from_numpy(X_adv)

        if oracle is not None:
            y_pred_adv = pred_batch(X_adv, oracle)
        else:
            y_pred_adv = pred_batch(X_adv, model)
        
        total_correct += (y_pred_adv.numpy() == y.numpy()).sum()

    acc = total_correct/total_samples

    print('Got %d/%d correct (%.2f%%) on the perturbed data' 
        % (total_correct, total_samples, 100 * acc))

    return acc


def batch_indices(batch_nb, data_length, batch_size):
    """
    This helper function computes a batch start and end index
    :param batch_nb: the batch number
    :param data_length: the total length of the data being parsed by batches
    :param batch_size: the number of inputs in each batch
    :return: pair of (start, end) indices
    """
    # Batch start and end index
    start = int(batch_nb * batch_size)
    end = int((batch_nb + 1) * batch_size)

    # When there are not enough inputs left, we reuse some to complete the
    # batch
    if end > data_length:
        shift = end - data_length
        start -= shift
        end -= shift

    return start, end
