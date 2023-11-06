from io import BytesIO

import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2


def compute_frr_far(outputs, targets):
    targets = np.squeeze(targets)
    outputs = np.squeeze(outputs)
    ind = np.argsort(outputs)
    targets = targets[ind]
    threshold = outputs[ind]
    frr = np.cumsum(targets) / np.sum(targets)
    far = 1 - np.cumsum(1 - targets) / np.sum(1 - targets)
    return frr, far, threshold


def compute_eer(outputs, targets):
    targets = targets.astype(bool)
    if targets.all() or (~targets).all():
        return (float('nan'), float('nan'))
    frr, far, _ = compute_frr_far(outputs, targets)
    i_eer = np.argmin(np.abs(frr - far))
    eer_value = 0.5 * (frr[i_eer] + far[i_eer])
    ind = np.argsort(outputs)
    return eer_value, outputs[ind[i_eer]]


def compute_accuracy_at_threshold(outputs, targets, threshold=0.5):
    outputs = (outputs > threshold).astype(int)
    return accuracy_score(outputs, targets)


def plot_frr_far(thresholds, frr, far, eer, eer_th):
    plt.figure()
    plt.xlabel('Threshold', fontsize = 14)
    plt.ylabel('Error Rate', fontsize = 14)
    plt.title('FRR/FAR', fontsize = 14)
    _ = plt.plot(thresholds, far, 'r--', linewidth = 2, label='FAR')
    _ = plt.plot(thresholds, frr, 'g--', linewidth = 2, label='FRR')
    _ = plt.plot(eer_th, eer, 'bo', label='EER')
    plt.yscale('log')
    plt.legend()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img = cv2.imdecode(np.frombuffer(buffer.read(), np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    plt.close()
    return img