import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score


def compute_accuracy(y_true, y_preds, lengths):
    assert len(y_true) == len(y_preds)
    accuracies = []
    for cur_y_true, cur_y_pred, cur_len in zip(y_true, y_preds, lengths):
        cur_y_true = cur_y_true[: int(cur_len.item())]
        cur_y_pred = cur_y_pred[: int(cur_len.item())]
        accuracy = accuracy_score(cur_y_true, cur_y_pred)
        accuracies.append(accuracy)
    return np.array(accuracies)

def compute_statistics(args, data_split, al_scores, accuracy_scores, step_num=None):
    all_ids = al_scores.keys()
    confidence = np.array([al_scores[id] for id in all_ids])
    accuracy = np.array([accuracy_scores[id] for id in all_ids])
    plot_confidence_vs_accuracy(args, data_split, confidence, accuracy, step_num)
    save_confidence_vs_accuracy(args, data_split, confidence, accuracy, step_num)
    sentence_calibration_errors(args, data_split, confidence, accuracy, step_num)

def get_file_to_save_name(args, data_split, ending_name, step_num=None):
    task_for_scoring_str = '_task_for_scoring_' + args.task_for_scoring if args.task_for_scoring is not None else ''
    sub_folder = args.al_selection_method
    if not os.path.isdir(os.path.join(args.output_dir, sub_folder)):
        os.mkdir(os.path.join(args.output_dir, sub_folder))
    if step_num is not None:
        sub_folder = os.path.join(sub_folder, str(step_num))
        if not os.path.isdir(os.path.join(args.output_dir, sub_folder)):
            os.mkdir(os.path.join(args.output_dir, sub_folder))
    file_to_save = os.path.join(args.output_dir, sub_folder, args.src_domain + '_' + data_split + '_' +
                                args.al_selection_method + task_for_scoring_str + '_' + '_'.join(args.tasks) + '_' + ending_name)
    return file_to_save

def plot_confidence_vs_accuracy(args, data_split, confidence, accuracy, step_num=None):
    # Calculate the point density
    xy = np.vstack([confidence, accuracy])
    try:
        z = gaussian_kde(xy)(xy) / 100.
    except:
        z = np.ones_like(confidence)
    count_dict = {}
    for tup in zip(confidence, accuracy):
        if tup in count_dict:
            count_dict[tup] += 1
        else:
            count_dict[tup] = 1
    len_x = len(confidence)
    for k, v in count_dict.items():
        if v > 10:
            v += 500
        count_dict[k] = v / len_x
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = confidence[idx], accuracy[idx], z[idx]

    fig, ax_joint = plt.subplots()

    cmap = cm.inferno
    sc = ax_joint.scatter(x, y, c=z, alpha=0.3, s=30, edgecolors='k', cmap=cmap)

    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(x, y)  # perform linear regression
    y_pred = linear_regressor.predict(x)  # make predictions
    # ax_joint.plot(x, y_pred, color='black')
    ax_joint.set_ylim([0,1])
    r_square = r2_score(y, y_pred)
    ax_joint.set_title("R squared = %.2f" % r_square, fontsize=20, weight="bold")
    # Set labels on joint
    ax_joint.set_ylim((np.min(y), np.max(y)))
    ax_joint.set_xlabel('Confidence', fontsize=16, weight="bold")
    ax_joint.set_ylabel('Accuracy', fontsize=16, weight="bold")
    plt.locator_params(axis='y', nbins=10)
    plt.locator_params(axis='x', nbins=5)
    plt.xticks(fontsize=12, weight="bold")
    plt.yticks(fontsize=12, weight="bold")
    plt.colorbar(sc)

    file_to_save = get_file_to_save_name(args, data_split, 'confidence_vs_accuracy_sentences.pdf', step_num)
    fig.savefig(file_to_save)
    plt.close()

    file_to_save_r_square = get_file_to_save_name(args, data_split, 'confidence_vs_accuracy_sentences_rsquare.csv', step_num)
    with open(file_to_save_r_square, 'w') as f:
        f.write("R square, %s\n" % r_square)

def save_confidence_vs_accuracy(args, data_split, confidence, accuracy, step_num=None):
    file_to_save = get_file_to_save_name(args, data_split, 'confidence_vs_accuracy_sentences.csv', step_num)
    with open(file_to_save, 'w') as f:
        f.write("confidence, accuracy\n")
        for c, a in zip(confidence, accuracy):
            f.write("%s, %s\n" % (c, a))

def save_confidence_per_task(args, data_split, sample_to_confidence, step_num=None):
    file_to_save = get_file_to_save_name(args, data_split, 'confidence_per_task.csv', step_num)
    with open(file_to_save, 'w') as f:
        f.write('sample_id,' + ','.join(args.tasks) + '\n')
        for sample_id, confidences in sample_to_confidence.items():
            if not isinstance(confidences, list):
                confidences = [confidences]
            f.write(str(sample_id) + ',' + ','.join([str(c) for c in confidences]) + '\n')

def save_sample_ids(al_iter_num, args, sample_ids):
    sample_ids_file = os.path.join(args.output_dir,
                                   args.src_domain + '_sample_ids_' + args.al_selection_method + '_' + '_'.join(
                                       args.tasks) + '_iter_' + str(al_iter_num) + '.csv')
    with open(sample_ids_file, 'w', newline='') as myfile:
        myfile.write("\n".join([str(id) for id in sample_ids]))


def sentence_calibration_errors(args, data_split, confidence, accuracy, step_num=None):
    """
    Computes several metrics used to measure calibration error:
        - Expected Calibration Error (ECE): \sum_k |acc(k) - conf(k)| / n
        - Maximum Calibration Error (MCE): max_k |acc(k) - conf(k)|
        - Overconfidence Calibration Error (OCE): \sum_k conf(k) * max(conf(k) - acc(k), 0) / n
    """

    delta = abs(accuracy - confidence)
    expected_error = delta.mean()
    max_error = max(delta)
    oc_expected_error = (confidence * np.where(confidence > accuracy, confidence - accuracy, 0)).mean()

    file_to_save = get_file_to_save_name(args, data_split, 'calibration_errors.csv', step_num)
    with open(file_to_save, 'w') as f:
        f.write("expected_error, %s\n" % expected_error)
        f.write("max_error, %s\n" % max_error)
        f.write("overconfident_expected_error, %s\n" % oc_expected_error)
