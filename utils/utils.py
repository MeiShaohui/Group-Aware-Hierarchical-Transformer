import numpy as np
import itertools
from sklearn.metrics import confusion_matrix


def split_info_print(train_gt, val_gt, test_gt, labels):
    train_class_num = []
    val_class_num = []
    test_class_num = []
    for i in range(len(labels)):
        train_class_num.append(np.sum(train_gt == i))
        val_class_num.append(np.sum(val_gt == i))
        test_class_num.append(np.sum(test_gt == i))
    print("class", "train", "val", "test")
    for i in range(len(labels)):
        print(labels[i], train_class_num[i], val_class_num[i], test_class_num[i])


def sliding_window(image, step=1, window_size=(20, 20), with_data=True):
    """Sliding window generator over an input image.
    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
        with_data (optional): bool set to True to return both the data and the corner indices
    Yields:
        ([data], x, y, w, h) where x and y are the top-left corner of the
        window, (w,h) the window size
    """
    # slide a window across the image
    w, h = window_size
    W, H = image.shape[:2]
    for x in range(0, W - w + step, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + step, step):
            if y + h > H:
                y = H - h
            if with_data:
                yield image[x:x + w, y:y + h], x, y, w, h
            else:
                yield x, y, w, h


def count_sliding_window(image, step=1, window_size=(20, 20)):
    """ Count the number of windows in an image.
    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral, ...
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
    Returns:
        int number of windows
    """
    sw = sliding_window(image, step, window_size, with_data=False)
    return sum(1 for _ in sw)


def grouper(n, iterable):
    """ Browse an iterable by grouping n elements by n elements.
    Args:
        n: int, size of the groups
        iterable: the iterable to Browse
    Yields:
        chunk of n elements from the iterable
    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def metrics(prediction, target, n_classes=None):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, accuracy by class, confusion matrix
    """
    ignored_mask = np.zeros(target.shape[:2], dtype=np.bool)
    ignored_mask[target < 0] = True
    ignored_mask = ~ignored_mask
    target = target[ignored_mask]
    prediction = prediction[ignored_mask]
    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes

    cm = confusion_matrix(
        target,
        prediction,
        labels=range(n_classes))

    results["Confusion matrix"] = cm

    # Compute global accuracy
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy /= float(total)

    results["Accuracy"] = accuracy * 100.0

    # Compute accuracy of each class
    class_acc = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            acc = cm[i, i] / np.sum(cm[i, :])
        except ZeroDivisionError:
            acc = 0.
        class_acc[i] = acc

    results["class acc"] = class_acc * 100.0
    results['AA'] = np.mean(class_acc) * 100.0
    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
         float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa * 100.0

    return results


def show_results(results, label_values=None, agregated=False):
    text = ""

    if agregated:
        accuracies = [r["Accuracy"] for r in results]
        aa = [r['AA'] for r in results]
        kappas = [r["Kappa"] for r in results]
        class_acc = [r["class acc"] for r in results]

        class_acc_mean = np.mean(class_acc, axis=0)
        class_acc_std = np.std(class_acc, axis=0)
        cm = np.mean([r["Confusion matrix"] for r in results], axis=0)
        text += "Agregated results :\n"
    else:
        cm = results["Confusion matrix"]
        accuracy = results["Accuracy"]
        aa = results['AA']
        classacc = results["class acc"]
        kappa = results["Kappa"]

    text += "Confusion matrix :\n"
    text += str(cm)
    text += "---\n"

    if agregated:
        text += ("Accuracy: {:.02f} +- {:.02f}\n".format(np.mean(accuracies),
                                                         np.std(accuracies)))
    else:
        text += "Accuracy : {:.02f}%\n".format(accuracy)
    text += "---\n"

    text += "class acc :\n"
    if agregated:
        for label, score, std in zip(label_values, class_acc_mean,
                                     class_acc_std):
            text += "\t{}: {:.02f} +- {:.02f}\n".format(label, score, std)
    else:
        for label, score in zip(label_values, classacc):
            text += "\t{}: {:.02f}\n".format(label, score)
    text += "---\n"

    if agregated:
        text += ("AA: {:.02f} +- {:.02f}\n".format(np.mean(aa),
                                                   np.std(aa)))
        text += ("Kappa: {:.02f} +- {:.02f}\n".format(np.mean(kappas),
                                                      np.std(kappas)))
    else:
        text += "AA: {:.02f}%\n".format(aa)
        text += "Kappa: {:.02f}\n".format(kappa)

    print(text)


