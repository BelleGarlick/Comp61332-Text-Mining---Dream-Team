import numpy as np


"""
This module deals with analysing the predicted results of a classifier.

Usage:
    clf.fit(X, y)
    analysis = roc.analyse(val_y, clf.predict(val_X))
"""


def analyse(true_labels: list, predicted_labels: list) -> dict:
    """
    Analyise accuracy and F1 score of the predicted labels.

    This function will analyise the difference between the true labels and predicted labels. This function creates a
    confusion matrix and runs analysis over the confusion matrix such as calculating the recall, precision and F1 score.

    Args:
        true_labels: The true labels of the data.
        predicted_labels: The predicted labels as given by the classifier

    Returns:
        A dictionary of analytic data structured as such:
            {
                "accuracy": The accuracy of the classifier - how many labels were correctly predicted.
                "f1": The F1 score of the classifier.
                "precision": The precision in results.
                "recall": The recal of the classifier.
                "confusion": The confusion matrix generated.
            }
    """
    # Map classifications to indexes
    classification_indexes = __create_classification_indexes(true_labels, predicted_labels)

    # Build confusion matrix
    conf_matrix = __build_conf_matrix(classification_indexes, true_labels, predicted_labels)

    # Calc Tps, Fps, Tns, Fns
    tp, fp, fn, tn = __decompose_conf_matrix(conf_matrix, len(true_labels))

    # TODO Talk in report about why we did micro averaging. There is a large class imabalanace so we needed a score
    # Calculate micro averaging scores
    micro_average_precision = np.sum(tp) / (np.sum(tp) + np.sum(fp))
    micro_average_recall = np.sum(tp) / (np.sum(tp) + np.sum(fn))
    f1 = (2 * micro_average_recall * micro_average_precision) / (micro_average_recall + micro_average_precision)

    accuracy = np.sum([
        int(predicted_labels[i] == true_labels[i])
        for i in range(len(predicted_labels))
    ]) / len(predicted_labels)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": micro_average_precision,
        "recall": micro_average_recall,
        "confusion": conf_matrix
    }


def __create_classification_indexes(true_labels: list, predicted_labels: list) -> dict:
    """
    Create the classification indexes based on the given labels.

    This function will extract all unique labels from all the given classes and puts them into the classification
    indexes.

    Args:
        true_labels: The true labels of the data.
        predicted_labels: The predicted labels of the data.

    Returns:
        The mapping of given labels to indexes within a confusion matrix.
    """
    classification_indexes = {}
    for classes in true_labels:
        if classes not in classification_indexes:
            classification_indexes[classes] = len(classification_indexes)
    for classes in predicted_labels:
        if classes not in classification_indexes:
            classification_indexes[classes] = len(classification_indexes)
    return classification_indexes


def __build_conf_matrix(class_indicies: dict, true_labels: list, predicted_labels: list) -> np.ndarray:
    """
    Calculate the confusion matrix from the given predictions.

    This function will calculate the confusion matrix given the true labels and estimated labels.

    Args:
        class_indicies: A dictionary which maps string labels to indexes in the conf matrix.
        true_labels: The true labels of the data.
        predicted_labels: The predicted labels of the data.

    Returns:
        Return the confusion matrix based upon the given labels.
    """
    conf_matrix = np.zeros((len(class_indicies), len(class_indicies)))

    for i in range(len(true_labels)):
        true_class_index = class_indicies[true_labels[i]]
        pred_class_index = class_indicies[predicted_labels[i]]
        conf_matrix[pred_class_index, true_class_index] += 1

    return conf_matrix


def __decompose_conf_matrix(conf_matrix: np.ndarray, n_items: int) -> tuple:
    """
    Extract TP, TN, FP & FN from the given confusion matrix.

    This function extracts the useful metrics from the conf. matrix to be used to calculate scores such as the F1 score.

    Args:
        conf_matrix:
            The confusion matrix.
        n_items:
            The total number of items in the dataset
            # TODO May not be neccessary as should be included in conf_matrix.

    Returns:
        The True positives, False positives, True negatives and False negatives from the conf matrix for each class.
    """
    n_classes = len(conf_matrix)

    # Setup arrays for varible.
    tp = np.diag(conf_matrix)
    fp = np.zeros(n_classes)
    fn = np.zeros(n_classes)
    tn = np.array([n_items for _ in range(n_classes)])

    # Iterate through to find the FP and FN rate.
    for prediction_classification in range(n_classes):
        for true_classification in range(n_classes):
            if true_classification != prediction_classification:
                fp[prediction_classification] += conf_matrix[prediction_classification, true_classification]
                fn[true_classification] += conf_matrix[prediction_classification, true_classification]

    tn = tn - fn - tp - fp

    return tp, fp, fn, tn
