import tqdm
import time
import random

import numpy as np

def get_random_key(a_huge_key_list):
    L = len(a_huge_key_list)
    i = np.random.randint(0, L)
    return a_huge_key_list[i]

def make_pairs(x, y):
    """Creates a tuple containing image pairs with corresponding label.

    Arguments:
        x: List containing images, each index in this list corresponds to one image.
        y: List containing labels, each label with datatype of `int`.

    Returns:
        Tuple containing two numpy arrays as (pairs_of_samples, labels),
        where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
        labels are a binary array of shape (2len(x)).
    """

    num_classes = max(y) + 1
    #print(num_classes)
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    pairs = []
    labels = []
    label_anchor, label_non_anchor = [], []

    for idx1 in tqdm.tqdm(range(len(x))):
        start = time.time()
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
        if len(digit_indices[label1]) == 0:
            print("label 1 : " + str(label1))
        idx2 = get_random_key(digit_indices[label1])
        x2 = x[idx2]
        #print("first : {}".format(time.time() - start))

        start = time.time()
        pairs += [[x1, x2]]
        labels += [0]
        label_anchor.append(label1)
        label_non_anchor.append(y[idx2])

        # add a non-matching example
        label2 = random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)
            
        if len(digit_indices[label2]) == 0:
            print("label 2 : " + str(label2))
        #print("second : {}".format(time.time() - start))

        start = time.time()
        idx2 = get_random_key(digit_indices[label2])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [1]
        
        label_anchor.append(label1)
        label_non_anchor.append(label2)
        
    return np.array(pairs), np.array(labels).astype("float32"), np.array(label_anchor), np.array(label_non_anchor)