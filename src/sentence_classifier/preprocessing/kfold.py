"""
KFold Implementation.

This module contains the KFold class which can be used for validation testings.

Usage:
    from preprocessing import KFold

    x, y = load_data() ...

    for train_idxs, val_idxs in KFold(x, 5):
        x_train, x_val = x[train_idxs], x[val_idxs]

        normalise....
        train...

Created by Sam Garlick. 23/02/2021
"""


class KFold:
    def __init__(self, data, k: int):
        """
        Initiate the KFold class.

        This function takes in two params, the data and the number of folds 'K'. This class will store the number of
        items in the data and K which can be used later in the __iter__ function.

        Args:
            data: The countable dataset which the number of items will be obtained.
            k: The number of folds.
        """
        self.__n = len(data)
        self.__k = k

        # Used during iteration to store the current fold.
        self.__iter_n = 0

    def __iter__(self):
        """
        Iterate through the class.

        This function can be used within for loops: for train_idx, val_idx in KFold(x, k):... to allow for KFold
        validation testing.

        Returns:
            self
        """
        # Reset the iter counter.
        self.__iter_n = 0

        return self

    def __next__(self):
        """
        Iterate to the next item in the __iter__ sequence.

        This function is called at every step in it's iteration sequence. First it checks if the current iteration is
        less than the number of folds as specified in the constructor otherwise stop the iteration. If we are in a valid
        step then we calculate the size of the window, formulate all possible indexes, calculate where the validation
        set starts and stops, then extract the validation set and testing set from the afformentioned indexes.

        Returns:
            The current iterations train and validation indexes or stop iteration.
        """
        if self.__iter_n < self.__k:
            fold_len = self.__n // self.__k

            all_idxs = [i for i in range(self.__n)]

            val_start_idx = self.__iter_n * fold_len
            val_end_idx = (self.__iter_n + 1) * fold_len if self.__iter_n < self.__k - 1 else self.__n
            val_idxs = all_idxs[val_start_idx:val_end_idx]

            train_idxs = all_idxs[:val_start_idx] + all_idxs[val_end_idx:]

            self.__iter_n += 1

            return train_idxs, val_idxs
        else:
            raise StopIteration
