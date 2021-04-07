import numpy as np
import pandas as pd
from PIL import Image
from keras.utils import Sequence, to_categorical
from sklearn.model_selection import train_test_split


class DataGenerator(Sequence):
    """Generates data for Keras"""

    def __init__(self, data_frame: pd.DataFrame, column_img="img_path", column_label="label", batch_size=32,
                 dim=(80, 80), nb_classes=10, shuffle=True):
        """Initialization"""
        self.batch_size = batch_size
        self.dim = dim
        self.nb_classes = nb_classes
        self.shuffle = shuffle
        self.data_frame = data_frame
        self.column_img = column_img
        self.column_label = column_label
        self.indexes = np.arange(len(self.data_frame))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.data_frame) / self.batch_size))

    def __getitem__(self, batch_index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]

        # Generate data
        x, y = self.__data_generation(indexes)

        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.data_frame))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_list_ids):
        """Generates data containing batch_size samples"""  
        # X : (n_samples, *dim)
        # Initialization
        x = np.empty((self.batch_size, *self.dim))
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, idx in enumerate(batch_list_ids):
            # Store sample
            sample = self.data_frame.iloc[idx]
            image_path = sample[self.column_img]
            x[i] = np.array(Image.open(image_path))
            # Store class
            y[i] = sample[self.column_label]
        return x, to_categorical(y, num_classes=self.nb_classes)


def split_dataset(data_frame: pd.DataFrame, rank: int, column_label="label", random_state=25):
    """ Split training, test and validation sets
        ids_main_data: ids of all available data, in this case images
        labels_main_data: a list of labels in integer form for all available data
    """
    df_train, df_test, _, _ = train_test_split(data_frame, data_frame[column_label],
                                               stratify=data_frame[column_label],
                                               test_size=0.1,
                                               shuffle=True, random_state=random_state)
    if rank == 1:
        df_train, df_valid, _, _ = train_test_split(df_train,
                                                    df_train[column_label],
                                                    stratify=df_train[column_label],
                                                    test_size=0.11111,
                                                    shuffle=True,
                                                    random_state=random_state)
    else:
        df_train, df_valid, _, _ = train_test_split(df_train,
                                                    df_train[column_label],
                                                    stratify=df_train[column_label],
                                                    test_size=0.11111,
                                                    shuffle=True,
                                                    random_state=random_state)

        seg_ratio = [0.5, 0.75, 0.875, 0.9375, 0.96875, 0.9875, 0.9975]
        df_train, df_left_over, _, _ = train_test_split(df_train,
                                                        df_train[column_label],
                                                        stratify=df_train[column_label],
                                                        test_size=seg_ratio[rank - 2],
                                                        shuffle=True,
                                                        random_state=random_state)
    df_train.reset_index(drop=True, inplace=True)
    df_valid.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    return df_train, df_valid, df_test

