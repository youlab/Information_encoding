import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

from resources.model_utils import evaluate_model
from resources.utils import label_distribution, prediction_standardized, aggregate_generator_labels

class TrainingSaver:
    def __init__(self, path_out, nb_classes, df_train, df_valid, df_test, column_label, stopped_epoch, model, prefix):
        self.path_out = path_out
        self.nb_classes = nb_classes
        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test
        self.column_label = column_label
        self.stopped_epoch = stopped_epoch
        self.model = model
        
        self.base_path = "{}class_{}_trainsize_{}".format(path_out, str(nb_classes), str(len(df_train)))
        self.base_path = self.base_path + prefix
        
    def save_class_count(self):
        """Save class counts"""
        train_path = self.base_path + "_classcount_train.csv"
        valid_path = self.base_path + "_classcount_val.csv"
        test_path = self.base_path + "_classcount_test.csv"

        class_count_train = label_distribution(data_frame=self.df_train, column_target=self.column_label)
        class_count_val = label_distribution(data_frame=self.df_valid, column_target=self.column_label)
        class_count_test = label_distribution(data_frame=self.df_test, column_target=self.column_label)

        np.savetxt(train_path, class_count_train, delimiter=',', fmt='%d', header="Class,Count")
        np.savetxt(valid_path, class_count_val, delimiter=',', fmt='%d', header="Class,Count")
        np.savetxt(test_path, class_count_test, delimiter=',', fmt='%d', header="Class,Count")

        print("DONE: Saving class counts.")

    def save_stopped_epoch(self):
        path = self.base_path + "_stopped_epoch.txt"
        np.savetxt(path, self.stopped_epoch, delimiter=',', fmt='%d')

        print("DONE: Saving stopped epoch value.")

    def save_model(self):
        path = self.base_path + "_model.h5"
        self.model.save(path)

        print("DONE: Saving model.")

    def save_training_history(self, val_loss_history, loss_history, val_acc_history, acc_history, model_name="CNN"):
        path_loss = self.base_path + "_history_loss.png"
        path_acc = self.base_path + "_history_acc.png"
        train_valid_loss_plot(val_loss_history, loss_history, model_name, path_loss)
        train_valid_acc_plot(val_acc_history,acc_history, model_name, path_acc)

        print("DONE: Saving history plots.")

    def save_accuracy(self, workers, train_generator, valid_generator, test_generator):
        # Prediction
        train_predictions = prediction_standardized(
            evaluate_model(model=self.model, evaluation_generator=train_generator, workers=workers))
        valid_predictions = prediction_standardized(
            evaluate_model(model=self.model, evaluation_generator=valid_generator, workers=workers))
        test_predictions = prediction_standardized(
            evaluate_model(model=self.model, evaluation_generator=test_generator, workers=workers))

        train_true_labels = aggregate_generator_labels(data_generator=train_generator)
        valid_true_labels = aggregate_generator_labels(data_generator=valid_generator)
        test_true_labels = aggregate_generator_labels(data_generator=test_generator)

        # Accuracy
        acc_train = accuracy_score(train_true_labels, train_predictions)
        acc_valid = accuracy_score(valid_true_labels, valid_predictions)
        acc_test = accuracy_score(test_true_labels, test_predictions)

        accuracy_list = [acc_train, acc_valid, acc_test]

        # Save accuracy
        filename = self.base_path + "_accuracy.txt"
        np.savetxt(filename, accuracy_list, delimiter=',', fmt='%f', header="Train,Valid,Test")

        print("DONE: Model evaluation.")


def train_valid_loss_plot(val_loss_history, loss_history, model_name: str, fig_path: str) -> None:
    """ Plots Train and Validation Loss graphs upon receiving the `history` of a model."""
    plt.ioff()
    fig, ax = plt.subplots(1, 1)
    ax.plot(val_loss_history, color='tab:blue', label="Validation Loss")
    ax.plot(loss_history, color='tab:red', label="Training Loss")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss vs. Iterations of {}'.format(model_name))
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)


def train_valid_acc_plot(val_acc_history, acc_history, model_name: str, fig_path: str) -> None:
    """ Plots Train and Validation accuracy graphs upon receiving the `history` of a model."""
    plt.ioff()
    fig, ax = plt.subplots(1, 1)
    ax.plot(val_acc_history, color='tab:blue', label="Validation Accuracy")
    ax.plot(acc_history, color='tab:red', label="Training Accuracy")
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Iterations of {}'.format(model_name))
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
