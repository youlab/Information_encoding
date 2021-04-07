import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.losses import categorical_crossentropy
from keras.optimizers import adam
from tqdm import tqdm
import random
import os
import sys
from resources.data_utils import split_dataset,DataGenerator
from resources.model_utils import build_model,train_model_noitr,build_model_feedforward
from resources.savers import TrainingSaver
from resources.utils import label_str_to_dec

# *** Setting global constants ***
num_classes = 15 # to change according to the dictionary size
char_list = [str(i) for i in range(1, num_classes+1)]
char2int = {char: i for i, char in enumerate(char_list)}
num_replicates = 1000
dataset_size = num_classes * num_replicates
image_dimension = (80, 80) 

dataset = 'final' # name of the data set
path_final = '/root_dir/'+dataset+'/preprocessed_80/' # preprossed images directory
filename = '/root_dir/'+dataset+'/labels_spot_binary.csv' # label file path

# for spliting data set
ds_list = [0, 0.5, 0.75, 0.875, 0.9375, 0.96875, 0.9875, 0.9975]
ds_dict = {i+1:round((1-j)*dataset_size*0.8) for i,j in enumerate(ds_list)}
rank = 1 # change to choose training data set size
ds = ds_dict.get(rank) # training dataset size

path_hp = dataset+'/'+str(ds)+'/' # directory where training details are saved
if not os.path.exists(path_hp):
    os.makedirs(path_hp)

# *** Load data ***
# labels
labels = []
with open(filename, "r") as csvFile:
for row in csvFile:
    labels.append(row[:-1])
labels = np.asarray(labels)
main_labels = label_str_to_dec(labels[0:dataset_size], char2int)

# collecting the paths to all images in a set
image_prefix = "FImg_ID_"
image_suffix = ".jpg"
images_str = [s
    "{}{}{}{}".format(path_final, image_prefix, img_idx, image_suffix) for img_idx in range(1, dataset_size + 1)]
main_dataset = pd.DataFrame({"img_path": images_str, "label": main_labels}) 

# *** Training ***
generation_params = {"dim": image_dimension,
                     "nb_classes": num_classes,
                     "column_img": "img_path",
                     "column_label": "label",
                    }

# hyperparameters
epochs = 500
min_delta = 0.0000001
patience = 20
monitor = 'val_loss'
early_stopping_loss = EarlyStopping(monitor=monitor,
                                    min_delta=min_delta,
                                    patience=patience,
                                    verbose=0,
                                    mode='auto',
                                    baseline=None,
                                    restore_best_weights=True
                                   )
batch_size_list = [4,8,16,27,32]
batch_size = batch_size_list[0]
lr = 0.00001
optimizer_name = 'Adam'
seed = 25 # seed for random initialization

# *** Train ***
optimizer = adam(lr=lr, beta_1=0.9, beta_2=0.999)

# Split data set
df_train, df_valid, df_test = split_dataset(data_frame=main_dataset,
                                            rank=rank,
                                            column_label="label",
                                            random_state=25 
                                            )

train_generator = DataGenerator(data_frame=df_train, batch_size=batch_size, shuffle=True, **generation_params)
valid_generator = DataGenerator(data_frame=df_valid, batch_size=100, shuffle=False, **generation_params)
test_generator = DataGenerator(data_frame=df_test, batch_size=100, shuffle=False, **generation_params)

# save hyperparameters
hp_filename = path_hp + "hp_details.txt"
with open(hp_filename,"w") as f:
    f.write('Dataset: %s \n' % (dataset))
    f.write('Batch size: %i \n' % (batch_size))
    f.write('Initialization random seed: %i \n' % (seed))
    f.write('Training epoch: %i \n' % (epochs))
    f.write('Optimizer: %s (lr = %f) \n' % (optimizer_name, lr))
    f.write('Early Stopping: %s (min_delta = %.9f, patience = %i) \n' % (monitor, min_delta, patience))   

# build model
model = build_model(nb_classes=num_classes, image_length=image_dimension[0],seed=seed)
model.compile(loss=categorical_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy']
             )
# train
model,history,val_loss_history,loss_history,val_acc_history,acc_history,stopped_epoch = train_model_noitr(model=model,train_generator=train_generator,valid_generator=valid_generator,epochs=epochs,early_stopping=early_stopping_loss,verbose=1)

# *** Save training results ***
prefix = 'model_' 
experiment_saver = TrainingSaver(path_out=path_hp,
                                 stopped_epoch=stopped_epoch,
                                 nb_classes=num_classes,
                                 df_train=df_train,
                                 df_valid=df_valid,
                                 df_test=df_test,
                                 column_label="label",
                                 model=model,
                                 prefix=prefix
                                 )
experiment_saver.save_accuracy(workers=6,
                               train_generator=DataGenerator(data_frame=df_train,
                                                             batch_size=batch_size,
                                                             shuffle=False,
                                                             **generation_params),
                               valid_generator=valid_generator,
                               test_generator=test_generator)
experiment_saver.save_model()
experiment_saver.save_stopped_epoch()
experiment_saver.save_class_count()
experiment_saver.save_training_history(val_loss_history,loss_history,val_acc_history,acc_history, model_name="CNN")

