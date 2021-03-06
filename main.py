import os
import cv2
import time
import yaml
import json
import wandb
import pandas
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
import seaborn as sns
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras import layers, models, optimizers, losses, metrics
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import get_random_img, f1_loss, macro_f1, get_timing, perfomance_grid

# --------------------------------------------------- Main parameters --------------------------------------------------
# Training
MODE = 'train'
MODEL_NAME = 'ResNet_V2'
BATCH_SIZE = 64
LR = 1e-5
EPOCHS = 100                    # TODO: Checkpoint
OPTIMIZER = 'radam'
LABEL_LOSS = 'bce'
LABEL_LOSS_WEIGHT = 1.0
POINT_LOSS = 'logcosh'
POINT_LOSS_WEIGHT = 10.0
IS_TRAINABLE = False            # TODO: Checkpoint

# Testing
TEST_MODEL_DIR = 'models/ResNet_V2'
TEST_FILES = ['data/video/004.avi']     # 001, 004, 016
VERBOSE = 1
# TEST_FILES = ['data/img/001_025.png', 'data/img/002_028.png', 'data/img/003_032.png', 'data/img/004_016.png']
# TEST_FILES = glob('data/img/*' + '004' + '_*')

# ------------------------------------------------ Additional parameters -----------------------------------------------
DATA_PATH = 'data/data.xlsx'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
BUFFER_SIZE = 4000
palette = sns.color_palette("pastel", n_colors=11)  # pastel, hls, Paired, Set2, Set3

CALLBACK_IMAGES = ['data/img/003_007.png', 'data/img/003_034.png', 'data/img/004_002.png', 'data/img/004_013.png',
                   'data/img/005_003.png', 'data/img/005_007.png', 'data/img/005_022.png', 'data/img/008_030.png',
                   'data/img/008_045.png', 'data/img/009_003.png', 'data/img/009_023.png', 'data/img/010_006.png',
                   'data/img/010_031.png', 'data/img/012_032.png', 'data/img/012_063.png', 'data/img/014_050.png',
                   'data/img/014_110.png', 'data/img/016_070.png', 'data/img/016_180.png', 'data/img/017_103.png']
LABEL_NAMES = ['AA1', 'AA2', 'STJ1', 'STJ2', 'CD', 'CM', 'CP', 'CT', 'PT', 'FE1', 'FE2']
POINT_NAMES = ['AA1_x', 'AA1_y', 'AA2_x', 'AA2_y', 'STJ1_x', 'STJ1_y', 'STJ2_x', 'STJ2_y', 'CD_x', 'CD_y',
               'CM_x', 'CM_y', 'CP_x', 'CP_y', 'CT_x', 'CT_y', 'PT_x', 'PT_y', 'FE1_x', 'FE1_y', 'FE2_x', 'FE2_y']
POINT_COLORS = {'AA1': palette[0], 'AA2':  palette[1], 'STJ1':  palette[2], 'STJ2':  palette[3],
                'CD':  palette[4], 'CM':  palette[5], 'CP':  palette[6], 'CT':  palette[7], 'PT':  palette[8],
                'FE1':  palette[9], 'FE2':  palette[10]}

# -------------------------------------------- Initialize ArgParse container -------------------------------------------
parser = argparse.ArgumentParser(description='Keypoint tracking and classification')
parser.add_argument('-mo', '--mode', metavar='', default=MODE, type=str, help='mode: train or test')
# Training arguments
parser.add_argument('-mn', '--model_name', metavar='', default=MODEL_NAME, type=str, help='architecture of the model')
parser.add_argument('-opt', '--optimizer', metavar='', default=OPTIMIZER, type=str, help='type of an optimizer')
parser.add_argument('-clo', '--label_loss', metavar='', default=LABEL_LOSS, type=str, help='classification loss')
parser.add_argument('-plo', '--point_loss', metavar='', default=POINT_LOSS, type=str, help='regression loss')
parser.add_argument('-cw', '--label_loss_weight', metavar='', default=LABEL_LOSS_WEIGHT, type=float, help='label loss weight')
parser.add_argument('-pw', '--point_loss_weight', metavar='', default=POINT_LOSS_WEIGHT, type=float, help='point loss weight')
parser.add_argument('-lr', '--learning_rate', metavar='', default=LR, type=float, help='learning rate')
parser.add_argument('-bas', '--batch_size', metavar='', default=BATCH_SIZE, type=int, help='batch size')
parser.add_argument('-ep', '--epochs', metavar='', default=EPOCHS, type=int, help='number of epochs for training')
parser.add_argument('-bus', '--buffer_size', metavar='', default=BUFFER_SIZE, type=int, help='buffer size')
parser.add_argument('-ist', '--is_trainable', action='store_true', default=IS_TRAINABLE, help='whether to train backbone')
parser.add_argument('--callback_images', metavar='', default=CALLBACK_IMAGES, type=list, help='images for callback prediction')
# Testing arguments
parser.add_argument('-tmd', '--test_model_dir', metavar='', default=TEST_MODEL_DIR, type=str, help='model directory for testing mode')
parser.add_argument('-tf', '--test_files', nargs='+', metavar='', default=TEST_FILES, type=str, help='list of tested images')
parser.add_argument('-ver', '--verbose', metavar='', default=VERBOSE, type=int, help='verbosity mode')
# Common arguments
parser.add_argument('--point_colors', metavar='', default=POINT_COLORS, type=dict, help='point colors')
parser.add_argument('--label_names', metavar='', default=LABEL_NAMES, type=list, help='list of label names')
parser.add_argument('--point_names', metavar='', default=POINT_NAMES, type=list, help='list of point names')
args = parser.parse_args()

if args.model_name == 'MobileNet_V2' or args.model_name == 'ResNet_V2':
    args.img_size = (224, 224, 3)
elif args.model_name == 'Inception_V3' or args.model_name == 'Inception_ResNet_v2':
    args.img_size = (299, 299, 3)
elif args.model_name == 'EfficientNet_B5':
    args.img_size = (456, 456, 3)
else:
    raise ValueError('Incorrect MODEL_NAME, please change it!')

# ----------------------------------------- Callback for logging parameters --------------------------------------------
class ParamsLogger(tf.keras.callbacks.Callback):
    def __init__(self, trainable_params, non_trainable_params):
        self.trainable_params = trainable_params
        self.non_trainable_params = non_trainable_params
        self.total_params = self.trainable_params + self.non_trainable_params
    def on_epoch_end(self, epoch, logs=None):
        wandb.log({'total_params': self.total_params}, commit=False)
        wandb.log({'trainable_params': self.trainable_params}, commit=False)
        wandb.log({'non_trainable_params': self.non_trainable_params}, commit=False)

# ----------------------------------------- Callback for logging total loss --------------------------------------------
class LossLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        total_loss = args.label_loss_weight * logs['label_loss'] + args.point_loss_weight * logs['point_loss']
        val_total_loss = args.label_loss_weight * logs['val_label_loss'] + args.point_loss_weight * logs['val_point_loss']
        wandb.log({'total_loss': total_loss}, commit=False)
        wandb.log({'val_total_loss': val_total_loss}, commit=False)

# -------------------------------------------- Callback for saving images ----------------------------------------------
class ImageSaver(tf.keras.callbacks.Callback):
    def __init__(self, image_paths, save_dir, draw_gt):
        data_processor = DataProcessor()
        self.net = Net()
        self.image_paths = image_paths
        self.imgs = data_processor.process_images(paths=image_paths,
                                                  img_height=args.img_size[0],
                                                  img_width=args.img_size[1],
                                                  img_channels=args.img_size[2])
        self.save_dir = os.path.join(args.train_model_dir, save_dir)
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        self.draw_gt = draw_gt
        source_df = pandas.read_excel(DATA_PATH, index_col=None, na_values=['NA'], usecols="C:AM")

        self.gt_labels = []
        self.gt_probs = []
        self.gt_coords = []
        for img_path in image_paths:
            gt_img_idx = int(source_df[source_df['Path'] == img_path].index.values)

            # Get GT values
            gt_label_probs = source_df.loc[gt_img_idx, args.label_names]
            gt_label_probs = gt_label_probs.to_numpy(dtype=np.float32)
            gt_label_probs = np.expand_dims(gt_label_probs, axis=0)
            out_bin = gt_label_probs == 1.0

            gt_point_coords = source_df.loc[gt_img_idx, args.point_names]
            gt_point_coords = gt_point_coords.to_numpy(dtype=np.float32)
            gt_point_coords = np.expand_dims(gt_point_coords, axis=0)

            # Get classification labels
            mlb = MultiLabelBinarizer(classes=args.label_names)
            mlb.fit(y=args.label_names)
            gt_labels = mlb.inverse_transform(yt=out_bin)
            gt_labels = list(gt_labels[0])

            # Get points coordinates
            x_coords = np.zeros((1, gt_label_probs.shape[1]), dtype=np.float)
            y_coords = np.zeros((1, gt_label_probs.shape[1]), dtype=np.float)
            for i in range(gt_label_probs.shape[1]):
                x_coords[0, i] = gt_point_coords[0, 2 * i]
                y_coords[0, i] = gt_point_coords[0, 2 * i + 1]

            # Get remaining classification probabilities
            deleted_points = np.argwhere(out_bin == False)
            deleted_points = list(deleted_points[:, 1])
            gt_label_probs = np.delete(gt_label_probs, deleted_points)
            gt_label_probs = np.round(gt_label_probs, decimals=2)
            gt_label_probs = np.expand_dims(gt_label_probs, axis=0)

            # Get remaining points coordinates
            x_coords = x_coords[out_bin]
            x_coords = np.expand_dims(x_coords, axis=0)
            y_coords = y_coords[out_bin]
            y_coords = np.expand_dims(y_coords, axis=0)
            gt_point_coords = np.concatenate((x_coords, y_coords))
            gt_point_coords = np.round(gt_point_coords, decimals=0)
            gt_point_coords = gt_point_coords.astype(int)

            self.gt_labels.append(gt_labels)
            self.gt_probs.append(gt_label_probs)
            self.gt_coords.append(gt_point_coords)

    def on_epoch_end(self, epoch, logs=None):
        start = time.time()
        model_probs = self.model.predict(self.imgs)
        inference_time = (time.time() - start)/self.imgs.shape[0]
        print('\n')
        print('-' * 100)
        print('Inference time: {:.3f} seconds'.format(inference_time))
        print('-' * 100)
        wandb.log({'inference_time': inference_time}, commit=False)
        images, pred_labels, pred_probs, pred_coords = self.net.process_predictions(model_output=model_probs, test_files=self.image_paths,
                                                                                    thresh_label=0.5, thresh_x=0.01, thresh_y=0.01)
        for idx, image in enumerate(images):
            pred_label = pred_labels[idx]
            pred_prob = pred_probs[idx]
            pred_coord = pred_coords[idx]
            gt_label = self.gt_labels[idx]
            gt_prob = self.gt_probs[idx]
            gt_coord = self.gt_coords[idx]

            if self.draw_gt:
                image = self.net.put_points_on_image(image, gt_label, gt_prob, gt_coord,
                                                     shape='square', add_label=False, add_prob=False)
            image = self.net.put_points_on_image(image, pred_label, pred_prob, pred_coord,
                                                 shape='star', add_label=True, add_prob=True)
            img_name = os.path.basename(self.image_paths[idx])
            img_name, img_ext = os.path.splitext(img_name)[0], os.path.splitext(img_name)[1]
            save_name = img_name + '_epoch=' + str(epoch).zfill(3) + img_ext
            save_path = os.path.join(self.save_dir, save_name)
            image = (255*image).astype(np.uint8)
            cv2.imwrite(save_path, image)

            if idx == 0 or idx == 3 or idx == 9:    # 003_007, 004_013, 009_003
                loss_val = args.label_loss_weight * logs['label_loss'] + args.point_loss_weight * logs['point_loss']
                f1_val = logs['label_macro_f1']
                mae_val = logs['point_mae']

                note_height = 50
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 1
                thickness = 1
                note = 255 * np.ones(shape=(note_height, image.shape[1], image.shape[2]), dtype=np.uint8)

                text = "Loss: {:.2f} | F1: {:.2f} | MAE: {:.2f}".format(loss_val, f1_val, mae_val)
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = (note.shape[1] - text_size[0]) // 2
                text_y = (note.shape[0] + text_size[1]) // 2

                cv2.putText(img=note, text=text, org=(text_x, text_y), color=(0, 0, 0),
                            fontFace=font, fontScale=font_scale, thickness=thickness, lineType=cv2.LINE_AA)
                image = np.vstack((note, image))

            if idx == 0:
                wandb.log({'P1' + '_' + args.train_model_dir.split(os.sep)[1]: [wandb.Image(image)]}, commit=False)
            elif idx == 3:
                wandb.log({'P2' + '_' + args.train_model_dir.split(os.sep)[1]: [wandb.Image(image)]}, commit=False)
            elif idx == 9:
                wandb.log({'P3' + '_' + args.train_model_dir.split(os.sep)[1]: [wandb.Image(image)]}, commit=False)

# ------------------------------------------ Data processing and prefetching -------------------------------------------
class DataProcessor():
    def __init__(self):
        pass

    def visualize(self, img_src, img_aug):
        """Function used to compare original and augmented image
        Args:
              img_src: original image
              img_aug: augmented image
        """
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.title('Original image')
        plt.imshow(np.squeeze(img_src))
        plt.subplot(1, 2, 2)
        plt.title('Compared image')
        plt.imshow(np.squeeze(img_aug))
        plt.show()

    def get_inputs_and_targets(self, path_to_data):
        """Function that returns a list of image names, array of label and array of coordinates.
        Args:
            path_to_data: string representing path to xlsx dataset info
        """
        # # TODO: Checkpoint (nrows=500)
        source_df = pandas.read_excel(path_to_data, index_col=None, na_values=['NA'], usecols="B:AM")
        path_df = source_df['Path']
        label_df = source_df[args.label_names]
        point_df = source_df[args.point_names]
        img_paths = list(path_df)
        label_targets = label_df.to_numpy()
        point_targets = point_df.to_numpy() / 1000
        return img_paths, label_targets, point_targets

    def map_func(self, path):
        img_string = tf.io.read_file(path)
        img_input = tf.image.decode_png(img_string, channels=args.img_size[2])
        img_resized = tf.image.resize(images=img_input, size=(args.img_size[0], args.img_size[1]))
        img_norm = img_resized / 255.0
        img_output = tf.reshape(tensor=img_norm, shape=(args.img_size[0], args.img_size[1], args.img_size[2]))
        # Used for debugging
        # self.visualize(img_input, img_output)
        return img_output

    def process_images(self, paths, img_height, img_width, img_channels):
        img_inputs = np.ndarray(shape=(len(paths), img_height, img_width, img_channels), dtype=np.float32)
        idx = 0
        start = time.time()
        for path in tqdm(paths):
            img_string = tf.io.read_file(path)
            img_input = tf.image.decode_png(img_string, channels=img_channels)
            img_resized = tf.image.resize(images=img_input, size=(img_height, img_width))
            img_norm = img_resized / 255.0
            img_inputs[idx] = img_norm
            idx += 1
            # Debugging only
            # self.visualize(img_input, img_output)
            # self.visualize(img_norm, img_output)
        proc_time = time.time() - start
        print('-' * 100)
        print('Total pre-processing time.....: {:1.3f} seconds'.format(proc_time))
        print('Average pre-processing time...: {:1.3f} seconds per image'.format(proc_time / img_inputs.shape[0]))
        print('Average pre-processing FPS....: {:1.1f} frames per second'.format(1. / (proc_time / img_inputs.shape[0])))
        print('-' * 100)
        return img_inputs

    def process_video(self, path, img_height, img_width, img_channels):
        cap = cv2.VideoCapture(path[0])
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        img_inputs = np.ndarray(shape=(num_frames, img_height, img_width, img_channels),
                                dtype=np.float32)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:     # frame is None
                break
            img_resized = tf.image.resize(images=frame, size=(img_height, img_width), method='bilinear')
            img_norm = img_resized / 255.0
            img_inputs[idx] = img_norm
            idx += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # Debugging only
            # self.visualize(frame, img_norm)
        return img_inputs

    def create_dataset(self, img_paths, label_targets, point_targets, is_caching=None, cache_name=None):
        # Create a dataset of file images, labels and coordinates
        # dataset = tf.data.Dataset.from_tensor_slices(({'img_input': img_inputs},
        #                                               {'label': label_targets, 'point': point_targets}))
        # dataset = tf.data.Dataset.from_tensor_slices((img_inputs, (label_targets, point_targets)))
        # Debugging only
        # temp_1 = self.map_func(img_paths[1])
        dataset_images = tf.data.Dataset.from_tensor_slices(img_paths)
        dataset_images = dataset_images.map(map_func=self.map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset_targets = tf.data.Dataset.from_tensor_slices((label_targets, point_targets))
        dataset = tf.data.Dataset.zip((dataset_images, dataset_targets))

        if is_caching:
            # This is a small dataset, only load it once, and keep it in memory.
            # use `.cache(filename)` to cache preprocessing work for datasets that don't fit in memory.
            if isinstance(cache_name, str):
                cache_path = os.path.join('data', cache_name)
                if not os.path.exists(os.path.split(cache_path)[0]):
                    os.makedirs(os.path.split(cache_path)[0])
                dataset = dataset.cache(cache_path)
            else:
                dataset = dataset.cache()

        # Shuffle the data each buffer size
        dataset = dataset.shuffle(buffer_size=args.buffer_size)

        # Repeats the dataset so each original value is seen `count` times
        dataset = dataset.repeat(count=1)

        # Batch the data for multiple steps
        dataset = dataset.batch(batch_size=args.batch_size)

        # Fetch batches in the background while the model is training.
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

# --------------------------------------------------- Neural network ---------------------------------------------------
class Net:
    def __init__(self):
        pass

    def get_optimizer(self, optimizer, learning_rate):
        if optimizer == 'adam':
            opt = optimizers.Adam(lr=learning_rate)
        elif optimizer == 'adamax':
            opt = optimizers.Adamax(lr=learning_rate)
        elif optimizer == 'radam':
            opt = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            # lambda x: 1.
            # lambda x: gamma ** x
            # lambda x: 1 / (2.0 ** (x - 1))
            lr_schedule = tfa.optimizers.cyclical_learning_rate.CyclicalLearningRate(initial_learning_rate=learning_rate,
                                                                                     maximal_learning_rate=100*learning_rate,
                                                                                     step_size=25,
                                                                                     scale_mode="iterations",
                                                                                     scale_fn=lambda x: 0.95 ** x,
                                                                                     name="CustomScheduler")
            opt = optimizers.SGD(learning_rate=lr_schedule)
        else:
            raise ValueError('Undefined OPTIMIZER_TYPE!')
        return opt

    def save_model(self, model):
        print('-' * 100)
        print('Saving of the model architecture...')
        start = time.time()
        with open(os.path.join(args.train_model_dir, 'architecture.json'), 'w') as f:
            f.write(model.to_json())
        end = time.time()
        img_path = os.path.join(args.train_model_dir, args.model_name + '.png')
        tf.keras.utils.plot_model(model, to_file=img_path, show_shapes=True)
        print('Saving of the model architecture takes ({:1.3f} seconds)'.format(end - start))
        print('-' * 100)

    def load_model(self, model_dir):
        print('-' * 100)
        print('Loading the model and its weights...')
        start = time.time()
        architecture_path = os.path.join(model_dir, 'architecture.json')
        weights_path = os.path.join(model_dir, 'weights.h5')
        with open(architecture_path, 'r') as f:
            model = models.model_from_json(f.read(), custom_objects={'KerasLayer': hub.KerasLayer})
        model.load_weights(weights_path)
        # Used for loading the whole model
        # model = tf.keras.models.load_model(filepath='model.h5',
        #                                    custom_objects={'KerasLayer': hub.KerasLayer},
        #                                    compile=False)
        # Old verson of model compilation (not suitable for MODE = 'test')
        # model.compile(optimizer=self.get_optimizer(learning_rate=self.config.learning_rate),
        #               loss=macro_f1_loss,
        #               metrics=[macro_f1])
        # Optional case for retraining the model
        # model.compile(optimizer=self.get_optimizer(optimizer=optimizer, learning_rate=learning_rate),
        #               loss=macro_f1_loss,
        #               metrics=[macro_f1])
        end = time.time()
        print('Loading the model and its weights takes ({:1.3f} seconds)'.format(end - start))
        print('-' * 100)
        return model

    def build_model(self):
        if args.model_name == 'MobileNet_V2':
            model_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
        elif args.model_name == 'ResNet_V2':
            model_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"
        elif args.model_name == 'Inception_V3':
            model_url = "https://tfhub.dev/google/imagenet/inception_v3/classification/4"
        elif args.model_name == 'Inception_ResNet_v2':
            model_url = "https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/4"
        elif args.model_name == 'EfficientNet_B5':
            model_url = 'https://tfhub.dev/google/efficientnet/b5/feature-vector/1'
        else:
            raise ValueError('Incorrect MODEL_TYPE value, please check it!')

        # -------------------------------------------- Building the model ----------------------------------------------
        img_input = layers.Input(shape=(args.img_size[0], args.img_size[1], args.img_size[2]), name='img_input')
        hub_layer = hub.KerasLayer(handle=model_url, trainable=args.is_trainable, name=args.model_name)
        output = hub_layer(img_input)
        label_layer = layers.Dense(1024, activation='relu', name='classifier')(output)
        label_output = layers.Dense(len(args.label_names), activation='sigmoid', name='label')(label_layer)
        point_layer = layers.Dense(1024, activation='relu', name='regressor')(output)
        point_output = layers.Dense(len(args.point_names), activation='sigmoid', name='point')(point_layer)
        model = models.Model(inputs=img_input, outputs=[label_output, point_output], name=args.model_name)
        model.summary()

        # -------------------------------------------- Compiling the model ---------------------------------------------
        if args.label_loss == 'f1':
            label_loss = f1_loss
        elif args.label_loss == 'bce':
            label_loss = losses.BinaryCrossentropy(from_logits=False, label_smoothing=0)
        else:
            raise ValueError('Incorrect LABEL_LOSS value, please check it!')

        if args.point_loss == 'mae':
            point_loss = losses.MeanAbsoluteError()
        elif args.point_loss == 'mse':
            point_loss = losses.MeanSquaredError()
        elif args.point_loss == 'huber':
            point_loss = losses.Huber(delta=0.1)
        elif args.point_loss == 'logcosh':
            point_loss = losses.LogCosh()
        else:
            raise ValueError('Incorrect POINT_LOSS value, please check it!')

        label_metrics = [macro_f1,
                         tfa.metrics.F1Score(num_classes=len(args.label_names), average='micro', threshold=0.5, name='micro_f1'),
                         metrics.BinaryAccuracy(name='accuracy', threshold=0.5),
                         metrics.Precision(top_k=None, thresholds=0.5, name='precision'),
                         metrics.Recall(top_k=None, thresholds=0.5, name='recall'),
                         metrics.TruePositives(name='tp', thresholds=0.5),
                         metrics.FalsePositives(name='fp', thresholds=0.5),
                         metrics.TrueNegatives(name='tn', thresholds=0.5),
                         metrics.FalseNegatives(name='fn', thresholds=0.5)]
        point_metrics = [metrics.MeanAbsoluteError(name='mae'),
                         metrics.RootMeanSquaredError(name='rmse'),
                         metrics.MeanSquaredError(name='mse')]

        model.compile(optimizer=self.get_optimizer(args.optimizer, args.learning_rate),
                      loss=[label_loss, point_loss],
                      metrics=[label_metrics, point_metrics],
                      loss_weights=[args.label_loss_weight, args.point_loss_weight])
        # model.compile(optimizer=self.get_optimizer(args.optimizer, args.learning_rate),
        #               loss={'label': label_loss, 'point': point_loss},
        #               metrics={'label': label_metrics, 'point': point_metrics},
        #               loss_weights={'label': args.label_loss_weight, 'point': args.point_loss_weight})
        return model

    def train_model(self):
        # -------------------------------------- Data processing and prefetching ---------------------------------------
        data_processor = DataProcessor()
        img_inputs, label_targets, point_targets = data_processor.get_inputs_and_targets(path_to_data=DATA_PATH)
        X_train_1, X_val_1, y_train, y_val = train_test_split(img_inputs, label_targets,
                                                              shuffle=True, test_size=0.2, random_state=11)
        X_train_2, X_val_2, z_train, z_val = train_test_split(img_inputs, point_targets,
                                                              shuffle=True, test_size=0.2, random_state=11)
        if np.array_equal(X_train_1, X_train_2) and np.array_equal(X_val_1, X_val_2):
            X_train = X_train_1
            X_val = X_val_1
        else:
            raise ValueError('Inputs for classification and regression subsets are not equal!')

        print('-' * 100)
        print("Number of images for training.....: {} ({:.1%})".format(len(X_train),
                                                                       round(len(X_train)/(len(X_train)+len(X_val)), 1)))
        print("Number of images for validation...: {}  ({:.1%})".format(len(X_val),
                                                                       round(len(X_val)/(len(X_train)+len(X_val)), 1)))
        print('-' * 100)

        # train_ds = data_processor.create_dataset(X_train, y_train, z_train,
        #                                          is_caching=True, cache_name='train_' + str(args.img_size[0]))
        # val_ds = data_processor.create_dataset(X_val, y_val, z_val,
        #                                        is_caching=True, cache_name='val_' + str(args.img_size[0]))
        train_ds = data_processor.create_dataset(X_train, y_train, z_train, is_caching=True, cache_name=None)
        val_ds = data_processor.create_dataset(X_val, y_val, z_val, is_caching=True, cache_name=None)

        for image, target in train_ds.take(1):
            print('-' * 100)
            print("Image batch shape...: {}".format(image.numpy().shape))
            print("Label batch shape...: {}".format(target[0].numpy().shape))
            print("Point batch shape...: {}".format(target[1].numpy().shape))
            print('-' * 100)

        # ------------------------------------------  Initialize W&B project -------------------------------------------
        run_time = datetime.now().strftime("%d%m_%H%M")
        run_name = args.model_name + '_{}'.format(run_time)
        args.train_model_dir = os.path.join('models', run_name)
        os.makedirs(args.train_model_dir)

        params = dict(img_size=(args.img_size[0], args.img_size[1], args.img_size[2]),
                      model_name=args.model_name,
                      model_dir=args.train_model_dir,
                      optimizer=args.optimizer,
                      label_loss=args.label_loss,
                      label_loss_weight=args.label_loss_weight,
                      point_loss=args.point_loss,
                      point_loss_weight=args.point_loss_weight,
                      batch_size=args.batch_size,
                      buffer_size=args.buffer_size,
                      epochs=args.epochs,
                      learning_rate=args.learning_rate,
                      trainable_backbone=args.is_trainable)
        # TODO: Checkpoint
        wandb.init(project='tavr', dir=args.train_model_dir, name=run_name, sync_tensorboard=True, config=params)
        wandb.config.update(params)

        # ------------------------------------------- Show training options --------------------------------------------
        print('-' * 100)
        print('Training options:')
        print('Model name............: {}'.format(args.model_name))
        print('Model directory.......: {}'.format(args.train_model_dir))
        print('Classification loss...: {} (weight: {})'.format(args.label_loss, args.label_loss_weight))
        print('Regression loss.......: {} (weight: {})'.format(args.point_loss, args.point_loss_weight))
        print('Optimizer.............: {}'.format(args.optimizer))
        print('Learning rate.........: {}'.format(args.learning_rate))
        print('Batch size............: {}'.format(args.batch_size))
        print('Epochs................: {}'.format(args.epochs))
        print('Trainable backbone....: {}'.format(args.is_trainable))
        print('Image dimensions......: {}x{}x{}'.format(args.img_size[0], args.img_size[1], args.img_size[2]))
        print('Buffer size...........: {}'.format(args.buffer_size))
        print('-' * 100)

        # ------------------------------------------------- Build model ------------------------------------------------
        model = self.build_model()
        self.save_model(model=model)
        trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])

        # -------------------------------------------  Initialize callbacks --------------------------------------------
        params_logger = ParamsLogger(trainable_params=trainable_params, non_trainable_params=non_trainable_params)
        loss_logger = LossLogger()
        img_saver = ImageSaver(image_paths=args.callback_images, save_dir='predictions_per_epoch', draw_gt=True)
        check_pointer = ModelCheckpoint(filepath=os.path.join(args.train_model_dir, 'weights.h5'),
                                        monitor='val_loss',
                                        save_best_only=True,
                                        save_weights_only=False,
                                        mode='min',
                                        verbose=1)
        wandb_logger = wandb.keras.WandbCallback(monitor='val_loss',
                                                 mode='min',
                                                 save_weights_only=False,
                                                 save_model=False,
                                                 log_evaluation=False,
                                                 verbose=1)
        earlystop = EarlyStopping(monitor='val_loss',
                                  min_delta=0.005,
                                  patience=5,
                                  mode='min',
                                  verbose=1)

        # ------------------------------------------ Check model's operability -----------------------------------------
        for batch in val_ds:
            print('-' * 100)
            print("Model operability test")
            print("Class label prediction: {}".format(np.around(model.predict(batch)[0][0], decimals=2)))
            print("Point coordinates prediction: {}".format(np.around(model.predict(batch)[1][0], decimals=2)))
            print('-' * 100)
            break

        # ------------------------------------------------- Train model ------------------------------------------------
        start = time.time()
        history = model.fit(x=train_ds,
                            epochs=args.epochs,
                            validation_data=val_ds,
                            verbose=1,
                            callbacks=[params_logger, loss_logger, img_saver, check_pointer, wandb_logger, earlystop])
        end = time.time()
        print('\nTraining of the model took: {}'.format(get_timing(end - start)))

        # --------------------------------------- Save WANDB history for the run ---------------------------------------
        wandb_dir = os.path.join(args.train_model_dir, 'wandb')
        for root, dirs, files in os.walk(wandb_dir):
            for file in files:
                if file == 'wandb-history.jsonl':
                    history_path = os.path.join(root, file)
                    metrics_df = pandas.DataFrame()
                    with open(history_path) as f:
                        for line in f:
                            row = json.loads(line)
                            metrics_row = pandas.DataFrame.from_records(data=[row])
                            metrics_df = metrics_df.append(metrics_row)
                    metrics_df = metrics_df.drop(metrics_df.columns[[0, 1, 2, 4, 5]], axis=1)
                    metrics_df.to_csv(os.path.join(args.train_model_dir, "logs.csv"), index=False)
                    break

        # ------------------------------------ Show training and validation outputs ------------------------------------
        label_loss, val_label_loss = history.history['label_loss'], history.history['val_label_loss']
        point_loss, val_point_loss = history.history['point_loss'], history.history['val_point_loss']
        accuracy, val_accuracy = history.history['label_accuracy'], history.history['val_label_accuracy']
        macro_f1, val_macro_f1 = history.history['label_macro_f1'], history.history['val_label_macro_f1']
        micro_f1, val_micro_f1 = history.history['label_micro_f1'], history.history['val_label_micro_f1']
        precision, val_precision = history.history['label_precision'], history.history['val_label_precision']
        recall, val_recall = history.history['label_recall'], history.history['val_label_recall']
        mae, val_mae = history.history['point_mae'], history.history['val_point_mae']
        rmse, val_rmse = history.history['point_rmse'], history.history['val_point_rmse']
        mse, val_mse = history.history['point_mse'], history.history['val_point_mse']

        print('-' * 100)
        print('Label prediction on training / validation')
        print("Loss........: {:.2f} / {:.2f}".format(label_loss[-1], val_label_loss[-1]))
        print("Accuracy....: {:.2f} / {:.2f}".format(accuracy[-1], val_accuracy[-1]))
        print("Macro F1....: {:.2f} / {:.2f}".format(macro_f1[-1], val_macro_f1[-1]))
        print("Micro F1....: {:.2f} / {:.2f}".format(micro_f1[-1], val_micro_f1[-1]))
        print("Precision...: {:.2f} / {:.2f}".format(precision[-1], val_precision[-1]))
        print("Recall......: {:.2f} / {:.2f}".format(recall[-1], val_recall[-1]))
        print('\nPoint prediction on training / validation')
        print("Loss........: {:.2f} / {:.2f}".format(point_loss[-1], val_point_loss[-1]))
        print("MAE.........: {:.2f} / {:.2f}".format(mae[-1], val_mae[-1]))
        print("RMSE........: {:.2f} / {:.2f}".format(rmse[-1], val_rmse[-1]))
        print("MSE.........: {:.2f} / {:.2f}".format(mse[-1], val_mse[-1]))
        print('-' * 100)

        # --------------------- Performance table of the model with different levels of threshold ----------------------
        perfomance_grid(ds=val_ds, target=y_val, label_names=args.label_names, model=model, save_dir=args.train_model_dir)

    def test_model(self, test_model_dir, test_files):
        # -------------------------------------- Getting of a YAML configuration ---------------------------------------
        for root, dirs, files in os.walk(test_model_dir):
            for file in files:
                if file == 'config.yaml':
                    config_path = os.path.join(root, file)
                    with open(config_path, 'r') as f:
                        config = yaml.load(f, Loader=yaml.FullLoader)
                    break
        args.img_size = config['img_size']['value']
        args.model_name = config['model_name']['value']

        # -------------------------------------------- Show testing options --------------------------------------------
        print('-' * 100)
        print('Testing options:')
        print('Test model name...: {}'.format(args.model_name))
        print('Test model dir....: {}'.format(test_model_dir))
        print('Tested images.....: {}'.format(test_files))
        print('-' * 100)

        # --------------------------------- Model loading and getting of the prediction --------------------------------
        model = self.load_model(model_dir=test_model_dir)
        data_processor = DataProcessor()
        if test_files[0].endswith('avi'):
            img_out = data_processor.process_video(path=test_files,
                                                   img_height=args.img_size[0],
                                                   img_width=args.img_size[1],
                                                   img_channels=args.img_size[2])
            num_images = img_out.shape[0]
            test_ds = tf.data.Dataset.from_tensor_slices(img_out)
            test_ds = test_ds.batch(batch_size=1)
        else:
            # img_out = data_processor.process_images(paths=test_files,
            #                                         img_height=args.img_size[0],
            #                                         img_width=args.img_size[1],
            #                                         img_channels=args.img_size[2])
            test_ds = tf.data.Dataset.from_tensor_slices(test_files)
            test_ds = test_ds.map(map_func=data_processor.map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            test_ds = test_ds.batch(batch_size=1)
            num_images = len(test_files)

        # -------------------------- Generate prediction and process probabilities and label ---------------------------
        mlb = MultiLabelBinarizer(classes=args.label_names)
        mlb.fit(y=args.label_names)
        start = time.time()
        model_probs = model.predict(test_ds, verbose=0)
        pred_time = time.time() - start
        print('Total prediction time.....: {:1.3f} seconds'.format(pred_time))
        print('Average prediction time...: {:1.3f} seconds per image'.format(pred_time / num_images))
        print('Average prediction FPS....: {:1.1f} frames per second'.format(1. / (pred_time / num_images)))
        print('-' * 100)
        return model_probs

    def process_predictions(self, model_output, test_files, thresh_label, thresh_x, thresh_y):
            batch_labels = []
            batch_label_probs = []
            batch_point_coords = []
            start = time.time()

            data_processor = DataProcessor()
            if test_files[0].endswith('avi'):
                batch_images = data_processor.process_video(path=test_files,
                                                            img_height=1000,
                                                            img_width=1000,
                                                            img_channels=3)
            else:
                batch_images = data_processor.process_images(paths=test_files,
                                                             img_height=1000,
                                                             img_width=1000,
                                                             img_channels=3)
            num_files = batch_images.shape[0]
            img_size = [batch_images.shape[1], batch_images.shape[2], batch_images.shape[3]]
            for idx in range(num_files):
                inp_label_probs = model_output[0][idx]
                inp_label_probs = np.expand_dims(inp_label_probs, axis=0)
                inp_point_coords = model_output[1][idx]
                inp_point_coords = np.expand_dims(inp_point_coords, axis=0)

                if 2 * inp_label_probs.shape[1] != inp_point_coords.shape[1]:
                    raise ValueError('Number of classes and coordinates must be equal!')

                # Get a list of the remaining points
                inp_label_bin = inp_label_probs > thresh_label
                x_coords = np.zeros((1, inp_label_probs.shape[1]), dtype=np.float)
                y_coords = np.zeros((1, inp_label_probs.shape[1]), dtype=np.float)
                for i in range(inp_label_probs.shape[1]):
                    x_coords[0, i] = inp_point_coords[0, 2 * i]
                    y_coords[0, i] = inp_point_coords[0, 2 * i + 1]
                x_coords_bin = x_coords > thresh_x
                y_coords_bin = y_coords > thresh_y
                inp_point_bin = np.logical_and(x_coords_bin, y_coords_bin)
                out_bin = np.logical_and(inp_label_bin, inp_point_bin)

                # Get classification labels
                mlb = MultiLabelBinarizer(classes=args.label_names)
                mlb.fit(y=args.label_names)
                out_labels = mlb.inverse_transform(yt=out_bin)
                out_labels = list(out_labels[0])

                # Get classification probabilities
                deleted_points = np.argwhere(out_bin == False)
                deleted_points = list(deleted_points[:, 1])
                out_label_probs = np.delete(inp_label_probs, deleted_points)
                out_label_probs = np.round(out_label_probs, decimals=2)
                out_label_probs = np.expand_dims(out_label_probs, axis=0)

                # Get regression coordinates
                x_coords *= img_size[0]
                y_coords *= img_size[1]
                x_coords = x_coords[out_bin]
                x_coords = np.expand_dims(x_coords, axis=0)
                y_coords = y_coords[out_bin]
                y_coords = np.expand_dims(y_coords, axis=0)
                out_point_coords = np.concatenate((x_coords, y_coords))
                out_point_coords = np.round(out_point_coords, decimals=0)
                out_point_coords = out_point_coords.astype(int)

                batch_labels.append(out_labels)
                batch_label_probs.append(out_label_probs)
                batch_point_coords.append(out_point_coords)
            proc_time = time.time() - start
            print('Total post-processing time.....: {:1.3f} seconds'.format(proc_time))
            print('Average post-processing time...: {:1.3f} seconds per image'.format(proc_time / len(test_files)))
            print('Average post-processing FPS....: {:1.1f} frames per second'.format(1. / (proc_time / len(test_files))))
            print('-' * 100)
            return batch_images, batch_labels, batch_label_probs, batch_point_coords

    def show_predictions(self, images, labels, probs, coords, verbose, save_dir, shape, add_label, add_prob):
        model_dir = os.path.basename(args.test_model_dir)
        save_dir = os.path.join(save_dir, model_dir)
        for idx, (image, label, prob, coord) in enumerate(zip(images, labels, probs, coords)):
            img_path = args.test_files[idx]

            if verbose == 0:
                print('-' * 100)
                print('Image...........: {}'.format(img_path))
                print('Labels..........: {}'.format(label))
                print('Probabilities...: {}'.format(label))
                print('X coordinates...: {}'.format(list(coord[0])))
                print('Y coordinates...: {}'.format(list(coord[1])))
                print('-' * 100)
            elif verbose == 1:
                img_labeled = self.put_points_on_image(image, label, prob, coord,
                                                       shape=shape, add_label=add_label, add_prob=add_prob)
                fig = plt.figure(img_path, figsize=(5, 7))
                plt.imshow(img_labeled, cmap='gray')
                title = '\n\nLabels: {}\n\nProbabilities: {}\n\nX coordinates: {}\n\nY coordinates: {}'\
                    .format(label, list(prob[0]), list(coord[0]), list(coord[1]))
                plt.title(title, fontsize=9)
                fig.show()
                fig.tight_layout()
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                fname = os.path.join(save_dir, os.path.basename(img_path))
                fig.savefig(fname, format='png', quality=100, transparent=False)
                plt.close(fig)
            else:
                raise ValueError('Incorrect VERBOSE value, please check it!')

    def save_to_images(self, images, labels, probs, coords, save_dir, shape, add_label, add_prob):
        model_dir = os.path.basename(args.test_model_dir)
        save_dir = os.path.join(save_dir, model_dir)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        for idx, (image, label, prob, coord) in enumerate(zip(images, labels, probs, coords)):
            img_labeled = self.put_points_on_image(image, label, prob, coord,
                                                   shape=shape, add_label=add_label, add_prob=add_prob)
            save_path = os.path.join(save_dir, os.path.basename(args.test_files[idx]))
            cv2.imwrite(save_path, 255*img_labeled)
        # Debugging only
        # plt.imshow(img_labeled)

    def save_to_video(self, images, probs, labels, coords, save_dir, shape, add_label, add_prob, fps):
        model_dir = os.path.basename(args.test_model_dir)
        save_dir = os.path.join(save_dir, model_dir)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        output_size = (images.shape[1], images.shape[2])
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        video_name = os.path.basename(args.test_files[0])
        video_name = os.path.splitext(video_name)[0]
        video_path = os.path.join(save_dir, video_name + '_prediction.avi')
        video_writer = cv2.VideoWriter(filename=video_path, fourcc=fourcc, fps=fps, frameSize=output_size)
        for idx, (image, label, prob, coord) in enumerate(zip(images, labels, probs, coords)):
            img_labeled = self.put_points_on_image(image, label, prob, coord,
                                                   shape=shape, add_label=add_label, add_prob=add_prob)
            img_labeled_uint8 = (img_labeled*255).astype(np.uint8)
            video_writer.write(img_labeled_uint8)
        video_writer.release()
        # Debugging only
        # plt.imshow(img_labeled)
        # cv2.imwrite('a.png', 255*img_labeled)

    def put_points_on_image(self, img, label, prob, coord, shape, add_label, add_prob):
        img_src = img.copy()
        scale = 0.025
        fontScale = min(img_src.shape[0], img_src.shape[1]) / (25 / scale)
        for idx, point_label in enumerate(label):
            point_prob = prob[0, idx]
            center_coord = (coord[0, idx], coord[1, idx])
            point_color = args.point_colors[point_label]
            if shape == 'circle':
                cv2.circle(img=img_src, center=center_coord, radius=7, color=point_color, thickness=-1)
            elif shape == 'star':
                radius = 7
                line_thick = 2
                alpha = 45 * np.pi/180
                c_x = coord[0, idx]
                c_y = coord[1, idx]
                p0_x = int(np.round(c_x))
                p0_y = int(np.round(c_y - radius))
                p1_x = int(np.round(c_x + np.cos(alpha)*radius))
                p1_y = int(np.round(c_y - np.sin(alpha)*radius))
                p2_x = int(np.round(c_x + radius))
                p2_y = int(np.round(c_y))
                p3_x = int(np.round(c_x + np.cos(alpha)*radius))
                p3_y = int(np.round(c_y + np.sin(alpha)*radius))
                p4_x = int(np.round(c_x))
                p4_y = int(np.round(c_y + radius))
                p5_x = int(np.round(c_x - np.cos(alpha)*radius))
                p5_y = int(np.round(c_y + np.sin(alpha)*radius))
                p6_x = int(np.round(c_x - radius))
                p6_y = int(np.round(c_y))
                p7_x = int(np.round(c_x - np.cos(alpha)*radius))
                p7_y = int(np.round(c_y - np.sin(alpha)*radius))
                cv2.line(img=img_src, pt1=(p0_x, p0_y), pt2=(p4_x, p4_y), color=point_color, thickness=line_thick, lineType=cv2.LINE_AA)
                cv2.line(img=img_src, pt1=(p1_x, p1_y), pt2=(p5_x, p5_y), color=point_color, thickness=line_thick, lineType=cv2.LINE_AA)
                cv2.line(img=img_src, pt1=(p2_x, p2_y), pt2=(p6_x, p6_y), color=point_color, thickness=line_thick, lineType=cv2.LINE_AA)
                cv2.line(img=img_src, pt1=(p3_x, p3_y), pt2=(p7_x, p7_y), color=point_color, thickness=line_thick, lineType=cv2.LINE_AA)
            elif shape == 'square':
                side = 7
                start_point = (coord[0, idx] - side, coord[1, idx] - side)
                end_point = (coord[0, idx] + side, coord[1, idx] + side)
                cv2.rectangle(img=img_src, pt1=start_point, pt2=end_point, color=point_color, thickness=-1)
            else:
                raise ValueError('Incorrect shape value, please check it!')

            if add_label and not add_prob:
                text = point_label
                text_coord = (center_coord[0] + 10, center_coord[1])
                cv2.putText(img=img_src, text=text, org=text_coord, color=point_color,
                            fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=fontScale, thickness=1, lineType=cv2.LINE_AA)
            elif add_prob and not add_label:
                text = str(np.round(100*point_prob).astype(int))
                text_coord = (center_coord[0] + 10, center_coord[1])
                cv2.putText(img=img_src, text=text, org=text_coord, color=point_color,
                            fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=fontScale, thickness=1, lineType=cv2.LINE_AA)
            elif add_prob and add_label:
                text = point_label + '(' + str(np.round(100*point_prob).astype(int)) + ')'
                text_coord = (center_coord[0] + 10, center_coord[1])
                cv2.putText(img=img_src, text=text, org=text_coord, color=point_color,
                            fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=fontScale, thickness=1, lineType=cv2.LINE_AA)
        # Debugging only
        # plt.imshow(img_src)
        return img_src

# ------------------------------------------------------- Handler ------------------------------------------------------
if __name__ == '__main__':
    net = Net()
    if args.mode == 'train':
        net.train_model()
    elif args.mode == 'test':
        model_output = net.test_model(test_model_dir=args.test_model_dir, test_files=args.test_files)
        images, labels, probs, coords = net.process_predictions(model_output=model_output, test_files=args.test_files,
                                                                thresh_label=0.50, thresh_x=0.01, thresh_y=0.01)
        if args.test_files[0].endswith('avi'):
            net.save_to_video(images=images, labels=labels, probs=probs, coords=coords, save_dir='video_predictions',
                              shape='star', add_label=True, add_prob=True, fps=8)
        else:
            net.save_to_images(images=images, labels=labels, probs=probs, coords=coords, save_dir='image_predictions',
                               shape='star', add_label=True, add_prob=True)
            net.show_predictions(images=images, labels=labels, probs=probs, coords=coords, verbose=1, save_dir='plt_predictions',
                                 shape='star', add_label=True, add_prob=False)
    else:
        raise ValueError('Incorrect MODE value, please check it!')
    print('-' * 100)
    print(args.mode.capitalize() + 'ing is finished!')
    print('-' * 100)
