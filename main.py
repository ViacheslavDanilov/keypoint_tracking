import os
import cv2
import time
import wandb
import pandas
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras import layers, models, optimizers, losses, metrics, preprocessing
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from utils import get_random_dcm, f1_loss, macro_f1, get_timing, perfomance_grid

# --------------------------------------------------- Main parameters --------------------------------------------------
MODE = 'test'
MODEL_NAME = 'MobileNet_V2'
BATCH_SIZE = 64
LR = 1e-5
EPOCHS = 2
OPTIMIZER = 'radam'
CLASS_LOSS = 'bce'
CLASS_WEIGHT = 1.
POINT_LOSS = 'mae'
POINT_WEIGHT = 4.

# ------------------------------------------------ Additional parameters -----------------------------------------------
DATA_PATH = 'data/data.xlsx'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
BUFFER_SIZE = 1000
IS_TRAINABLE = False
VERBOSE = 1
TEST_IMAGES = ['data/img/001_025.png', 'data/img/002_028.png', 'data/img/003_032.png']
TEST_MODEL_DIR = 'models/MobileNet_V2_2703_2250'
CLASS_NAMES = ['AA1', 'AA2', 'STJ1', 'STJ2', 'CD', 'CM', 'CP', 'CT', 'PT', 'FE1', 'FE2']
POINT_NAMES = ['AA1_x', 'AA1_y', 'AA2_x', 'AA2_y', 'STJ1_x', 'STJ1_y', 'STJ2_x', 'STJ2_y', 'CD_x', 'CD_y',
               'CM_x', 'CM_y', 'CP_x', 'CP_y', 'CT_x', 'CT_y', 'PT_x', 'PT_y', 'FE1_x', 'FE1_y', 'FE2_x', 'FE2_y']
if 'MobileNet_V2' or 'ResNet_V2' in MODEL_NAME:
    IMG_SIZE = (224, 224, 3)
elif 'Inception_V3' or 'Inception_ResNet_v2' in MODEL_NAME:
    IMG_SIZE = (299, 299, 3)
elif 'EfficientNet_B7' in MODEL_NAME:
    IMG_SIZE = (600, 600, 3)
else:
    raise ValueError('Incorrect MODEL_NAME, please change it!')

# -------------------------------------------- Initialize ArgParse container -------------------------------------------
parser = argparse.ArgumentParser(description='Keypoint tracking and classification')
# Train options
parser.add_argument('-mo', '--mode', metavar='', default=MODE, type=str, help='train or test')
parser.add_argument('-mn', '--model_name', metavar='', default=MODEL_NAME, type=str, help='architecture of the model')
parser.add_argument('-opt', '--optimizer', metavar='', default=OPTIMIZER, type=str, help='type of an optimizer')
parser.add_argument('-clo', '--class_loss', metavar='', default=CLASS_LOSS, type=str, help='classification loss')
parser.add_argument('-plo', '--point_loss', metavar='', default=POINT_LOSS, type=str, help='regression loss')
parser.add_argument('-cw', '--class_weight', metavar='', default=CLASS_WEIGHT, type=float, help='class loss weight')
parser.add_argument('-pw', '--point_weight', metavar='', default=POINT_WEIGHT, type=float, help='point loss weight')
parser.add_argument('-lr', '--learning_rate', metavar='', default=LR, type=float, help='learning rate')
parser.add_argument('-bas', '--batch_size', metavar='', default=BATCH_SIZE, type=int, help='batch size')
parser.add_argument('-ep', '--epochs', metavar='', default=EPOCHS, type=int, help='number of epochs for training')
parser.add_argument('-bus', '--buffer_size', metavar='', default=BUFFER_SIZE, type=int, help='buffer size')
parser.add_argument('-ist', '--is_trainable', action='store_true', default=IS_TRAINABLE, help='whether to train backbone')
# Test options
parser.add_argument('-ti', '--test_images', nargs='+', metavar='', default=TEST_IMAGES, type=str, help='list of tested images')
# parser.add_argument('-tmd', '--test_model_dir', metavar='', default='models/' + parser.parse_args().model_name, type=str,
#                     help='model directory for testing mode')
parser.add_argument('-tmd', '--test_model_dir', metavar='', default=TEST_MODEL_DIR, type=str, help='model directory for testing mode')
parser.add_argument('-ver', '--verbose', metavar='', default=VERBOSE, type=int, help='verbosity mode')
# Additional arguments
parser.add_argument('--class_names', metavar='', default=CLASS_NAMES, type=list, help='list of class names')
parser.add_argument('--point_names', metavar='', default=POINT_NAMES, type=list, help='list of point names')
parser.add_argument('--img_size', metavar='', default=IMG_SIZE, type=int, help='image size')
args = parser.parse_args()

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
        """Function that returns a list of image names, array of labels and array of coordinates.
        Args:
            path_to_data: string representing path to xlsx dataset info
        """
        # TODO: delete nrows for the full dataset
        source_df = pandas.read_excel(path_to_data, index_col=None, na_values=['NA'], usecols="B:AM", nrows=1752)
        path_df = source_df['Path']
        class_df = source_df[args.class_names]
        point_df = source_df[args.point_names]
        # Used for debugging
        # paths = path_df[0:5]
        ds_path = os.path.join('data', 'img_inputs_' + str(args.img_size[0]) + '.npy')
        if os.path.isfile(ds_path):
            img_inputs = np.load(ds_path)
        else:
            img_inputs = self.process_images(paths=path_df)
            np.save(ds_path, img_inputs)
        class_targets = class_df.to_numpy()
        point_targets = point_df.to_numpy() / 1000
        return img_inputs, class_targets, point_targets

    def process_images(self, paths):
        img_inputs = np.ndarray(shape=(len(paths), args.img_size[0], args.img_size[1], args.img_size[2]),
                                dtype=np.float32)
        idx = 0
        for path in tqdm(paths):
            img_string = tf.io.read_file(path)
            img_input = tf.image.decode_png(img_string, channels=args.img_size[2])
            img_resized = tf.image.resize(images=img_input, size=(args.img_size[0], args.img_size[1]))
            img_norm = img_resized / 255.0
            img_inputs[idx] = img_norm
            idx += 1
            # Used for debugging
            # self.visualize(img_input, img_output)
            # self.visualize(img_norm, img_output)
        return img_inputs

    def create_dataset(self, img_inputs, class_targets, point_targets, is_caching=None, cache_name=None):
        """Load and parse a dataset.
        Args:
            img_inputs: list of image paths
            class_targets: numpy array of labels
            point_targets: numpy array of coordinates
            is_caching: boolean to indicate caching mode
            cache_name: name of data cache file
        """
        # Create a first dataset of file images, labels and coordinates
        # dataset = tf.data.Dataset.from_tensor_slices(({'img_input': img_inputs},
        #                                               {'class': class_targets, 'point': point_targets}))
        dataset = tf.data.Dataset.from_tensor_slices((img_inputs, (class_targets, point_targets)))

        # Used for debugging
        # temp_1 = self.parse_image(paths[1], class_targets[1], point_targets[1])
        # temp_2 = dataset.element_spec

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
        with open(os.path.join(args.model_dir, 'architecture.json'), 'w') as f:
            f.write(model.to_json())
        end = time.time()
        img_path = os.path.join(args.model_dir, args.model_name + '.png')
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
        elif args.model_name == 'EfficientNet_B7':
            model_url = 'https://tfhub.dev/google/efficientnet/b7/feature-vector/1'
        else:
            raise ValueError('Incorrect MODEL_TYPE value!')

        # -------------------------------------------- Building the model ----------------------------------------------
        img_input = layers.Input(shape=(args.img_size[0], args.img_size[1], args.img_size[2]), name='img_input')
        hub_layer = hub.KerasLayer(handle=model_url, trainable=args.is_trainable, name=args.model_name)
        output = hub_layer(img_input)
        class_layer = layers.Dense(1024, activation='relu', name='classifier')(output)
        class_output = layers.Dense(len(args.class_names), activation='sigmoid', name='class')(class_layer)
        point_layer = layers.Dense(1024, activation='relu', name='regressor')(output)
        point_output = layers.Dense(len(args.point_names), activation='sigmoid', name='point')(point_layer)
        model = models.Model(inputs=img_input, outputs=[class_output, point_output], name=args.model_name)
        model.summary()

        # -------------------------------------------- Compiling the model ---------------------------------------------
        if args.class_loss == 'f1':
            class_loss = f1_loss
        elif args.class_loss == 'bce':
            class_loss = losses.BinaryCrossentropy(from_logits=False, label_smoothing=0)
        else:
            raise ValueError('Incorrect CLASS_LOSS value, please change it!')

        if args.point_loss == 'mae':
            point_loss = losses.MeanAbsoluteError()
        elif args.point_loss == 'mse':
            point_loss = losses.MeanSquaredError()
        else:
            raise ValueError('Incorrect POINT_LOSS value, please change it!')

        class_metrics = [macro_f1,
                         tfa.metrics.F1Score(num_classes=len(args.class_names), average='micro', threshold=0.5, name='micro_f1'),
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
                      loss=[class_loss, point_loss],
                      metrics=[class_metrics, point_metrics],
                      loss_weights=[args.class_weight, args.point_weight])
        # model.compile(optimizer=self.get_optimizer(args.optimizer, args.learning_rate),
        #               loss={'class': class_loss, 'point': point_loss},
        #               metrics={'class': class_metrics, 'point': point_metrics},
        #               loss_weights={'class': args.class_weight, 'point': args.point_weight})
        return model

    def train_model(self):
        # -------------------------------------- Data processing and prefetching ---------------------------------------
        data_processor = DataProcessor()
        img_inputs, class_targets, point_targets = data_processor.get_inputs_and_targets(path_to_data=DATA_PATH)
        X_train_class, X_val_class, y_train, y_val = train_test_split(img_inputs, class_targets,
                                                                      shuffle=True, test_size=0.2, random_state=11)
        X_train_point, X_val_point, z_train, z_val = train_test_split(img_inputs, point_targets,
                                                                      shuffle=True, test_size=0.2, random_state=11)
        if np.array_equal(X_train_class, X_train_point) and np.array_equal(X_val_class, X_val_point):
            X_train = X_train_class
            X_val = X_val_class
        else:
            raise ValueError('Inputs for classification and regression subsets are not equal!')

        print('-' * 100)
        print("Number of images for training.....: {} ({:.1%})".format(len(X_train),
                                                                       round(len(X_train)/(len(X_train)+len(X_val)), 1)))
        print("Number of images for validation...: {} ({:.1%})".format(len(X_val),
                                                                       round(len(X_val)/(len(X_train)+len(X_val)), 1)))
        print('-' * 100)

        train_ds = data_processor.create_dataset(X_train, y_train, z_train,
                                                 is_caching=True, cache_name=None)
        val_ds = data_processor.create_dataset(X_val, y_val, z_val,
                                               is_caching=True, cache_name=None)
        for image, target in train_ds.take(1):
            print('-' * 100)
            print("Image batch shape...: {}".format(image.numpy().shape))
            print("Class batch shape...: {}".format(target[0].numpy().shape))
            print("Point batch shape...: {}".format(target[1].numpy().shape))
            print('-' * 100)

        # ------------------------------------------  Initialize W&B project -------------------------------------------
        run_time = datetime.now().strftime("%d%m_%H%M")
        run_name = args.model_name + '_{}'.format(run_time)
        args.model_dir = os.path.join('models', run_name)
        os.makedirs(args.model_dir)

        params = dict(img_size=(args.img_size[0], args.img_size[1], args.img_size[2]),
                      model_name=args.model_name,
                      model_dir=args.model_dir,
                      optimizer=args.optimizer,
                      class_loss=args.class_loss,
                      point_loss=args.point_loss,
                      batch_size=args.batch_size,
                      buffer_size=args.buffer_size,
                      epochs=args.epochs,
                      learning_rate=args.learning_rate,
                      trainable_backbone=args.is_trainable)
        wandb.init(project='temp', dir=args.model_dir, name=run_name, sync_tensorboard=True, config=params)
        wandb.run.id = wandb.run.id
        wandb.config.update(params)

        # -------------------------------------------  Initialize callbacks --------------------------------------------
        csv_logger = CSVLogger(os.path.join(args.model_dir, 'logs.csv'), separator=',', append=False)
        check_pointer = ModelCheckpoint(filepath=os.path.join(args.model_dir, 'weights.h5'),
                                        monitor='val_loss',
                                        save_best_only=True,
                                        save_weights_only=False,
                                        mode='min',
                                        verbose=1)
        wb_logger = wandb.keras.WandbCallback(monitor='val_loss',
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

        # ------------------------------------------- Show training options --------------------------------------------
        print('-' * 100)
        print('Training options:')
        print('Model name............: {}'.format(args.model_name))
        print('Model directory.......: {}'.format(args.model_dir))
        print('Classification loss...: {} (weight: {})'.format(args.class_loss, args.class_weight))
        print('Regression loss.......: {} (weight: {})'.format(args.point_loss, args.class_weight))
        print('Optimizer.............: {}'.format(args.optimizer))
        print('Learning rate.........: {}'.format(args.learning_rate))
        print('Batch size............: {}'.format(args.batch_size))
        print('Epochs................: {}'.format(args.epochs))
        print('Trainable backbone....: {}'.format(args.is_trainable))
        print('Image dimensions......: {}x{}x{}'.format(args.img_size[0], args.img_size[1], args.img_size[2]))
        print('Buffer size...........: {}'.format(args.buffer_size))
        print('-' * 100)

        # --------------------------------------- Get model and then train it ------------------------------------------
        model = self.build_model()
        self.save_model(model=model)

        # Check model's operability
        for batch in val_ds:
            print('-' * 100)
            print("Model operability test")
            print("Class labels prediction: {}".format(np.around(model.predict(batch)[0][0], decimals=2)))
            print("Point coordinates prediction: {}".format(np.around(model.predict(batch)[1][0], decimals=2)))
            print('-' * 100)
            break

        start = time.time()
        # history = model.fit(x=X_train, y=[y_train, z_train],
        #                     batch_size=args.batch_size,
        #                     epochs=args.epochs,
        #                     validation_data=(X_val, [y_val, z_val]),
        #                     callbacks=[csv_logger, wb_logger, check_pointer, earlystop])
        history = model.fit(x=train_ds,
                            epochs=args.epochs,
                            validation_data=val_ds,
                            callbacks=[csv_logger, wb_logger, check_pointer, earlystop])
        end = time.time()
        print('\nTraining of the model took: {}'.format(get_timing(end - start)))

        # ------------------------------------ Show training and validation outputs ------------------------------------
        class_loss, val_class_loss = history.history['class_loss'], history.history['val_class_loss']
        point_loss, val_point_loss = history.history['point_loss'], history.history['val_point_loss']
        accuracy, val_accuracy = history.history['class_accuracy'], history.history['val_class_accuracy']
        macro_f1, val_macro_f1 = history.history['class_macro_f1'], history.history['val_class_macro_f1']
        micro_f1, val_micro_f1 = history.history['class_micro_f1'], history.history['val_class_micro_f1']
        precision, val_precision = history.history['class_precision'], history.history['val_class_precision']
        recall, val_recall = history.history['class_recall'], history.history['val_class_recall']
        mae, val_mae = history.history['point_mae'], history.history['val_point_mae']
        rmse, val_rmse = history.history['point_rmse'], history.history['val_point_rmse']
        mse, val_mse = history.history['point_mse'], history.history['val_point_mse']

        print('-' * 100)
        print('Class prediction on training / validation')
        print("Loss........: {:.2f} / {:.2f}".format(class_loss[-1], val_class_loss[-1]))
        print("Accuracy....: {:.2f} / {:.2f}".format(accuracy[-1], val_accuracy[-1]))
        print("Macro F1....: {:.2f} / {:.2f}".format(macro_f1[-1], val_macro_f1[-1]))
        print("Micro F1....: {:.2f} / {:.2f}".format(micro_f1[-1], val_micro_f1[-1]))
        print("Precision...: {:.2f} / {:.2f}".format(precision[-1], val_precision[-1]))
        print("Recall......: {:.2f} / {:.2f}".format(recall[-1], val_recall[-1]))
        print('\nPoint tracking on training / validation')
        print("Loss........: {:.2f} / {:.2f}".format(point_loss[-1], val_point_loss[-1]))
        print("MAE.........: {:.2f} / {:.2f}".format(mae[-1], val_mae[-1]))
        print("RMSE........: {:.2f} / {:.2f}".format(rmse[-1], val_rmse[-1]))
        print("MSE.........: {:.2f} / {:.2f}".format(mse[-1], val_mse[-1]))
        print('-' * 100)

        # --------------------- Performance table of the model with different levels of threshold ----------------------
        perfomance_grid(ds=val_ds, target=y_val, label_names=args.class_names, model=model, save_dir=args.model_dir)

    def test_model(self):
        # -------------------------------------------- Show testing options --------------------------------------------
        print('-' * 100)
        print('Testing options:')
        print('Test model name...: {}'.format(args.model_name))
        print('Test model dir....: {}'.format(args.test_model_dir))
        print('Tested images.....: {}'.format(args.test_images))
        print('Verbosity mode....: {}'.format(args.verbose))
        print('-' * 100)

        # --------------------------------- Model loading and getting of the prediction --------------------------------
        model = self.load_model(model_dir=args.test_model_dir)
        data_processor = DataProcessor()
        img_out = data_processor.process_images(paths=args.test_images)

        # Generate prediction and process probabilities and labels
        mlb = MultiLabelBinarizer(classes=args.class_names)
        mlb.fit(y=args.class_names)
        start = time.time()
        model_probs = model.predict(img_out)
        # TODO: решить проблему оценки времени предсказания для батча
        pred_time = time.time() - start
        print('Average prediction time...: {:1.2f} seconds per image'.format(pred_time / len(args.test_images)))
        print('Average FPS...............: {:1.1f} frames per second'.format(1. / (pred_time / len(args.test_images))))
        print('-' * 100)
        return model_probs

    def show_predictions(self, model_probs, thresh):
        for idx, img_path in enumerate(args.test_images):
            img_string = tf.io.read_file(img_path)
            img_source = tf.image.decode_png(img_string, channels=args.img_size[2])
            img_source = img_source.numpy()
            img_size = img_source.shape

            inp_class_probs = model_probs[0][idx]
            inp_class_probs = np.expand_dims(inp_class_probs, axis=0)
            inp_point_probs = model_probs[1][idx]
            inp_point_probs = np.expand_dims(inp_point_probs, axis=0)
            out_class_probs, labels, out_point_probs = self.process_probs(inp_class_probs, inp_point_probs, thresh, img_size)
            img_labeled = self.put_points_on_image(img_source, labels, out_point_probs)

            # Show image, predicted coordinates and label
            if args.verbose == 0:
                print('-' * 100)
                print('Image.................: {}'.format(img_path))
                print('Threshold.............: {}'.format(thresh))
                print('Predicted labels......: {}'.format(labels))
                print('Label probabilities...: {}'.format(out_class_probs.flatten()))
                print('X coordinates.........: {}'.format(out_point_probs[0].flatten()))
                print('Y coordinates.........: {}'.format(out_point_probs[1].flatten()))
                print('-' * 100)
            elif args.verbose == 1:
                fig = plt.figure(img_path, figsize=(5, 7))
                plt.imshow(img_labeled, cmap='gray')
                plt.title('\n\nPredicted labels: {}\n\nLabel probabilities: {}'
                          '\n\nX coordinates: {}\n\nY coordinates: {}'
                          .format(labels, out_class_probs.flatten(),
                                  out_point_probs[0].flatten(), out_point_probs[1].flatten()), fontsize=10)
                plt.show()
                fig.tight_layout()
            elif args.verbose == 2:
                # TODO: добавить для оценки точек предсказанные и исходные точки
                print('TO BE RESOLVED')
            else:
                raise ValueError('Incorrect VERBOSE value!')

    def process_probs(self, inp_class_probs, inp_point_probs, thresh, source_img_size):
        if 2*inp_class_probs.shape[1] != inp_point_probs.shape[1]:
            raise ValueError('Number of classes and coordinates must be equal!')
        mlb = MultiLabelBinarizer(classes=args.class_names)
        mlb.fit(y=args.class_names)

        # Get classification labels
        inp_class_bin = inp_class_probs > thresh
        out_class_labels = mlb.inverse_transform(yt=inp_class_bin)
        out_class_labels = list(out_class_labels[0])

        # Get classification probabilities
        deleted_points = np.argwhere(inp_class_bin == False)
        deleted_points = list(deleted_points[:, 1])
        out_class_probs = np.delete(inp_class_probs, deleted_points)
        out_class_probs = np.round(out_class_probs, decimals=2)
        out_class_probs = np.expand_dims(out_class_probs, axis=0)

        # Get regression probabilities
        x_probs = np.zeros((1, inp_class_probs.shape[1]), dtype=np.float)
        y_probs = np.zeros((1, inp_class_probs.shape[1]), dtype=np.float)
        for i in range(inp_class_probs.shape[1]):
            x_probs[0, i] = inp_point_probs[0, 2*i]
            x_probs[0, i] *= source_img_size[0]
            y_probs[0, i] = inp_point_probs[0, 2*i + 1]
            y_probs[0, i] *= source_img_size[1]
        x_probs = x_probs[inp_class_bin]
        x_probs = np.expand_dims(x_probs, axis=0)
        y_probs = y_probs[inp_class_bin]
        y_probs = np.expand_dims(y_probs, axis=0)
        out_point_probs = np.concatenate((x_probs, y_probs))
        out_point_probs = np.round(out_point_probs, decimals=0)
        out_point_probs = out_point_probs.astype(int)
        return out_class_probs, out_class_labels, out_point_probs

    def put_points_on_image(self, img, labels, coords):
        radius = 10
        thickness = -1
        colors = {'AA1': (247, 6, 0), 'AA2': (253, 1, 246), 'STJ1': (1, 253, 132), 'STJ2': (0, 16, 247),
                  'CD': (255, 247, 0), 'CM': (0, 250, 247), 'CP': (249, 157, 0), 'CT': (253, 253, 253),
                  'PT': (64, 53, 182), 'FE1': (107, 123, 245), 'FE2': (216, 36, 240)}
        for idx, point_label in enumerate(labels):
            point_color = colors[point_label]
            point_coord = (coords[0, idx], coords[1, idx])
            img = cv2.circle(img, point_coord, radius, point_color, thickness)
        # Used for debugging
        # plt.imshow(img)
        return img

# ------------------------------------------------------- Handler ------------------------------------------------------
if __name__ == '__main__':
    net = Net()
    if args.mode == 'train':
        net.train_model()
    elif args.mode == 'test':
        model_probs = net.test_model()
        net.show_predictions(model_probs, thresh=0.5)
    else:
        raise ValueError('Incorrect MODE value!')
    print('-' * 100)
    print(args.mode.capitalize() + 'ing is finished!')
    print('-' * 100)