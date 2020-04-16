import os
import cv2
import json
import wandb
import random
import pandas
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import matplotlib.style as style
from sklearn.metrics import f1_score

def convert_images_to_video(model_dir, images_prefix, add_note, fps, save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    images_dir = os.path.join('models', model_dir, 'predictions_per_epoch')
    image_paths = glob(images_dir + '/*' + images_prefix + '_*')
    image_paths.sort()
    image = cv2.imread(image_paths[0])

    note_height = 150
    if add_note:
        frame_size = (image.shape[0], image.shape[1] + note_height)
        logs_path = os.path.join('models', model_dir, 'logs.csv')
        cols_to_use = ['total_loss', 'label_macro_f1', 'point_mae']
        logs = pandas.read_csv(logs_path, usecols=cols_to_use)[cols_to_use]
        logs = logs.to_numpy()
    else:
        frame_size = (image.shape[0], image.shape[1])

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video_path = os.path.join(save_dir, images_prefix + '_training.avi')
    video_writer = cv2.VideoWriter(filename=video_path, fourcc=fourcc, fps=fps, frameSize=frame_size)
    for idx, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        if add_note:
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 1
            thickness = 1
            note = 255 * np.ones(shape=(note_height, image.shape[1], image.shape[2]), dtype=np.uint8)
            text_1 = "Model: {}".format(model_dir)
            text_size_1 = cv2.getTextSize(text_1, font, font_scale, thickness)[0]
            text_1_x = (note.shape[1] - text_size_1[0]) // 2
            text_1_y = (note.shape[0] + text_size_1[1]) // 2 - 45

            text_2 = "Loss: {:.2f} | F1: {:.2f} | MAE: {:.2f}".format(logs[idx, 0], logs[idx, 1], logs[idx, 2])
            text_size_2 = cv2.getTextSize(text_2, font, font_scale, thickness)[0]
            text_2_x = (note.shape[1] - text_size_2[0]) // 2
            text_2_y = (note.shape[0] + text_size_2[1]) // 2

            text_3 = "Epoch: {}".format(str(idx).zfill(3))
            text_size_3 = cv2.getTextSize(text_3, font, font_scale, thickness)[0]
            text_3_x = (note.shape[1] - text_size_3[0]) // 2
            text_3_y = (note.shape[0] + text_size_3[1]) // 2 + 45

            cv2.putText(img=note, text=text_1, org=(text_1_x, text_1_y), color=(0, 0, 0),
                        fontFace=font, fontScale=font_scale, thickness=thickness, lineType=cv2.LINE_AA)
            cv2.putText(img=note, text=text_2, org=(text_2_x, text_2_y), color=(0, 0, 0),
                        fontFace=font, fontScale=font_scale, thickness=thickness, lineType=cv2.LINE_AA)
            cv2.putText(img=note, text=text_3, org=(text_3_x, text_3_y), color=(0, 0, 0),
                        fontFace=font, fontScale=font_scale, thickness=thickness, lineType=cv2.LINE_AA)
            image = np.vstack((note, image))
            # image = np.concatenate((note, image), axis=0)
            # plt.imshow(image)
        video_writer.write(image)
    video_writer.release()
    print('Video converted to {}'.format(video_path))

def convert_json_to_xlsx(ann_dir, img_dir, save_dir):
    ann_paths = glob(ann_dir + "/*.json")
    ann_paths.sort()
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    column_order = ['Name', 'Path', 'Height', 'Width', 'Points',
                    'AA1', 'AA2', 'STJ1', 'STJ2', 'CD', 'CM', 'CP', 'CT', 'PT', 'FE1', 'FE2',
                    'AA1_x', 'AA1_y', 'AA2_x', 'AA2_y', 'STJ1_x', 'STJ1_y', 'STJ2_x', 'STJ2_y',
                    'CD_x', 'CD_y', 'CM_x', 'CM_y', 'CP_x', 'CP_y', 'CT_x', 'CT_y',
                    'PT_x', 'PT_y', 'FE1_x', 'FE1_y', 'FE2_x', 'FE2_y']

    ann_df = pandas.DataFrame(columns=column_order)
    idx = 0
    for _, ann_path in tqdm(enumerate(ann_paths), unit=' json files'):
        with open(ann_path) as f:
            json_data = json.load(f)
        num_points = len(json_data['objects'])
        if num_points > 0:
            json_name = os.path.basename(ann_path)
            img_name = os.path.splitext(json_name)[0]
            img_path = os.path.join(img_dir, img_name)
            img_path = os.path.normpath(img_path)
            height = json_data['size']['height']
            width = json_data['size']['width']

            ann_df.loc[idx, 0:len(column_order)] = 0
            ann_df.loc[idx, 'Name'] = img_name
            ann_df.loc[idx, 'Path'] = img_path
            ann_df.loc[idx, 'Height'] = height
            ann_df.loc[idx, 'Width'] = width
            ann_df.loc[idx, 'Points'] = num_points

            for point_idx in range(num_points):
                point_label = json_data['objects'][point_idx]['classTitle']
                point_x = json_data['objects'][point_idx]['points']['exterior'][0][0]
                point_y = json_data['objects'][point_idx]['points']['exterior'][0][1]
                ann_df.loc[idx, point_label] = 1
                ann_df.loc[idx, point_label + '_x'] = point_x
                ann_df.loc[idx, point_label + '_y'] = point_y
            idx += 1

    xlsx_name = os.path.join(save_dir, 'data.xlsx')
    ann_df.to_excel(xlsx_name, sheet_name='Data', index=True, startrow=0, startcol=0)
    print('JSON files converted to XLSX file to {}'.format(xlsx_name))

def extract_images_from_video(video_dir, output_dims, save_freq, save_dir):
    file_names = os.listdir(video_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pbar = tqdm(file_names)
    for file_name in pbar:
        i = 0
        times = 0
        cap = cv2.VideoCapture(os.path.join(video_dir, file_name))
        video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        while True:
                ret, frame = cap.read()
                if ret == True:
                    times += 1
                    if times % save_freq == 0:
                        if video_height != video_width:
                            frame = frame[0:896, 485:1380]
                        frame = cv2.resize(frame, output_dims, interpolation=cv2.INTER_AREA)
                        video_name = os.path.splitext(file_name)[0]
                        image_path = os.path.join(save_dir, video_name + '_' + str(i + 1).zfill(3) + '.png')
                        cv2.imwrite(image_path, frame)
                        i += 1
                else:
                    break
        pbar.set_description("Processing %s" % file_name)
        print('\n{0:d} images saved for {1:s}'.format(i, file_name))

def get_random_img(dir):
    img = os.path.join(dir, random.choice(os.listdir(dir)))
    while os.path.isdir(img):
        img = os.path.join(img, random.choice(os.listdir(img)))
    return img

def macro_f1(y_true, y_pred, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across label)
    Args:
        y_true (int32 Tensor): label array
        y_pred (float32 Tensor): probability matrix from forward propagation
        thresh: probability value above which a model predicts positive

    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_pred, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y_true, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y_true), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y_true, axis=0), tf.float32)
    f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1

def f1_loss(y_true, y_pred):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all label).
    Use probability values instead of binary predictions.

    Args:
        y_true (int32 Tensor): targets array
        y_pred (float32 Tensor): probability matrix from forward propagation

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    tp = tf.reduce_sum(y_pred * y_true, axis=0)
    fp = tf.reduce_sum(y_pred * (1 - y_true), axis=0)
    fn = tf.reduce_sum((1 - y_pred) * y_true, axis=0)
    soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    cost = 1 - soft_f1                              # reduce (1 - f1) in order to increase f1
    macro_cost = tf.reduce_mean(cost)               # average on all label
    return macro_cost

# def micro_f1_new(y_true, y_pred, thresh_class=0.5):
#     f1 = tfa.metrics.F1Score(num_classes=3, average='micro', name='Privet')
#     y_true = tf.cast(y_true, dtype=tf.int32)
#     y_pred = tf.cast(tf.greater(y_pred, thresh_class), tf.int32)
#     y_true = tf.one_hot(y_true, depth=2)
#     y_pred = tf.one_hot(y_pred, depth=2)
#     f1.update_state(y_true, y_pred)
#     # print('F1 Score is: ', f1.result().numpy())
#     return f1.result().numpy()
#
# def micro_f1_new_scikit(y_true, y_pred, thresh_class=0.5):
#     y_pred = tf.keras.backend.get_value(y_pred)
#     y_pred = (y_pred > thresh_class).astype('int')
#     y_true = tf.keras.backend.get_value(y_true)
#     # y_true = y_true.numpy()
#     y_true = y_true.astype('int')
#     f1 = f1_score(y_true, y_pred, average='micro')
#     return f1

def get_timing(sec):
    """Function that converts time period in seconds into %h:%m:%s expression.
    Args:
        sec (float): time period in seconds
    Returns:
        output (string): formatted time period
    """
    sec = int(sec)
    h = sec // 3600
    m = sec % 3600 // 60
    s = sec % 3600 % 60
    output = '{:02d}h:{:02d}m:{:02d}s'.format(h, m, s)
    return output

def learning_curves(history, fname):
    """Plot the learning curves of loss and macro f1 score
    for the training and validation datasets.

    Args:
        history: history callback of fitting a tensorflow model
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    macro_f1 = history.history['macro_f1']
    val_macro_f1 = history.history['val_macro_f1']

    epochs = len(loss)

    style.use("seaborn-whitegrid")
    plt.figure(figsize=(12, 9))

    plt.subplot(2, 1, 1)
    plt.plot(range(1, epochs + 1), loss, label='Loss (Training)')
    plt.plot(range(1, epochs + 1), val_loss, label='Loss (Validation)')
    plt.ylim(0, 1)
    plt.ylabel('Loss')
    plt.title('Loss Dynamics')
    plt.legend(loc='upper right', frameon=True)

    plt.subplot(2, 1, 2)
    plt.plot(range(1, epochs + 1), macro_f1, label='Macro F1-score (Training)')
    plt.plot(range(1, epochs + 1), val_macro_f1, label='Macro F1-score (Validation)')
    plt.ylim(0, 1)
    plt.ylabel('Macro F1-score')
    plt.title('Macro F1-score Dynamics')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right', frameon=True)
    plt.savefig(fname=fname, dpi=300, quality=100,
                bbox_inches='tight', pad_inches=0.05,
                facecolor='w', edgecolor='w', orientation='landscape')
    # plt.show()
    return loss, val_loss, macro_f1, val_macro_f1

def perfomance_grid(ds, target, label_names, model, save_dir, n_thresh=100):
    """Computes the performance table containing target, label names,
    label frequencies, thresholds between 0 and 1, number of tp, fp, fn, tn,
    precision, recall and f-score metrics for each label.

    Args:
        ds (tf.data.Datatset): contains the features array
        target (numpy array): target matrix of shape (BATCH_SIZE, N_LABELS)
        label_names (list of strings): column names in target matrix
        model (tensorflow keras model): model to use for prediction
        n_thresh (int): number of thresholds to compute
        save_dir (str): path where the perfomance xlsx file is saved
    Returns:
        grid (Pandas dataframe): performance table
    """

    # Get model predictions
    y_pred_vals = model.predict(ds)
    y_pred_vals = y_pred_vals[0]
    # Define target matrix
    y_true_vals = target
    # Find label frequencies in the validation set
    label_freq = target.sum(axis=0)
    # Get label indexes
    label_index = [i for i in range(len(label_names))]
    # Define thresholds
    thresholds = np.linspace(0, 1, n_thresh + 1).astype(np.float32)

    # Compute all metrics for all label
    ids, labels, freqs, tps, tns, fps, fns, precisions, recalls, f1s = [], [], [], [], [], [], [], [], [], []
    for idx in label_index:
        for thresh in thresholds:
            ids.append(idx)
            labels.append(label_names[idx])
            freqs.append(round(label_freq[idx] / len(y_true_vals), 2))
            y_pred = y_pred_vals[:, idx]
            y_true = y_true_vals[:, idx]
            y_pred = y_pred > thresh
            tp = np.count_nonzero(y_pred * y_true)
            tn = np.count_nonzero((1 - y_pred)*(1 - y_true))
            fp = np.count_nonzero(y_pred * (1 - y_true))
            fn = np.count_nonzero((1 - y_pred) * y_true)
            precision = tp / (tp + fp + 1e-16)
            recall = tp / (tp + fn + 1e-16)
            f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
            tps.append(tp)
            tns.append(tn)
            fps.append(fp)
            fns.append(fn)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

    # Create the performance dataframe
    grid = pd.DataFrame({
        'ID': ids,
        'Label': labels,
        'Frequency': freqs,
        'Threshold': list(thresholds) * len(label_index),
        'TP': tps,
        'TN': tns,
        'FP': fps,
        'FN': fns,
        'Precision': precisions,
        'Recall': recalls,
        'F1': f1s})

    grid = grid[['ID', 'Label', 'Frequency', 'Threshold', 'TP', 'TN', 'FN', 'FP', 'Precision', 'Recall', 'F1']]
    grid_name = os.path.join(save_dir, 'perfomance_grid.xlsx')
    grid.to_excel(grid_name, sheet_name='Perfomance', index=True, startrow=0, startcol=0)
    return grid

def pull_video_to_wandb(project, run_name, video_dir, fps, prefix, models_dir = 'models'):
    run_folder = os.path.join(models_dir, run_name, 'wandb')
    dir_list = os.listdir(run_folder)
    dir_list.sort()
    run_id = dir_list[1]
    run_id = run_id.split('-')
    run_id = run_id[2]
    wandb.init(entity='viacheslav_danilov',
               project=project,
               id=run_id,
               name=run_name,
               resume=True)
    video_paths = glob(os.path.join(video_dir, run_name) + '/*.avi')
    for idx, video_path in enumerate(video_paths):
        video = convert_video_to_ndarray(video_path)
        # TODO: Checkpoint
        wandb.log({prefix + str(idx+1) + '_' + run_name: wandb.Video(data_or_path=video, fps=fps, format='mp4')}, commit=False)
    print('Pulling to WANDB complete!')

def convert_video_to_ndarray(video_path):
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video = np.empty(shape=(num_frames, frame_height, frame_width, 3), dtype=np.dtype('uint8'))
    frame_counter = 0
    ret = True
    while (frame_counter < num_frames and ret):
        ret, video[frame_counter] = cap.read()
        frame_counter += 1
    cap.release()
    video = np.swapaxes(video, axis1=1, axis2=3)
    video = np.swapaxes(video, axis1=2, axis2=3)
    return video

def crop_video(input_path, y1, y2, x1, x2, output_dims, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Video reader
    cap = cv2.VideoCapture(input_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video = np.empty(shape=(num_frames, frame_height, frame_width, 3), dtype=np.dtype('uint8'))

    # Video writer
    basename = os.path.split(input_path)[-1]
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video_path = os.path.join(save_dir, basename)
    video_writer = cv2.VideoWriter(filename=video_path, fourcc=fourcc, fps=fps, frameSize=output_dims)

    frame_counter = 0
    ret = True
    while (frame_counter < num_frames and ret):
        ret, frame = cap.read()
        # frame = frame[0:896, 485:1380]
        frame = frame[y1:y2, x1:x2]
        frame = cv2.resize(frame, output_dims, interpolation=cv2.INTER_AREA)
        video_writer.write(frame)
        frame_counter += 1
    video_writer.release()
    cap.release()


# ------------------------------------------------------- Handler ------------------------------------------------------
if __name__ == '__main__':
    # Extract images from video
    # extract_images_from_video(video_dir='data/video2', output_dims=(1000, 1000), save_freq=1, save_dir='data/temp2')

    # Crop video
    # input_path = 'data/video2/006.avi'
    # input_paths = glob('data/video2/*.avi')
    # for input_path in tqdm(input_paths):
        # crop_video(input_path=input_path, y1=0, y2=896, x1=484, x2=1380, output_dims=(1000, 1000), save_dir='data/temp2')
    # crop_video(input_path='data/video2/066.avi', y1=1, y2=856, x1=11, x2=906, output_dims=(1000, 1000), save_dir='data/temp2')

    # Get XLSX data file using annotations and images
    # convert_json_to_xlsx(ann_dir='data/ann', img_dir='data/img', save_dir='data')

    # Convert callback images to video
    model_dirs = ['MobileNet_V2', 'MobileNet_V2_ft', 'ResNet_V2', 'ResNet_V2_ft', 'Inception_V3', 'Inception_V3_ft',
                  'Inception_ResNet_V2', 'Inception_ResNet_V2_ft', 'EfficientNet_B5']
    video_dir = 'video_training_all'            # video_training, video_predictions
    ids = ['003_007', '003_034', '004_002', '004_013', '005_003', '005_007', '005_022', '008_030', '008_045', '009_003',
           '009_023', '010_006', '010_031', '012_032', '012_063', '014_050', '014_110', '016_070', '016_180', '017_103']
    # run_names = model_dirs = ['MobileNet_V2']
    # video_dir = 'video_predictions'            # video_training, video_predictions
    # ids = ['003_007', '004_013', '009_003']
    for model_dir in model_dirs:
        for id in ids:
            convert_images_to_video(model_dir=model_dir, images_prefix=id, fps=8, add_note=True,
                                    save_dir=os.path.join(video_dir, model_dir))

    # Pull video to WANDB server
    # for run_name in run_names:
    #     pull_video_to_wandb(project='tavr', run_name=run_name, video_dir=video_dir, fps=8, prefix='VP')

