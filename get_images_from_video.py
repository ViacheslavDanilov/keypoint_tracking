from tqdm import tqdm
import cv2
import os

DATA_DIR = 'src_data/video'       # source folder for video
SAVE_DIR = 'src_data/imgs'        # save folder for output images
EXCEPT_CASE = ['006.avi', '046.avi', '048.avi', '050.avi', '051.avi', '052.avi', '053.avi', '054.avi', '059.avi', '062.avi']
OUTPUT_DIMS = (1000, 1000)        # output image size
SAVE_FREQ = 1                     # process and save each SAVE_FREQ image from a video

file_names = os.listdir(DATA_DIR)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

pbar = tqdm(file_names)
for file_name in pbar:
    i = 0
    times = 0
    cap = cv2.VideoCapture(os.path.join(DATA_DIR, file_name))
    while True:
        ret, frame = cap.read()
        if ret == True:
            times += 1
            if times % SAVE_FREQ == 0:
                # if file_name == EXCEPT_CASE:
                if file_name in EXCEPT_CASE:
                    frame = frame[0:896, 485:1380]
                frame = cv2.resize(frame, OUTPUT_DIMS, interpolation=cv2.INTER_AREA)
                video_name = os.path.splitext(file_name)[0]
                image_path = os.path.join(SAVE_DIR, video_name + '_' + str(i+1).zfill(3) + '.png')
                cv2.imwrite(image_path, frame)
                i += 1
        else:
            break
    pbar.set_description("Processing %s" % file_name)
    print('{0:d} images_1 saved for {1:s}'.format(i, file_name))
