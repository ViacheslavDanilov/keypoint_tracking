import os
import json
import glob
import pandas
from tqdm import tqdm

# --------------------------------------------------- INPUT SETTINGS ---------------------------------------------------
ANN_DIR = 'data/ann'
IMG_DIR = 'data/img'
SAVE_DIR = 'data'
# ----------------------------------------------------------------------------------------------------------------------

ann_paths = glob.glob(ANN_DIR + "/*.json")
ann_paths.sort()
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

column_order = ['Name', 'Path', 'Height', 'Width', 'Points',
                'AA1', 'AA2', 'STJ1', 'STJ2', 'CD', 'CM', 'CP', 'CT', 'PT', 'FE1', 'FE2',
                'AA1_x', 'AA1_y', 'AA2_x', 'AA2_y', 'STJ1_x', 'STJ1_y', 'STJ2_x', 'STJ2_y',
                'CD_x', 'CD_y', 'CM_x', 'CM_y', 'CP_x', 'CP_y', 'CT_x', 'CT_y',
                'PT_x', 'PT_y', 'FE1_x', 'FE1_y', 'FE2_x', 'FE2_y']

ann_df = pandas.DataFrame(columns=column_order)

for ann_idx, ann_path in tqdm(enumerate(ann_paths), unit=' json'):
    ann_df.loc[ann_idx, 0:len(column_order)] = 0
    with open(ann_path) as f:
        json_data = json.load(f)
    json_name = os.path.basename(ann_path)
    img_name = os.path.splitext(json_name)[0]
    img_path = os.path.join(IMG_DIR, img_name)
    img_path = os.path.normpath(img_path)
    height = json_data['size']['height']
    width = json_data['size']['width']
    num_points = len(json_data['objects'])
    ann_df.loc[ann_idx, 'Name'] = img_name
    ann_df.loc[ann_idx, 'Path'] = img_path
    ann_df.loc[ann_idx, 'Height'] = height
    ann_df.loc[ann_idx, 'Width'] = width
    ann_df.loc[ann_idx, 'Points'] = num_points

    for point_idx in range(num_points):
        point_class = json_data['objects'][point_idx]['classTitle']
        point_x = json_data['objects'][point_idx]['points']['exterior'][0][0]
        point_y = json_data['objects'][point_idx]['points']['exterior'][0][1]
        ann_df.loc[ann_idx, point_class] = 1
        ann_df.loc[ann_idx, point_class + '_x'] = point_x
        ann_df.loc[ann_idx, point_class + '_y'] = point_y

xlsx_name = os.path.join(SAVE_DIR, 'data.xlsx')
ann_df.to_excel(xlsx_name, sheet_name='Data', index=True, startrow=0, startcol=0)
print('Сonversion completed!')