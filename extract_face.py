#python extract_face.py --mafa_root "/Users/shirley/Desktop/NEU/5500/data/MAFA" --test_mat "/Users/shirley/Desktop/NEU/5500/data/MAFA/LabelTestAll.mat" --export_dir  "/Users/shirley/Desktop/NEU/5500/data/MAFA/test_align" --export_label_dir               "/Users/shirley/Desktop/NEU/5500/data/MAFA"

from argparse import ArgumentParser
from os import path
from pprint import pprint

import cv2
import hdf5storage
import numpy as np
from tqdm import tqdm
import pandas as pd

from pts_tools import expand_box, fit_box

argparser = ArgumentParser()
argparser.add_argument("--mafa_root", type=str, default=None,
                       help="MAFA dataset root folder.")
argparser.add_argument("--train_mat", type=str, default=None,
                       help="The mat file contains the train labels.")
argparser.add_argument("--test_mat", type=str, default=None,
                       help="The mat file contains the test labels.")
argparser.add_argument("--export_dir", default=None, type=str,
                       help="Where the extracted images should be saved.")
argparser.add_argument("--export_label_dir", default=None, type=str,
                       help="Where the extracted images should be saved.")


log_file = open('error_log.log', 'w')

def load_labels(label_file, is_train):
    out = hdf5storage.loadmat(label_file)
    samples = []
    if is_train:
        record = out['label_train'][0]
        for item in record:
            samples.append(
                {
                    'image_file': item[1][0],
                    'labels': [v for v in item[2].astype(int)]
                }
            )
    else:
        record = out['LabelTest'][0]
        for item in record:
            samples.append(
                {
                    'image_file': item[0][0],
                    'labels': [v for v in item[1].astype(int)]
                }
            )
    return samples


def parse_labels(raw_labels, is_train=True):
    """
    FOR TRAIN LABELS
        raw labels form: [x,y,w,h, x1,y1,x2,y2, x3,y3,w3,h3, occ_type, occ_degree,
        gender, race, orientation, x4,y4,w4,h4]
        (a) (x,y,w,h) is the bounding box of a face,
        (b) (x1,y1,x2,y2) is the position of two eyes.
        (c) (x3,y3,w3,h3) is the bounding box of the occluder. Note that (x3,y3)
            is related to the face bounding box position (x,y)
        (d) occ_type stands for the occluder type and has: 1 for simple, 2 for
            complex and 3 for human body.
        (e) occ_degree stands for the number of occluded face parts
        (f) gender and race stand for the gender and race of one face
        (g) orientation stands for the face orientation/pose, and has: 1-left,
            2-left frontal, 3-frontal, 4-right frontal, 5-right
        (h) (x4,y4,w4,h4) is the bounding box of the glasses and is set to
            (-1,-1,-1,-1) when no glasses. Note that (x4,y4) is related to the
            face bounding box position (x,y)

    FOR TEST LABELS
        The format is stored in a 18d array (x,y,w,h,face_type,x1,y1,w1,h1, occ_type,
        occ_degree, gender, race, orientation, x2,y2,w2,h2), where
        (a) (x,y,w,h) is the bounding box of a face,
        (b) face_type stands for the face type and has: 1 for masked face, 2 for
            unmasked face and 3 for invalid face.
        (c) (x1,y1,w1,h1) is the bounding box of the occluder. Note that (x1,y1)
            is related to the face bounding box position (x,y)
        (d) occ_type stands for the occluder type and has: 1 for simple, 2 for
            complex and 3 for human body.
        (e) occ_degree stands for the number of occluded face parts
        (f) gender and race stand for the gender and race of one face
        (g) orientation stands for the face orientation/pose, and has: 1-left,
            2-left frontal, 3-frontal, 4-right frontal, 5-right
        (h) (x2,y2,w2,h2) is the bounding box of the glasses and is set to
            (-1,-1,-1,-1) when no glasses.  Note that (x2,y2) is related to the
            face bounding box position (x,y)

    """
    labels = []
    if is_train:
        for raw_label in raw_labels:
            labels.append(
                {
                    'face': [raw_label[0], raw_label[1], raw_label[2], raw_label[3]],
                    'eyes': [raw_label[4], raw_label[5], raw_label[6], raw_label[7]],
                    'occlude': {
                        'location': [raw_label[8], raw_label[9], raw_label[10], raw_label[11]],
                        'type': raw_label[12],
                        'degree': raw_label[13]},
                    'gender': raw_label[14],
                    'race': raw_label[15],
                    'orientation': raw_label[16],
                    'glass': [raw_label[17], raw_label[18], raw_label[19], raw_label[20]]
                }
            )
    else:
        for raw_label in raw_labels:
            labels.append(
                {
                    'face': [raw_label[0], raw_label[1], raw_label[2], raw_label[3]],
                    'face_type': raw_label[4],
                    'occlude': {
                        'location': [raw_label[5], raw_label[6], raw_label[7], raw_label[8]],
                        'type': raw_label[9],
                        'degree': raw_label[10]},
                    'gender': raw_label[11],
                    'race': raw_label[12],
                    'orientation': raw_label[13],
                    'glass': [raw_label[14], raw_label[15], raw_label[16], raw_label[17]]
                }
            )

    return labels


def draw_face(image, labels, color=(0, 255, 0)):
    for label in labels:
        x, y, w, h = label['face']


def draw_mask(image, labels, color=(0, 0, 255)):
    for label in labels:
        x, y, w, h = label['face']
        _x, _y, _w, _h = label['occlude']['location']


def export_face(image, labels, export_file, label_file,occ_types=[1, 2, 3],min_prop = 0.05, export_size=160):
    """
    Export face areas in an image.
    Args:
        image: the image as a numpy array.
        labels: MAFA labels.
        export_file: the output file name. If more than one face exported, a subfix
            number will be appended to the file name.
        occ_types: a list of occlusion type which should be exported.
        min_size: the minimal size of faces should be exported.
        exprot_size: the output size of the square image.
    Returns:
        the exported image, or None.
    """
    # Crop the face
    idx_for_face = 0
    image_faces = []
    for label in labels:
        # Not all faces in label is occluded. Filter the image by occlusion,
        # size, etc.
        W, H, _ = image.shape
        x, y, w, h = label['face']
        if w/H < min_prop or h/H < min_prop:
            continue

        if label['occlude']['type'] not in occ_types:
            continue
        
        # Enlarge the face area and make it a square.
        box = expand_box([x, y, x+w, y+h], 1.3)
        box = fit_box(box, image, [(box[0], box[1]), (box[2], box[3])])
        if box is not None:
            image_face = image[box[1]:box[3], box[0]:box[2]]
        else:
            log_file.write(export_file + "\n")
            return None

        # Resize and save image.
        image_face = cv2.resize(image_face, (export_size, export_size))
        new_file = export_file.rstrip(
            '.jpg') + '-{}.jpg'.format(idx_for_face)
        
        # get the human body occlusion label
        if label['occlude']['type'] == 3:
            new_file_o = new_file.rsplit('/',1)[0] + '/human_body/' + new_file.rsplit('/',1)[1]
        else:
            new_file_o = new_file.rsplit('/',1)[0] + '/non_hb/' + new_file.rsplit('/',1)[1]
        cv2.imwrite(new_file_o, image_face)
        
        # get the occlusion degree label
        if label['occlude']['degree'] <= 2:
            new_file_d = (new_file.rsplit('/',1)[0] + '/not_correct/' + new_file.rsplit('/',1)[1])
        else:
            new_file_d = (new_file.rsplit('/',1)[0] + '/correct/' + new_file.rsplit('/',1)[1])
        cv2.imwrite(new_file_d, image_face)
        
        image_faces.append(image_face)
        
        # output the labels and filename
#         label_file.setdefault('glass', []).append(1 if label['occlude']['type'] == 3 else 0)
#         label_file.setdefault('occlude', []).append(0 if label['occlude']['degree'] == 3 else 1)
#         label_file.setdefault('file_g',[]).append(new_file_o)
#         label_file.setdefault('file_d',[]).append(new_file_d)
        
        idx_for_face += 1
    
    
    return image_faces


if __name__ == "__main__":
    # Get all args.
    args = argparser.parse_args()

    is_train = False if args.test_mat is not None else True



    img_dir = 'train images' if is_train else 'test images'
    mat = args.train_mat if is_train else args.test_mat

    # Load annotations from the mat file.
    samples = load_labels(mat, is_train=is_train)


    # loop through all the annotations and do processing. Here we are going to
    # extract all the occluded faces and save them in a new image file.
    label_file = {}
    for sample in tqdm(samples):
        img_file_name = sample['image_file']
        labels = parse_labels(sample['labels'], is_train=is_train)
        img_url = path.join(args.mafa_root, img_dir, img_file_name)
        image = cv2.imread(img_url)

        # Preview the data. Only draw annotation when not exporting.

        #if args.export_dir is None:
        draw_face(image, labels)
        draw_mask(image, labels)

        # Extract the face images.
        if args.export_dir is not None:
            face_image_file = path.join(args.export_dir, img_file_name)
            exported_face = export_face(image, labels, face_image_file, occ_types=[1, 2, 3], min_prop=0.05, export_size=160, label_file = label_file)
    # output labels 
    label_csv = pd.DataFrame.from_dict(label_file)
    if is_train:
        label_csv.to_csv(path.join(args.export_label_dir, 'train_labels.csv'), index = False)
        
    label_csv.to_csv(path.join(args.export_label_dir, 'test_labels.csv'), index = False)

log_file.close()

            # # Preview the exported face
            # if exported_faces is not None:
            #     for exported_face in export_faces:
            #         cv2.imshow('preview', exported_face)
            #         if cv2.waitKey() == 27:
            #             break
