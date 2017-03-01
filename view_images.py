import dicom
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

def get_slice_location(dcm):
    return float(dcm[0x0020, 0x1041].value)

url = os.getcwd()

for folder in os.listdir(os.path.join(url, 'sample_images')):
    df = pd.DataFrame(columns=['slice_location'],)
    try:
        os.makedirs(os.path.join(url, 'data', folder))
    except FileExistsError:
        pass

    for sample in os.listdir(os.path.join(url, 'sample_images', folder)):
        if not '.png' in sample:
            ds = dicom.read_file(os.path.join(url, 'sample_images', folder, sample))
            df.loc[sample[:-4]] = get_slice_location(ds)
            pix = ds.pixel_array
            cv2.imwrite(os.path.join(url, 'data', folder, sample)[:-4] + '.png', pix / 16)
    df.to_csv(os.path.join(url, 'data', folder, 'slice_info.csv'))
    plus = len(os.listdir(os.path.join(url, 'sample_images', folder)))
    print('Patient {}: {} scans'.format(folder, plus))


