import pdb

import keras.backend as K
import numpy as np
from keras.layers import Input
from keras.models import Model

from spp.RoiPoolingConv import RoiPoolingConv

image_data_format = K.image_data_format()
assert image_data_format in {'channels_first', 'channels_last'}, 'image_data_format must be in {channels_last, channels_first}'
# assert image_data_format in {'tf', 'th'}, 'image_data_format must be in {tf, th}'

pooling_regions = 2
num_rois = 4
num_channels = 12

if image_data_format == 'channels_last':
    in_img = Input(shape=(None, None, num_channels))
elif image_data_format == 'channels_last':
    in_img = Input(shape=(num_channels, None, None))

in_roi = Input(shape=(num_rois, 4))

out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([in_img, in_roi])

model = Model([in_img, in_roi], out_roi_pool)
model.summary()

model.compile(loss='mse', optimizer='sgd')

for img_size in [32]:
    if image_data_format == 'channels_first':
        X_img = np.random.rand(1, num_channels, img_size, img_size)
        row_length = [float(X_img.shape[2]) / pooling_regions]
        col_length = [float(X_img.shape[3]) / pooling_regions]
    elif image_data_format == 'channels_last':
        X_img = np.random.rand(1, img_size, img_size, num_channels)
        row_length = [float(X_img.shape[1]) / pooling_regions]
        col_length = [float(X_img.shape[2]) / pooling_regions]

    X_roi = np.array([[0, 0, img_size / 2, img_size / 2],
                      [0, img_size / 2, img_size / 2, img_size / 2],
                      [img_size / 2, 0, img_size / 2, img_size / 2],
                      [img_size / 2, img_size / 2, img_size / 2, img_size / 2]])

    X_roi = np.reshape(X_roi, (1, num_rois, 4))

    Y = model.predict([X_img, X_roi])

    for roi in range(num_rois):

        if image_data_format == 'channels_first':
            X_curr = X_img[0, :, intint(X_roi[0, roi, 1]):intint(X_roi[0, roi, 1]) + intint(X_roi[0, roi, 3]),
                     intint(X_roi[0, roi, 0]):intint(X_roi[0, roi, 0]) + intint(X_roi[0, roi, 2])]
            row_length = float(X_curr.shape[1]) / pooling_regions
            col_length = float(X_curr.shape[2]) / pooling_regions
        elif image_data_format == 'channels_last':
            X_curr = X_img[0, int(X_roi[0, roi, 1]):int(X_roi[0, roi, 1]) + int(X_roi[0, roi, 3]),
                     int(X_roi[0, roi, 0]):int(X_roi[0, roi, 0]) + int(X_roi[0, roi, 2]), :]
            row_length = float(X_curr.shape[0]) / pooling_regions
            col_length = float(X_curr.shape[1]) / pooling_regions

        idx = 0

        for ix in range(pooling_regions):
            for jy in range(pooling_regions):
                for cn in range(num_channels):

                    x1 = int((ix * col_length))
                    x2 = int((ix * col_length + col_length))
                    y1 = int((jy * row_length))
                    y2 = int((jy * row_length + row_length))
                    dx = max(1, x2 - x1)
                    dy = max(1, y2 - y1)
                    x2 = x1 + dx
                    y2 = y1 + dy

                    if image_data_format == 'channels_first':
                        m_val = np.max(X_curr[cn, y1:y2, x1:x2])
                        if abs(m_val - Y[0, roi, cn, jy, ix]) > 0.01:
                            pdb.set_trace()
                        np.testing.assert_almost_equal(
                            m_val, Y[0, roi, cn, jy, ix], decimal=6)
                        idx += 1
                    elif image_data_format == 'channels_last':
                        m_val = np.max(X_curr[y1:y2, x1:x2, cn])
                        if abs(m_val - Y[0, roi, jy, ix, cn]) > 0.01:
                            pdb.set_trace()
                        np.testing.assert_almost_equal(
                            m_val, Y[0, roi, jy, ix, cn], decimal=6)
                        idx += 1

print('Passed roi pooling test')
