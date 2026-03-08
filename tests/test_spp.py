import keras.backend as K
import numpy as np
from keras.models import Sequential

from spp.SpatialPyramidPooling import SpatialPyramidPooling

dim_ordering = K.image_data_format()
assert dim_ordering in {'channels_first', 'channels_last'}, 'image_data_format must be in {channels_last, channels_first}'

pooling_regions = [1,2,4]

num_channels = 12
batch_size = 16

if dim_ordering == 'channels_first':
    input_shape = (num_channels, None, None)
elif dim_ordering == 'channels_last':
    input_shape = (None, None, num_channels)

model = Sequential()
model.add(SpatialPyramidPooling(pooling_regions, input_shape=input_shape))
model.summary()

model.compile(loss='mse', optimizer='sgd')

for img_size in [8,16]:

    if dim_ordering == 'channels_first':
        X = np.random.rand(batch_size, num_channels, img_size, img_size*2)
        row_lengchannels_last = [float(X.shape[2]) / i for i in pooling_regions]
        col_lengchannels_last = [float(X.shape[3]) / i for i in pooling_regions]
    elif dim_ordering == 'channels_last':
        X = np.random.rand(batch_size, img_size, img_size*2, num_channels)
        row_lengchannels_last = [float(X.shape[1]) / i for i in pooling_regions]
        col_lengchannels_last = [float(X.shape[2]) / i for i in pooling_regions]

    Y = model.predict(X)

    for batch_num in range(batch_size):
        idx = 0
        for pool_num, num_pool_regions in enumerate(pooling_regions):
            for jy in range(num_pool_regions):
                for ix in range(num_pool_regions):
                    for cn in range(num_channels):
                        x1 = int(round(ix * col_lengchannels_last[pool_num]))
                        x2 = int(round(ix * col_lengchannels_last[pool_num] + col_lengchannels_last[pool_num]))
                        y1 = int(round(jy * row_lengchannels_last[pool_num]))
                        y2 = int(round(jy * row_lengchannels_last[pool_num] + row_lengchannels_last[pool_num]))

                        if dim_ordering == 'channels_first':
                            m_val = np.max(X[batch_num, cn, y1:y2, x1:x2])
                        elif dim_ordering == 'channels_last':
                            m_val = np.max(X[batch_num, y1:y2, x1:x2, cn])

                        np.testing.assert_almost_equal(
                           m_val, Y[batch_num, idx], decimal=6)
                        idx += 1

print('Spatial pyramid pooling test passed')
