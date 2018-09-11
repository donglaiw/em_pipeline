import numpy as np
import h5py
from scipy.misc import imread,imresize

def dataset(dataset_dict, x0, x1, y0, y1, z0, z1, tile_sz, tile_ratio=1.0, tile_resize_mode='bilinear'):
    # no padding at the boundary
    result = np.zeros((z1-z0, y1-y0, x1-x0), np.uint8)
    c0 = x0 // tile_sz # floor
    c1 = (x1 + tile_sz-1) // tile_sz # ceil
    r0 = y0 // tile_sz
    r1 = (y1 + tile_sz-1) // tile_sz
    for z in range(z0, z1):
        pattern = dataset_dict["sections"][z]
        for row in range(r0, r1):
            for column in range(c0, c1):
                path = pattern.format(row=row+1, column=column+1)
                patch = imread(path, 0)
                if tile_ratio != 1.0: # float -> fraction
                    patch = imresize(patch, tile_ratio, tile_resize_mode)
                xp0 = column * tile_sz
                xp1 = (column+1) * tile_sz
                yp0 = row * tile_sz
                yp1 = (row + 1) * tile_sz
                if patch is not None:
                    x0a = max(x0, xp0)
                    x1a = min(x1, xp1)
                    y0a = max(y0, yp0)
                    y1a = min(y1, yp1)
                    result[z-z0, y0a-y0:y1a-y0, x0a-x0:x1a-x0] = patch[y0a-yp0:y1a-yp0, x0a-xp0:x1a-xp0]
    return result


def writeh5(filename, datasetname, dtarray):
    fid=h5py.File(filename,'w')
    ds = fid.create_dataset(datasetname, dtarray.shape, compression="gzip", dtype=dtarray.dtype)
    ds[:] = dtarray
    fid.close()


