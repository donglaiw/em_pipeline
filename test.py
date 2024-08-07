import os, sys
import numpy as np

from em_util.io import *
from em_util.ng import *

opt = sys.argv[1]
fn_aff = '/data/adhinart/em100um/output/Zebrafinch_UNet_LR/test.zarr'
Dd = "/data/projects/weilab/dataset/zebrafinch/"
if opt == '0':
    # check the size of the zarr file
    import zarr
    vol = zarr.open(fn_aff, mode='r')
    print(vol.shape)
elif opt[0] == '1':
    # ng visualization
    import neuroglancer
    ip='localhost' # or public IP of the machine for sharable display
    port=9092 # change to an unused port number
    neuroglancer.set_server_bind_address(bind_address=ip,bind_port=port)
    viewer=neuroglancer.Viewer()
    D0='precomputed://gs://j0126-nature-methods-data/GgwKmcKgrcoNxJccKuGIzRnQqfit9hnfK1ctZzNbnuU/rawdata_realigned'
    with viewer.txn() as s:
        s.layers['image'] = neuroglancer.ImageLayer(source=D0)
    
    if opt == '1': 
        # visualize mask
    zz = 1000
    mask = read_image(f'{Dd}/mask_align_10nm_thres/{zz:04d}.png')
    res = np.array([9,9,20])
    with viewer.txn() as s:
        s.layers['image'] = neuroglancer.ImageLayer(source=D0)
        s.layers.append(name='mask',layer=ng_layer(mask[None], res, oo=[0,0,zz]))
    
    print(viewer)
