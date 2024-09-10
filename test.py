import os, sys
import numpy as np

from em_util.io import *
from em_util.seg import *

opt = sys.argv[1]
fn_aff = '/data/adhinart/em100um/output/Zebrafinch_UNet_LR/test.zarr'
Dd = "/data/projects/weilab/dataset/zebrafinch/"
Do = '/data/projects/weilab/weidf/eng/db/zebrafinch/'
if opt[0] == '0':
    if opt == '0':
        # check the size of the zarr file
        seg = read_h5(f'{Do}/waterz/0_57_soma2d.h5', ['seg'])[::2,::4,::4]
        write_h5('db/0_57_soma2d_40-36-36.h5',seg)
    elif opt == '0.1': # create bv mask as the same size as cb_80nm
        seg = read_image_folder(f'{Dd}/mask_align_10nm_thres/%04d.png', range(0,5700,4), 'seg', [0.125,0.125])
        write_h5(f'{Dd}/mask_80nm.h5', seg)
    elif opt == '0.2': # find out problematic soma id
        mask = read_h5(f'{Dd}/mask_80nm.h5') 
        soma = read_h5(f'{Dd}/yl_cb_80nm.h5')
        soma = soma[:mask.shape[0], :mask.shape[1], :mask.shape[2]]
        ui, uc = np.unique(soma[soma>0], return_counts=True)
        ui2, uc2 = np.unique(soma * (mask==0), return_counts=True)
        uc_m = get_query_count(ui2, uc2, ui)
        bid = uc_m/uc < 0.9
        # 3,  84, 144, 146, 159, 192, 216, 217, 226, 231, 233, 237, 278,279, 403, 419, 428, 475, 485, 522, 523, 539, 551, 631, 658, 712,713, 719, 784, 785, 787, 788, 797, 798, 805, 806, 808, 809, 814,818, 819, 821
        print(ui[bid], uc[bid], uc_m[bid])
        import pdb; pdb.set_trace()
        for i in ui[bid]:
            # which chunk
            print(i, compute_bbox(soma==i))
        import pdb; pdb.set_trace()
                    
    
elif opt[0] == '1':
    from em_util.ng import *
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
        # visualize bv mask
        zz = 1000
        mask = read_image(f'{Dd}/mask_align_10nm_thres/{zz:04d}.png')
        res = np.array([9,9,20])
        with viewer.txn() as s:
            s.layers['image'] = neuroglancer.ImageLayer(source=D0)
            s.layers.append(name='mask',layer=ng_layer(mask[None], res, oo=[0,0,zz]))
    elif opt == '1.1': 
        #ls;python -i test.py 1.1
        # visualize seg chunk result        
        # zz=0;mask = read_h5(f'{Do}/waterz/0_57_soma2d.h5', ['seg']);res = np.array([9,9,20])
        zz=0;mask = read_h5('db/0_57_soma2d_40-36-36.h5'); res = np.array([36,36,40])
        
        # somabfs
        #rl = read_h5('db/0_57_somaBFS_mapping.h5','mapping')
        rl = read_h5(f'{Do}/waterz/0_57_soma2d_iou-0.8.h5')
        
        
        mask[mask<20000000] = rl[mask[mask<20000000]]
        numz = seg3d_to_zcount(mask)
        # print(np.unique(mask[mask>20000000]))
        #zz=0; mask = read_h5('db/0_57_soma2d_40-36-36.h5')
        #res = np.array([36,36,40])
        #soma = read_h5('/data/projects/weilab/dataset/zebrafinch/yl_cb_80nm.h5'); res_soma = np.array([72,72,80])
        
        
        with viewer.txn() as s:
            s.layers['image'] = neuroglancer.ImageLayer(source=D0)
            s.layers.append(name='mask',layer=ng_layer(mask, res, oo=[0,0,zz]))
            #s.layers.append(name='mask',layer=ng_layer(soma, res_soma))
        
        """
        aff = read_h5('test_aff.h5')[None]
        with viewer.txn() as s:            
            s.layers.append(name='mask',layer=ng_layer(aff, res, oo=[0,0,8], tt='image'))
        """
    elif opt == '1.2': 
        res = [9,9,20]
        sz = 1024;oset = [4820, 4939,0]
        soma = read_h5(f'{Do}//test_agglo/soma2d_{sz}_iou-0.8.h5')
        ffn = read_h5(f'{Do}/test_agglo/ffn_{sz}.h5')
                
        with viewer.txn() as s:
            s.layers['image'] = neuroglancer.ImageLayer(source=D0)
            s.layers.append(name='ffn',layer=ng_layer(ffn, res, oo=oset))
            s.layers.append(name='soma',layer=ng_layer(soma, res, oo=oset))
    print(viewer)

elif opt[0] == '2': # create benchmark datasets
    Do = f'{Do}/test_agglo/'
    center = [5456, 5332]
    szs = [1024, 2048, 4096]
    if opt == '2': # soma_2d
        #seg = read_h5(f'{Do}/../waterz/0_57_soma2d.h5',['seg']);sn='soma2d'
        import cc3d
        import fastremap
        seg = read_h5(f'{Do}/ffn.h5');sn='ffn'

        for sz in szs:
            out = seg[:, center[0]-sz//2:center[0]+sz//2, \
                         center[1]-sz//2:center[1]+sz//2]
            write_h5(f'{Do}/{sn}_{sz}.h5', fastremap.refit(cc3d.connected_components(out)))
    elif opt == '2.1': # iou
        for sz in szs:
            seg = read_h5(f'{Do}/soma2d_{sz}.h5')
            get_seg = lambda x : seg[x]
            iou = segs_to_iou(get_seg, range(seg.shape[0]))
            write_h5(f'{Do}/soma2d_{sz}_iou.h5', iou)
    elif opt == '2.2': # ffn
        out = np.zeros([100,10913,10664], np.uint64)
        tsz = 2048
        for r in range(6):
            for c in range(6):
                out[:,r*tsz:(r+1)*tsz,c*tsz:(c+1)*tsz] = read_h5(f'{Dd}/ffn/0000/%d_%d.h5'%(r,c))[:100]
        write_h5(f'{Do}/ffn.h5', out)
    elif opt[:3] == '2.3': # 1024
        sz = 1024
        if opt == '2.3':
            #fn = f'{Do}/ffn_{sz}.h5'
            fn = f'{Do}/soma2d_{sz}_iou-0.8.h5'
            mask = read_h5(fn)
            count = seg3d_to_zcount(mask)
            print((count[1]==100).sum(), ((count[1]>50)*(count[1]<100)).sum())
        elif opt == '2.31':
            from waterz.region_graph import merge_id
            iou_thres = 0.8
            mask = read_h5(f'{Do}/soma2d_{sz}.h5')
            iou = read_h5(f'{Do}/soma2d_{sz}_iou.h5')
            iou = np.vstack(iou)
            soma_id0 = 2e7
            # ids that are not related to soma
            iou = iou[(iou[:,:2] > soma_id0).max(axis=1) == 0]
            score = iou[:, 4].astype(float) / (iou[:, 2] + iou[:, 3] - iou[:, 4])            
            gid = score >= iou_thres
            relabel = merge_id(iou[gid, 0].astype(np.uint32), iou[gid, 1].astype(np.uint32))
            mask[mask<len(relabel)] = relabel[mask[mask<len(relabel)]]
            write_h5(f'{Do}/soma2d_{sz}_iou-{iou_thres}.h5', mask)