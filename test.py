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
        for i in ui[bid]:
            # which chunk
            print(i, compute_bbox(soma==i))
                    
    
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
        #soma = read_h5(f'{Do}//test_agglo/soma2d_{sz}_iou-0.8.h5')
        soma = read_h5(f'{Do}//test_agglo/soma2d_{sz}_iou0.8_sz0.6.h5')
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
    elif opt == '2.21': # aff
        import yaml, pickle
        with open('conf/j0126.yml', 'r') as file:
            conf = yaml.safe_load(file)             
        dask_array = pickle.load(open(conf['aff']['path'], "rb"))        
        aff = dask_array[:3, :100].compute()
        aff[aff < conf['waterz']['low']] = 0    
        print('apply mask')
        border_width = conf['mask']['border_width'] 
        z0 = 0
        for z in range(aff.shape[1]):
            # blood vessel
            mask = read_image(conf['mask']['blood_vessel'] % (z0+z)) == 0                
            # border
            bd = np.loadtxt(conf['mask']['border'] % (z0+z)).astype(int)
            mask[:bd[0]+border_width] = 0
            mask[bd[1]-border_width:] = 0
            mask[:, :bd[2]+border_width] = 0
            mask[:, bd[3]-border_width:] = 0
            aff[:,z] = aff[:,z] * mask
        write_h5(f'{Do}/aff.h5', aff)
    elif opt == '2.22': # aff
        aff = read_h5(f'{Do}/aff.h5');sn='aff'
        for sz in szs:
            out = aff[:, :, center[0]-sz//2:center[0]+sz//2, \
                         center[1]-sz//2:center[1]+sz//2]
            write_h5(f'{Do}/{sn}_{sz}.h5', out)
    elif opt[:3] == '2.3': # 1024
        from waterz.region_graph import merge_id
        sz = 1024
        zz = 100
        # 1-sided iou is enough
        # high thres -> has to be best buddy
        iou_thres = 0.8 
        sz_thres = 0.6
        #sz_thres = 0.8
        z_thres = 5
        z_thres = 1
        os_thres = 0.6
        soma_id0 = int(2e7)
        if opt == '2.3':
            from em_util.eval import *
            fn = f'{Do}/ffn_{sz}.h5'
            #fn = f'{Do}/soma2d_{sz}_iou0.8.h5'
            #fn = f'{Do}/soma2d_{sz}_iou0.8_sz0.8.h5'
            #fn = f'{Do}/soma2d_{sz}_iou0.8_sz0.6.h5'
            fn = f'{Do}/soma2d_{sz}_iou0.8_sz0.6_z5.h5'
            fn = f'{Do}/soma2d_{sz}_iou0.8_sz0.6_z1.h5'
            fn = f'{Do}/soma2d_{sz}_iou0.8_sz0.6_z1_os0.6.h5'
            #fn = f'{Do}/soma2d_{sz}_iou0.8_sz0.6_z5_aff1.h5'
            mask = read_h5(fn)
            do_eval = True
            #do_eval = False
            if do_eval:
                gt = read_h5(f'{Do}/ffn_{sz}_r0.5.h5')
                stats = adapted_rand(mask, gt, True)
                print('%.2f, %.2f, %.2f' % (stats[0], stats[1], stats[2]))
                iou = seg_to_iou(gt, mask)
                score = iou[:, 4].astype(float) / (iou[:, 2] + iou[:, 3] - iou[:, 4])
                # false split
                # print(iou[score<0.5,1])
                # false merge
                print(iou[iou[:,3]-iou[:,4] > 0.5 * iou[:,4], 1])
                # bin iou score
                print(np.histogram(score,[0,0.5,0.8,1]))
                # print(iou[iou[:,1]==0])
            else:
                count = seg3d_to_zcount(mask)
                print((count[1]==100).sum(),((count[1]>=50)*(count[1]<100)).sum(), ((count[1]>=30)*(count[1]<50)).sum())            
            # print_arr(count[0][count[1]==1])
        elif opt == '2.30':
            ratio, numv = 0.5, 10000
            sn = f'{Do}/ffn_{sz}'
            mask = read_h5(f'{sn}.h5')
            if not os.path.exists(f'{sn}-bb.h5'):
                bb = compute_bbox_all(mask, True)
                write_h5(f'{sn}-bb.h5', bb)
            else:
                bb = read_h5(f'{sn}-bb.h5')            
            # remove small seg
            bb_ratio = (bb[:,2::2] - bb[:,1:-1:2] + 1)/[zz,sz,sz]
            gid = bb[(bb_ratio.max(axis=1) >= ratio) * (bb[:,-1]>=numv), 0]
            mask = seg_remove_id(mask, gid, True)
            write_h5(f'{Do}/ffn_{sz}_r{ratio}.h5', mask)
        elif opt == '2.31': # vertical branches
            mask = read_h5(f'{Do}/soma2d_{sz}.h5')
            iou = read_h5(f'{Do}/soma2d_{sz}_iou.h5')
            iou = np.vstack(iou)            
            # ids that are not related to soma
            iou = iou[(iou[:,:2] > soma_id0).max(axis=1) == 0]
            score = iou[:, 4].astype(float) / (iou[:, 2] + iou[:, 3] - iou[:, 4])            
            gid = score >= iou_thres
            relabel = merge_id(iou[gid, 0].astype(np.uint32), iou[gid, 1].astype(np.uint32))
            mask[mask<len(relabel)] = relabel[mask[mask<len(relabel)]]
            write_h5(f'{Do}/soma2d_{sz}_iou{iou_thres}.h5', mask)
            
            bbox = compute_bbox_all(mask, True)
            write_h5(f'{Do}/soma2d_{sz}_iou{iou_thres}-bb.h5', bbox)

            # need to update iou
            iou_id = iou[:, :2]            
            iou_id[iou_id<len(relabel)] = relabel[iou_id[iou_id<len(relabel)]]
            iou[:, :2] = iou_id
            iou = iou[iou[:,0] != iou[:,1]]            
            write_h5(f'{Do}/soma2d_{sz}_iou{iou_thres}-iou.h5', iou)

        elif opt == '2.32': # sloped branches
            # not updated iou                        
            mask = read_h5(f'{Do}/soma2d_{sz}_iou{iou_thres}.h5')
            bbox = read_h5(f'{Do}/soma2d_{sz}_iou{iou_thres}-bb.h5')
            iou = read_h5(f'{Do}/soma2d_{sz}_iou{iou_thres}-iou.h5')            
                
            sid = bbox[bbox[:,1]==bbox[:,2], 0]
            #sid_g = np.in1d(iou[:,0], sid) * np.in1d(iou[:,1], sid) 
            # need to add constraint that the non-singleton seg must stop at that slice
            sid_g = np.in1d(iou[:,0], sid) + np.in1d(iou[:,1], sid)            
            sz_ratio = iou[sid_g, 2] /iou[sid_g, 3]            
            mid = iou[sid_g][(sz_ratio >= sz_thres) * (sz_ratio <= 1/sz_thres),:2]
            mid = mid.astype(np.uint32)
            print(mid.shape)
            # best buddy: need backward iou
            # mid2 = np.stack([mid.min(axis=1), mid.max(axis=1)], axis=1)             
            #ui, uc = np.unique(mid2, return_counts=True, axis=0)
            relabel = merge_id(mid[:, 0], mid[:, 1])
            mask[mask<len(relabel)] = relabel[mask[mask<len(relabel)]]
            write_h5(f'{Do}/soma2d_{sz}_iou{iou_thres}_sz{sz_thres}.h5', mask)            
            
            bbox = compute_bbox_all(mask, True)
            write_h5(f'{Do}/soma2d_{sz}_iou{iou_thres}_sz{sz_thres}-bb.h5', bbox)
            
            count = seg3d_to_zcount(mask)
            print((count[1]==100).sum(),((count[1]>=50)*(count[1]<100)).sum(), ((count[1]>=10)*(count[1]<50)).sum())

        elif opt == '2.33': # iou_f, iou_b
            sn = f'{Do}/soma2d_{sz}_iou{iou_thres}_sz{sz_thres}'
            sn = f'{Do}/soma2d_{sz}_iou{iou_thres}_sz{sz_thres}_z{z_thres}'
            
            mask = read_h5(f'{sn}.h5')
            get_seg = lambda x: mask[x]
            iou_f = segs_to_iou(get_seg, range(mask.shape[0]))
            iou_b = segs_to_iou(get_seg, range(mask.shape[0])[::-1])
            write_h5(f'{sn}-iouF.h5', iou_f)
            write_h5(f'{sn}-iouB.h5', iou_b)
            
        elif opt in ['2.34', '2.341']: # bbox-connect
            sn = f'{Do}/soma2d_{sz}_iou{iou_thres}_sz{sz_thres}'
            bbox = read_h5(f'{sn}-bb.h5')
            num_b = (bbox[:,1:-1:2] == 0).sum(axis=1) + (bbox[:,2] == zz-1) + (bbox[:,4::2] == sz-1).sum(axis=1)
            
            # big enough but not enough bd
            # only z
            num_z = bbox[:,2] - bbox[:,1] + 1
            todo = (num_z >= z_thres) * (num_b<2)            
            todo_id = bbox[todo, 0]
            # xyz
            # ratio = (bbox[:,2::2] - bbox[:,1:-1:2] + 1) / [100, sz, sz]            
            # todo = (ratio.min(axis=1) >= 0.1) * (num_b<2)
            iouF = read_h5(f'{sn}-iouF.h5')
            iouB = read_h5(f'{sn}-iouB.h5')
            # import pdb;pdb.set_trace()
            if opt == '2.34': # object-centric
                done = np.zeros([2, bbox[:,0].max()+1], np.uint8)                
                done[:, soma_id0:] = 1 # no merge soma
                # done = (num_z >= z_thres) * (num_b>=2)
                todo_order = np.argsort(-num_z[todo])
                mid = []
                for bb in bbox[todo][todo_order]:
                    if done[0, bb[0]] == 0:
                        # complete top
                        s0 = bb[0]
                        z0 = bb[1]                    
                        while z0 != 0:
                            iou = iouB[zz-1-z0]                                
                            cid = iou[iou[:,0]==s0, 1][0]                            
                            if done[1, cid] == 0 and bbox[bbox[:,0]==cid, 2] == bb[1] - 1:
                                # best buddy
                                iou = iouF[z0-1]
                                if s0 == iou[iou[:,0]==cid, 1][0]:                                
                                    mid.append([s0, cid])
                                    z0 = bbox[bbox[:,0]==cid, 1][0]
                                    s0 = cid
                                    done[1, cid] = 1
                                else:
                                    break
                            else:
                                break
                    if done[1, bb[0]] == 0:
                        z0 = bb[2]
                        s0 = bb[0]
                        while z0 != zz-1:                             
                            iou = iouF[z0]
                            cid = iou[iou[:,0]==s0, 1][0]
                            # cid in todo_id ?
                            if done[0, cid] == 0 and bbox[bbox[:,0]==cid, 1] == bb[2] + 1:
                                iou = iouB[zz-1-z0-1]
                                if s0 == iou[iou[:,0]==cid, 1][0]:
                                    mid.append([s0, cid])
                                    z0 = bbox[bbox[:,0]==cid, 2][0]
                                    s0 = cid
                                    done[0, cid] = 1
                                else:
                                    break                                            
                            else:
                                break                           
            elif opt == '2.341': # only pairwise
                mid = []
                for bb in bbox[todo]:                
                    sid = bb[0]
                    # z-face candidate
                    if bb[1] != 0: 
                        iou = iouB[zz-1-bb[1]]
                        cid = iou[iou[:,0]==sid, 1][0]
                        # cid in todo_id ?
                        if bbox[bbox[:,0]==cid, 2] == bb[1] - 1:
                            mid.append([sid, cid])
                    if bb[2] != 99: 
                        iou = iouF[bb[2]]
                        cid = iou[iou[:,0]==sid, 1][0]
                        # cid in todo_id ?
                        if bbox[bbox[:,0]==cid, 1] == bb[2] + 1:
                            mid.append([sid, cid])                            
            mid = np.vstack(mid).astype(np.uint32)
            print(mid.shape)
            import pdb;pdb.set_trace()
            relabel = merge_id(mid[:, 0], mid[:, 1])
            mask = read_h5(f'{sn}.h5')
            mask[mask<len(relabel)] = relabel[mask[mask<len(relabel)]]
            write_h5(f'{sn}_z{z_thres}.h5', mask)            
            
            bbox = compute_bbox_all(mask, True)
            write_h5(f'{sn}_z{z_thres}-bb.h5', bbox)
        
        elif opt in ['2.36']: # 1-sided iou
            sn = f'{Do}/soma2d_{sz}_iou{iou_thres}_sz{sz_thres}_z{z_thres}'
            bbox = read_h5(f'{sn}-bb.h5')
            num_b = (bbox[:,1:-1:2] == 0).sum(axis=1) + (bbox[:,2] == zz-1) + (bbox[:,4::2] == sz-1).sum(axis=1)
            
            # big enough but not enough bd
            # only z
            num_z = bbox[:,2] - bbox[:,1] + 1
            todo = (num_z >= z_thres) * (num_b<2)            
            todo_id = bbox[todo, 0]
            # xyz
            # ratio = (bbox[:,2::2] - bbox[:,1:-1:2] + 1) / [100, sz, sz]            
            # todo = (ratio.min(axis=1) >= 0.1) * (num_b<2)
            iouF = read_h5(f'{sn}-iouF.h5')
            iouB = read_h5(f'{sn}-iouB.h5')
            # import pdb;pdb.set_trace()
            if opt == '2.36': # object-centric
                done = np.zeros([2, bbox[:,0].max()+1], np.uint8)                
                done[:, soma_id0:] = 1 # no merge soma
                # done = (num_z >= z_thres) * (num_b>=2)
                todo_order = np.argsort(-num_z[todo])
                mid = []
                for bb in bbox[todo][todo_order]:
                    if done[0, bb[0]] == 0:
                        # complete top
                        s0 = bb[0]
                        z0 = bb[1]                    
                        while z0 != 0:
                            iou = iouB[zz-1-z0]       
                            iou_l = iou[iou[:,0]==s0][0]
                            # small to big
                            cid, score = iou_l[1], iou_l[4]/iou_l[2] 
                            if done[1, cid] == 0 and score >= os_thres:
                                mid.append([s0, cid])
                                # import pdb;pdb.set_trace() 
                                z0 = bbox[bbox[:,0]==cid, 1][0]
                                s0 = cid
                                done[1, cid] = 1                                
                            else:
                                break
                    if done[1, bb[0]] == 0:
                        z0 = bb[2]
                        s0 = bb[0]
                        while z0 != zz-1:                             
                            iou = iouF[z0]
                            iou_l = iou[iou[:,0]==s0][0]
                            cid, score = iou_l[1], iou_l[4]/iou_l[2]                            
                            # cid in todo_id ?
                            if done[0, cid] == 0 and score >= os_thres:                                
                                mid.append([s0, cid])
                                z0 = bbox[bbox[:,0]==cid, 2][0]
                                s0 = cid
                                done[0, cid] = 1                                
                            else:
                                break                                                                 
            mid = np.vstack(mid).astype(np.uint32)
            print(mid.shape)
            import pdb;pdb.set_trace()
            relabel = merge_id(mid[:, 0], mid[:, 1])
            mask = read_h5(f'{sn}.h5')
            mask[mask<len(relabel)] = relabel[mask[mask<len(relabel)]]
            write_h5(f'{sn}_os{os_thres}.h5', mask)            
            
            bbox = compute_bbox_all(mask, True)
            write_h5(f'{sn}_os{os_thres}-bb.h5', bbox)

        elif opt in ['2.35', '2.351']: # zaff
            import waterz
            aff_thres = 1
            sn = f'{Do}/soma2d_{sz}_iou{iou_thres}_sz{sz_thres}_z{z_thres}'
            bbox = read_h5(f'{sn}-bb.h5')
            num_b = (bbox[:,1:-1:2] == 0).sum(axis=1) + (bbox[:,2] == zz-1) + (bbox[:,4::2] == sz-1).sum(axis=1)
            bid = bbox[num_b<2, 0]
            bid = bid[bid < soma_id0]
            mask = read_h5(f'{sn}.h5')
            
            if not os.path.exists(f'{sn}-rg.h5'): 
                aff = read_h5(f'{Do}/aff_{sz}.h5') 
                rg = waterz.getRegionGraph(aff, mask, 2, "aff75_his256_ran255", rebuild=False)
                write_h5(f'{sn}-rg.h5', list(rg))
            else:
                rg = read_h5(f'{sn}-rg.h5') 
            
            if opt == '2.35': 
                pass            
            elif opt == '2.351':
                gid = np.in1d(rg[0][:,0], bid) * np.in1d(rg[0][:,1], bid) * (rg[1]<=aff_thres)            
                mid = rg[0][gid]
            import pdb;pdb.set_trace()
            relabel = merge_id(mid[:, 0], mid[:, 1])
            mask[mask<len(relabel)] = relabel[mask[mask<len(relabel)]]
            write_h5(f'{sn}_aff{aff_thres}.h5', mask)
            
            
            