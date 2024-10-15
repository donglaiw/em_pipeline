import os, sys
import numpy as np

from em_util.io import *
from em_util.seg import *
from em_pipeline.lib import R_pca

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
        # soma = read_h5(f'{Do}//test_agglo/soma2d_{sz}_iou-0.8.h5')
        # soma = read_h5(f'{Do}//test_agglo/soma2d_{sz}.h5')
        # soma = read_h5(f'{Do}//test_agglo/soma2d_{sz}_iou0.8_sz0.6_v2.h5', ['mask'])
        # soma = read_h5(f'{Do}//test_agglo/soma2d_{sz}_iou0.8_sz0.6_bb128_v2.h5', ['mask'])
        # soma = read_h5(f'{Do}//test_agglo/soma2d_{sz}_iou0.8_sz0.6_z1.h5')
        soma = read_h5(f'{Do}//test_agglo/soma2d_{sz}_iou0.8_sz0.6_rg32_bb_v2.h5', ['mask'])
        # soma2 = read_h5(f'{Do}//test_agglo/soma2d_{sz}_iou0.8_sz0.6_z1_os0.6.h5')
        #ffn = read_h5(f'{Do}/test_agglo/ffn_{sz}.h5')
        ffn = read_h5(f'{Do}/test_agglo/ffn_{sz}_r0.5.h5')
        aff = read_h5(f'{Do}/test_agglo/aff_{sz}.h5')
                
        with viewer.txn() as s:
            s.layers['image'] = neuroglancer.ImageLayer(source=D0)
            s.layers.append(name='ffn',layer=ng_layer(ffn, res, oo=oset))
            s.layers.append(name='soma',layer=ng_layer(soma, res, oo=oset))
            s.layers.append(name='aff',layer=ng_layer(aff[0], res, oo=oset, tt='image'))
            #s.layers.append(name='soma2',layer=ng_layer(soma2, res, oo=oset))
    print(viewer)

elif opt[0] == '2': # create benchmark datasets
    Do = f'{Do}/test_agglo/'
    center = [5456, 5332]
    szs = [1024, 2048, 4096]
    if opt == '2': # soma_2d
        import cc3d
        import fastremap
        #seg = read_h5(f'{Do}/ffn.h5');sn='ffn'
        seg = read_h5(f'{Do}/../waterz/0_57_soma2d.h5',['seg']);sn='soma2d'
        for sz in szs:
            out = seg[:, center[0]-sz//2:center[0]+sz//2, \
                         center[1]-sz//2:center[1]+sz//2]            
            # out[0][out[0]<2e7].max(), out[1][out[1]>0].min()
            write_h5(f'{Do}/{sn}_{sz}.h5', fastremap.refit(cc3d.connected_components(out, connectivity=6)))
    elif opt == '2.00': # compare 2d cc3d
        seg0 = read_h5(f'{Do}/soma2d_1024.h5')
        seg1 = read_h5(f'{Do}/v0/soma2d_1024.h5')
        #print(seg0.max(), seg1.max())
        print(len(np.unique(seg0)), len(np.unique(seg1)))
        
    elif opt == '2.0': # ffn
        out = np.zeros([100,10913,10664], np.uint64)
        tsz = 2048
        for r in range(6):
            for c in range(6):
                out[:,r*tsz:(r+1)*tsz,c*tsz:(c+1)*tsz] = read_h5(f'{Dd}/ffn/0000/%d_%d.h5'%(r,c))[:100]
        write_h5(f'{Do}/ffn.h5', out)
    elif opt == '2.01':
        ratio, numv = 0.5, 10000
        sz = 1024
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
    elif opt == '2.02':
        ratio = 0.5
        sz = 1024
        mask = read_h5(f'{Do}/ffn_{sz}_r{ratio}.h5')
        if sz == 1024:
            gid = [78,43,186,119,138,210,448,425,1128,703,553,816,986,96]
        mask = seg_remove_id(mask, gid)        
        write_h5(f'{Do}/ffn_{sz}_r{ratio}.h5', mask)
    elif opt == '2.1': # aff
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
    elif opt == '2.11': # aff
        aff = read_h5(f'{Do}/aff.h5');sn='aff'
        for sz in szs:
            out = aff[:, :, center[0]-sz//2:center[0]+sz//2, \
                         center[1]-sz//2:center[1]+sz//2]
            write_h5(f'{Do}/{sn}_{sz}.h5', out)
    elif opt[:3] == '2.3': # 1024
        from waterz.region_graph import merge_id
        sz = 1024
        sz = 4096 
        zz = 100
        # 1-sided iou is enough
        # high thres -> has to be best buddy
        iou_thres = 0.8 
        sz_thres = 0.6
        #sz_thres = 0.8
        z_thres = 5
        z_thres = 1
        rg_thres = 32
        os_thres = 0.6
        os_sz = 128
        # rg_thres = 256
        soma_id0 = int(2e7)
        if opt == '2.3':
            from em_util.eval import *
            fn = f'{Do}/ffn_{sz}.h5'
            
            #fn = f'{Do}/soma2d_{sz}_iou0.8_sz0.8.h5'
            #
            fn = f'{Do}/soma2d_{sz}_iou0.8_sz0.6_z5.h5'            
            #fn = f'{Do}/v0/soma2d_{sz}_iou0.8.h5'
            #fn = f'{Do}/v0/soma2d_{sz}_iou0.8_sz0.6.h5'
            #fn = f'{Do}/v0/soma2d_{sz}_iou0.8_sz0.6_z1.h5'
            #fn = f'{Do}/soma2d_{sz}_iou0.8.h5'
            #fn = f'{Do}/soma2d_{sz}_iou0.8_sz0.6.h5'
            #fn = f'{Do}/soma2d_{sz}_iou0.8_sz0.6_z5_aff1.h5'
            
            # tracking-based
            fn = f'{Do}/soma2d_{sz}_iou0.8_sz0.6_z1.h5'                        
            #fn = f'{Do}/soma2d_{sz}_iou0.8_sz0.6_z1_os0.6.h5'
            #fn = f'{Do}/v0/soma2d_{sz}_iou0.8_sz0.6_z1_os0.6.h5'            
            # mask = read_h5(fn)
            
            # vector processing
            #fn = f'{Do}/soma2d_{sz}_iou0.8_sz0.6_bb256_v2.h5'
            fn = f'{Do}/soma2d_{sz}_iou0.8_sz0.6_rg32_v2.h5'
            #fn = f'{Do}/soma2d_{sz}_iou0.8_sz0.6_rg32_bb_v2.h5'
            #fn = f'{Do}/soma2d_{sz}_iou0.8_sz0.6_rg32_bb_os0.6_v2.h5'
            mask = read_h5(fn, ['mask'])
            do_eval = True
            #do_eval = False
            if do_eval:
                gt = read_h5(f'{Do}/ffn_{sz}_r0.5.h5')
                # stats = adapted_rand(mask, gt, True)
                # print('%.2f, %.2f, %.2f' % (stats[0], stats[1], stats[2]))
                iou = seg_to_iou(gt, mask)
                score = iou[:, 4].astype(float) / (iou[:, 2] + iou[:, 3] - iou[:, 4])
                # false split                                
                print(iou[score<0.5, 1])
                # false merge
                print(iou[iou[:,3]-iou[:,4] > 0.5 * iou[:,4], 1])
                # bin iou score
                print(np.histogram(score,[0,0.5,0.8,1]))                
                # print(iou[iou[:,1]==0])
            else:
                count = seg3d_to_zcount(mask)
                print((count[1]==100).sum(),((count[1]>=50)*(count[1]<100)).sum(), ((count[1]>=30)*(count[1]<50)).sum())            
            # print_arr(count[0][count[1]==1])
        elif opt == '2.30': # iou
            for sz in szs:
                seg = read_h5(f'{Do}/soma2d_{sz}.h5')
                get_seg = lambda x : seg[x]
                sn = f'{Do}/soma2d_{sz}_iouF.h5'
                import pdb;pdb.set_trace()
                if not os.path.exists(sn): 
                    iou = segs_to_iou(get_seg, range(seg.shape[0]))
                    write_h5(sn, iou)        
                
                sn = f'{Do}/soma2d_{sz}_iouB.h5'
                if not os.path.exists(sn): 
                    iou = segs_to_iou(get_seg, range(seg.shape[0])[::-1])
                    write_h5(sn, iou)
        elif opt == '2.301': # zaff            
            import waterz
            aff_thres = 1
            for sz in szs:
                sn = f'{Do}/soma2d_{sz}'
                mask = read_h5(f'{sn}.h5')            
                if not os.path.exists(f'{sn}-rg.h5'): 
                    aff = read_h5(f'{Do}/aff_{sz}.h5') 
                    rg = waterz.getRegionGraph(aff, mask, 2, "aff75_his256_ran255", rebuild=False)
                    write_h5(f'{sn}-rg.h5', list(rg))
            
        elif opt == '2.39':
            # combine 2.31 and 2.32
            mask = read_h5(f'{Do}/soma2d_{sz}.h5')
            iou = read_h5(f'{Do}/soma2d_{sz}_iouF.h5')
            iou = np.vstack(iou)            
            # step 1
            # ids that are not related to soma
            iou = iou[(iou[:,:2] > soma_id0).max(axis=1) == 0]
            iou = iou[iou[:,1]-iou[:,0]!=0]
            # relabeled soma_id -> exist across multiple z
            
            score = iou[:, 4].astype(float) / (iou[:, 2] + iou[:, 3] - iou[:, 4])
            gid = iou[score >= iou_thres, :2].astype(np.uint32)

            # step 2: singleton (not in the pairs above)
            score2 = iou[:, 2].astype(float) / iou[:, 3]
            gid2 = iou[(score2 >= sz_thres)* (score2 <= 1/sz_thres), :2].astype(np.uint32)
            singleton = np.in1d(gid2.ravel(), gid.ravel(), invert=True).reshape(gid2.shape).max(axis=1)
            gid = np.vstack([gid, gid2[singleton]])

            # step 3: check rg
            rg_id, rg_score = read_h5(f'{Do}/soma2d_{sz}-rg.h5')
            arr2str = lambda x: f'{min(x)}-{max(x)}'            
            score = get_query_count_dict(rg_id, arr2str, rg_score, gid)
            gid = gid[score < rg_thres]
            
                         
            relabel = merge_id(gid[:, 0], gid[:, 1])
            mask[mask<len(relabel)] = relabel[mask[mask<len(relabel)]]
            write_h5(f'{Do}/soma2d_{sz}_iou{iou_thres}_sz{sz_thres}_rg{rg_thres}_v2.h5', [mask, relabel], ['mask', 'relabel'])               
        elif opt == '2.390': # bbox conenct + rg + best buddy
            mask, relabel = read_h5(f'{Do}/soma2d_{sz}_iou{iou_thres}_sz{sz_thres}_rg{rg_thres}_v2.h5')

            # 1. get candidate seg ids
            bbox = compute_bbox_all(mask, True)
            bbox = bbox[bbox[:,0] < soma_id0] 
            num_b = (bbox[:,1:-1:2] == 0).sum(axis=1) + (bbox[:,2] == zz-1) + (bbox[:,4::2] == sz-1).sum(axis=1)            
            to_merge = bbox[num_b<=1, 0]
            to_orphan = bbox[num_b==0, 0]

            # 2. get unique iouF pairs
            iou = read_h5(f'{Do}/soma2d_{sz}_iouF.h5')            
            for z in range(len(iou)):
                iou[z] = iou[z][:,:3]
                iou[z][:, 2] = z
            iou = np.vstack(iou)            
            iou = iou[(iou[:,:2] > soma_id0).max(axis=1) == 0]
            # unique pairs after relabel
            iou_new = iou[:, :2].copy()
            iou_new[iou_new < len(relabel)] = relabel[iou_new[iou_new < len(relabel)] ] 
            is_unique = (iou_new[:,0]-iou_new[:,1]) != 0
            iou_unique = iou_new[is_unique]
            # import pdb;pdb.set_trace()
            # iou_unique[iou_unique[:,0]==7600]
            
            # 3. bbox touches
            # both are to_merge or one of them is to_orphan
            # is_candidate = np.in1d(iou_unique.ravel(), to_merge).reshape(-1,2).min(axis=1)
            # is_candidate += np.in1d(iou_unique.ravel(), to_orphan).reshape(-1,2).max(axis=1)
            # do all
            # is_candidate[:] = True
             
            # match bbox
            iou_z = iou[is_unique, 2]            
            # iou_z[iou_unique[:,0]==7600]
            # is_last[iou_unique[:,0]==7600]
            is_last = get_query_count(bbox[:,0], bbox[:,2], iou_unique[:,0]) == iou_z
            is_first = get_query_count(bbox[:,0], bbox[:,1], iou_unique[:,1]) == (iou_z + 1)            
            mid = iou[is_unique,:2][is_last*is_first]
            #mid = iou[is_unique,:2][is_candidate*is_last*is_first]
            
            # relabel[mid[relabel[mid[:,0]]==7600]]
            # import pdb;pdb.set_trace()
             
            # 4. check affinity
            rg_id, rg_score = read_h5(f'{Do}/soma2d_{sz}-rg.h5')
            arr2str = lambda x: f'{min(x)}-{max(x)}'            
            score = get_query_count_dict(rg_id, arr2str, rg_score, mid)            
            mid = mid[score < rg_thres]            
            # import pdb;pdb.set_trace()
            
            # 5: iou-best buddy
            iou = read_h5(f'{Do}/soma2d_{sz}_iouB.h5')
            for z in range(len(iou)):
                iou[z] = iou[z][:,:2]
            iou = np.vstack(iou)
            arr2str2 = lambda x: f'{x[0]}-{x[1]}'
            mid_bb = get_query_in(iou[:,:2], arr2str2, mid[:, ::-1]) 
            mid = mid[mid_bb].astype(np.uint32)
            mid[mid < len(relabel)] = relabel[mid[mid < len(relabel)]] 
            # import pdb;pdb.set_trace()
            
            relabel = merge_id(mid[:, 0], mid[:, 1])
            mask[mask<len(relabel)] = relabel[mask[mask<len(relabel)]]
            write_h5(f'{Do}/soma2d_{sz}_iou{iou_thres}_sz{sz_thres}_rg{rg_thres}_bb_v2.h5', [mask, relabel], ['mask', 'relabel'])            
        
        elif opt == '2.391':            
            # merge horizontal branches
            relabel0 = read_h5(f'{Do}/soma2d_{sz}_iou{iou_thres}_sz{sz_thres}_rg{rg_thres}_v2.h5', ['relabel'])
            mask, relabel = read_h5(f'{Do}/soma2d_{sz}_iou{iou_thres}_sz{sz_thres}_rg{rg_thres}_bb_v2.h5')

            # 1. get candidate seg ids
            bbox = compute_bbox_all(mask, True)
            bbox = bbox[bbox[:,0] < soma_id0]
            # hack for 1024
            bbox = bbox[bbox[:,0] != 266]
            
            num_b = (bbox[:,1:-1:2] == 0).sum(axis=1) + (bbox[:,2] == zz-1) + (bbox[:,4::2] == sz-1).sum(axis=1)
            to_merge = bbox[num_b<=1, 0]
            to_orphan = bbox[num_b==0, 0]
            

            # 2. get unique iou pairs
            iou = np.vstack(read_h5(f'{Do}/soma2d_{sz}_iouF.h5'))            
            iou = np.vstack([iou] + read_h5(f'{Do}/soma2d_{sz}_iouB.h5'))
            iou = iou[(iou[:,:2] > soma_id0).max(axis=1) == 0]
            # hack for 1024
            iou = iou[(iou[:,:2] == 266).max(axis=1) == 0]
                                    
            # unique pairs after relabel
            iou_new = iou.copy()
            tmp = iou_new[:, :2] 
            tmp[tmp < len(relabel0)] = relabel0[tmp[tmp < len(relabel0)]]
            tmp[tmp < len(relabel)] = relabel[tmp[tmp < len(relabel)]]
            is_unique = (iou_new[:,0]-iou_new[:,1]) != 0
            iou_unique = iou_new[is_unique]
            
            # both are to_merge or one of them is to_orphan
            is_candidate = np.in1d(iou_unique[:,:2].ravel(), to_merge).reshape(-1,2).min(axis=1)
            # is_candidate += np.in1d(iou_unique[:,:2].ravel(), to_orphan).reshape(-1,2).max(axis=1)
            
            score_1side = iou_unique[:, 4] / iou_unique[:, 2]
            mid = iou[is_unique,:2][is_candidate * (score_1side >= os_thres) * (iou_unique[:,4] >= os_sz)]            
            
            # 4. check affinity
            rg_id, rg_score = read_h5(f'{Do}/soma2d_{sz}-rg.h5')
            arr2str = lambda x: f'{min(x)}-{max(x)}'            
            score = get_query_count_dict(rg_id, arr2str, rg_score, mid)            
            mid = mid[score < rg_thres]            
                         
            mid[mid < len(relabel0)] = relabel0[mid[mid < len(relabel0)]]
            mid[mid < len(relabel)] = relabel[mid[mid < len(relabel)]]
            mid = np.hstack([mid.min(axis=1).reshape(-1,1), mid.max(axis=1).reshape(-1,1)])
            mid = np.unique(mid, axis=1)
            
            # 5. ordered agglomeration
            #ratio = ((bbox[:,2::2] - bbox[:,1:-1:2] + 1) * [2,1,1]).max(axis=1)
            ratio = ((bbox[:,2::2] - bbox[:,1:-1:2] + 1) / [zz, sz, sz]).max(axis=1)
            num_z = bbox[:,2]-bbox[:,1]+1
            # remove singleton used more than once
            uid, uc = np.unique(mid.ravel(), return_counts=True)
            uid_z = get_query_count(bbox[:,0], num_z, uid)
            bid = uid[(uid_z==1) * (uc>1)]
            mid = mid[np.in1d(mid.ravel(), bid, invert=True).reshape(-1,2).min(axis=1)]
            uid = uid[np.in1d(uid, bid, invert=True)]
                        
            uid_r = get_query_count(bbox[:,0], ratio, uid)            
            uid_ratio = dict(zip(uid, uid_r))
            uid_num_b = dict(zip(uid, get_query_count(bbox[:,0], num_b, uid)))
            uid_done = dict(zip(uid, np.zeros(len(uid))))
            mid_sel = []
            uid_sorted = uid[np.argsort(-uid_r)]
            for i in uid_sorted:
                #print(i)
                # tt=''
                if uid_done[i] == 0:
                    # uid[i] involved and not done
                    total_num_b = 0                    
                    todo_id = i
                    uid_done[todo_id] = 1
                    total_num_b = uid_num_b[todo_id]
                    todo_stat = np.zeros([0, 3])
                    while total_num_b < 2:                        
                        new_mid = mid[(mid == todo_id).max(axis=1)]                    
                        new_id = new_mid[new_mid != todo_id]
                        new_id_ratio = np.array([uid_ratio[x] if uid_done[x]==0 else 0 for x in new_id])
                        new_stat = np.hstack([np.ones([len(new_id),1])*todo_id, new_id.reshape(-1,1), new_id_ratio.reshape(-1,1)])
                        todo_stat = np.vstack([todo_stat, new_stat])
                    
                        if len(todo_stat)==0:
                             break
                        else:
                            todo_sel = np.argmax(todo_stat[:,2])
                            if todo_stat[todo_sel, 2] == 0:
                                break
                        todo_id = int(todo_stat[todo_sel, 1])
                        uid_done[todo_id] = 1                        
                        total_num_b += uid_num_b[todo_id]
                        mid_sel += [list(todo_stat[todo_sel,:2])]
                        todo_stat[todo_sel] = 0
                        """
                        if i == 8556:
                           tt+=str(todo_id)+','
                        """
                        # np.vstack(mid_sel).astype(int)
                """
                if i == 8556:
                    print(tt[:-1])
                    import pdb;pdb.set_trace() 
                """  
            mid = np.vstack(mid_sel).astype(np.uint32)
            relabel = merge_id(mid[:, 0], mid[:, 1])
            mask[mask<len(relabel)] = relabel[mask[mask<len(relabel)]]
            write_h5(f'{Do}/soma2d_{sz}_iou{iou_thres}_sz{sz_thres}_rg{rg_thres}_bb_os{os_thres}_v2.h5', [mask, relabel], ['mask', 'relabel'])
                
        elif opt == '2.3911': # compute tubularity
            mask = read_h5(f'{Do}/soma2d_{sz}_iou{iou_thres}_sz{sz_thres}_rg{rg_thres}_bb_v2.h5', ['mask'])
            bbox = compute_bbox_all(mask, True)
            ratio = (bbox[:,2::2] - bbox[:,1:-1:2] + 1) / [zz, sz, sz]            
            uid = bbox[ratio > 0.3, 0]
            tube = np.zeros([uid.max()+1,3])
            # https://nirpyresearch.com/robust-pca/            
            
                
            
            
            
        elif opt == '2.399':
            # greedy merge branch
            mask, relabel = read_h5(f'{Do}/soma2d_{sz}_iou{iou_thres}_sz{sz_thres}_v2.h5')
            iou = read_h5(f'{Do}/soma2d_{sz}_iouF.h5')            
            iou = np.vstack(iou)
            # ids that are not related to soma
            iou = iou[(iou[:,:2] > soma_id0).max(axis=1) == 0]
            
            # step 1.1: get number of borders
            bbox = compute_bbox_all(mask, True)
            bbox[bbox[:,0] < soma_id0]
            num_b = (bbox[:,1:-1:2] == 0).sum(axis=1) + (bbox[:,2] == zz-1) + (bbox[:,4::2] == sz-1).sum(axis=1)
            num_z = bbox[:,2]-bbox[:,1]+1
            
            # step 1.2: get number of neighbor
            iou_new = iou[:, :2].copy()
            iou_new[iou_new < len(relabel)] = relabel[iou_new[iou_new < len(relabel)] ] 
            iou_nb = iou_new[iou_new[:,0]-iou_new[:,1]!=0]
            # remove singleton
            iou_nb = iou_nb[np.in1d(iou_nb.ravel(), bbox[num_z==1,0], invert=True).reshape(-1,2).min(axis=1)]
            
            # get candidate segs
            nb_ind, nb_count = np.unique(iou_nb.ravel(), return_counts=True)
            num_b_nb = get_query_count(nb_ind, nb_count, bbox[:,0])
            gid = (num_b_nb==2) * (num_b==0) + (num_b_nb==1) * (num_b==1)
            #import pdb;pdb.set_trace()
            # num_b_nb[bbox[:,0]==8769]
            # step 2: rg good
            iou_g = iou[np.in1d(iou_new.ravel(), bbox[gid,0]).reshape(-1,2).min(axis=1) * (iou_new[:,0]-iou_new[:,1]!=0), :2]            
            rg_id, rg_score = read_h5(f'{Do}/soma2d_{sz}-rg.h5')
            
            arr2str = lambda x: f'{min(x)}-{max(x)}'
            
            score = get_query_count_dict(rg_id, arr2str, rg_score, iou_g)            
            mid = iou_g[score < rg_thres]
            import pdb;pdb.set_trace()
            
            # step 3: iou-best buddy
            iou = read_h5(f'{Do}/soma2d_{sz}_iouB.h5')
            iou = np.vstack(iou)
            mid_bb = get_query_in(iou[:,:2], arr2str, mid[:, ::-1])
            
            mid = mid[mid_bb].astype(np.uint32)
            mid[mid < len(relabel)] = relabel[mid[mid < len(relabel)]]            
            import pdb;pdb.set_trace()
             
            relabel = merge_id(mid[:, 0], mid[:, 1])
            mask[mask<len(relabel)] = relabel[mask[mask<len(relabel)]]
            write_h5(f'{Do}/soma2d_{sz}_iou{iou_thres}_sz{sz_thres}_bb{rg_thres}_v2.h5', [mask, relabel], ['mask', 'relabel'])
                
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
            
            #count = seg3d_to_zcount(mask)
            #print((count[1]==100).sum(),((count[1]>=50)*(count[1]<100)).sum(), ((count[1]>=10)*(count[1]<50)).sum())
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
                # no double usage
                done = np.zeros([bbox[:,0].max()+1], np.uint8)                
                done[soma_id0:] = 1 # no merge soma
                # done = (num_z >= z_thres) * (num_b>=2)
                todo_order = np.argsort(-num_z[todo])
                mid = []
                for bb in bbox[todo][todo_order]:
                    if done[bb[0]] == 0:
                        # complete top
                        s0 = bb[0]
                        z0 = bb[1]                    
                        while z0 != 0:
                            iou = iouB[zz-1-z0]                                
                            cid = iou[iou[:,0]==s0, 1][0]
                            if done[cid] == 0 and bbox[bbox[:,0]==cid, 2] == bb[1] - 1:
                                # best buddy
                                iou = iouF[z0-1]
                                if s0 == iou[iou[:,0]==cid, 1][0]:                                
                                    mid.append([s0, cid])
                                    z0 = bbox[bbox[:,0]==cid, 1][0]
                                    s0 = cid
                                    done[cid] = 1
                                else:
                                    break
                            else:
                                break
                    
                        z0 = bb[2]
                        s0 = bb[0]
                        while z0 != zz-1:                             
                            iou = iouF[z0]
                            cid = iou[iou[:,0]==s0, 1][0]
                            # cid in todo_id ?
                            if done[cid] == 0 and bbox[bbox[:,0]==cid, 1] == bb[2] + 1:
                                iou = iouB[zz-1-z0-1]
                                if s0 == iou[iou[:,0]==cid, 1][0]:
                                    mid.append([s0, cid])
                                    z0 = bbox[bbox[:,0]==cid, 2][0]
                                    s0 = cid
                                    done[cid] = 1
                                else:
                                    break
                            else:
                                break
                        done[bb[0]] = 1
                            
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
            #import pdb;pdb.set_trace()
            relabel = merge_id(mid[:, 0], mid[:, 1])
            mask = read_h5(f'{sn}.h5')
            mask[mask<len(relabel)] = relabel[mask[mask<len(relabel)]]
            write_h5(f'{sn}_z{z_thres}.h5', mask)            
            
            bbox = compute_bbox_all(mask, True)
            write_h5(f'{sn}_z{z_thres}-bb.h5', bbox)
  
        elif opt in ['2.37']: # horizontal grafted            
            sn = f'{Do}/soma2d_{sz}_iou{iou_thres}_sz{sz_thres}_z{z_thres}'
            bbox = read_h5(f'{sn}-bb.h5')
            num_b = (bbox[:,1:-1:2] == 0).sum(axis=1) + (bbox[:,2] == zz-1) + (bbox[:,4::2] == sz-1).sum(axis=1)            
            # xyz
            ratio = (bbox[:,2::2] - bbox[:,1:-1:2] + 1) / [100, sz, sz]
            ratio_m = ratio.max(axis=1)
            todo = (ratio_m >= 0.3) * (num_b<2)
            todo_order = np.argsort(-ratio_m[todo])
            import pdb;pdb.set_trace()
            mask = read_h5(f'{sn}.h5')
            iouF = read_h5(f'{sn}-iouF.h5')
            iouB = read_h5(f'{sn}-iouB.h5')
            iouF_all = np.vstack(iouF)
            iouB_all = np.vstack(iouB)
            num_skip = 2
            
            # import pdb;pdb.set_trace()
            if opt == '2.37': # object-centric
                done = np.zeros([bbox[:,0].max()+1], np.uint8)                
                done[soma_id0:] = 1 # no merge soma
                # done = (num_z >= z_thres) * (num_b>=2)
                mid = []                
                for bb in bbox[todo][todo_order]:
                    if done[bb[0]] == 0:
                        # z: skip slices
                        s0, z0 = bb[0], bb[1]
                        while z0 > 1:
                            # connect to top
                            do_connect = False
                            b0 = compute_bbox(mask[z0]==s0)
                            for z in range(min(num_skip, z0-1)):
                                ui, uc = np.unique(mask[z0-z-2, b0[0]:b0[1]+1, b0[2]:b0[3]+1], return_counts=True)
                                uc[ui==0] = 0
                                cid = ui[np.argmax(uc)]
                                # if bbox match and best buddy
                                if bbox[bbox[:,0]==cid, 2] == z0-z-2:
                                    b1 = compute_bbox(mask[z0-2-z] == cid)
                                    ui2, uc2 = np.unique(mask[z0, b1[0]:b1[1]+1, b1[2]:b1[3]+1], return_counts=True) 
                                    uc2[ui2==0] = 0
                                    if s0 == ui2[np.argmax(uc2)]:
                                        mid.append([s0, cid])
                                        s0, z0 = cid, bbox[bbox[:,0]==cid, 1][0] 
                                        do_connect = True
                                        break
                            if not do_connect:                                
                                break
                        s0, z0 = bb[0], bb[2]                        
                        while z0 < zz-2:
                            do_connect = False
                            # connect to bottom 
                            b0 = compute_bbox(mask[z0]==s0)
                            for z in range(min(num_skip, zz-2-z0)):
                                ui, uc = np.unique(mask[z0+z+2, b0[0]:b0[1]+1, b0[2]:b0[3]+1], return_counts=True)
                                uc[ui==0] = 0
                                cid = ui[np.argmax(uc)]
                                # if bbox match and best buddy
                                if bbox[bbox[:,0]==cid, 1] == z0+z+2:
                                    b1 = compute_bbox(mask[z0+z+2] == cid)
                                    ui2, uc2 = np.unique(mask[z0, b1[0]:b1[1]+1, b1[2]:b1[3]+1], return_counts=True) 
                                    uc2[ui2==0] = 0
                                    if s0 == ui2[np.argmax(uc2)]:
                                        mid.append([s0, cid])
                                        s0, z0 = cid, bbox[bbox[:,0]==cid, 2][0] 
                                        do_connect = True
                                        break
                            if not do_connect:                                
                                break
                        done[bb[0]] = 1
                        import pdb;pdb.set_trace()                
                                    
                                
                            
                        # complete top
                        s0 = bb[0]
                        z0 = bb[1]                    
                        while z0 != 0:
                            
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
            #import pdb;pdb.set_trace()
            relabel = merge_id(mid[:, 0], mid[:, 1])
            mask = read_h5(f'{sn}.h5')
            mask[mask<len(relabel)] = relabel[mask[mask<len(relabel)]]
            write_h5(f'{sn}_os{os_thres}.h5', mask)            
            
            bbox = compute_bbox_all(mask, True)
            write_h5(f'{sn}_os{os_thres}-bb.h5', bbox)

      
        elif opt in ['2.36']: # 1-sided iou
            # not good
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
            #import pdb;pdb.set_trace()
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
            mask = read_h5(f'{sn}.h5')
            
            if not os.path.exists(f'{sn}-rg.h5'): 
                aff = read_h5(f'{Do}/aff_{sz}.h5') 
                rg = waterz.getRegionGraph(aff, mask, 2, "aff75_his256_ran255", rebuild=False)
                write_h5(f'{sn}-rg.h5', list(rg))
            else:
                rg = read_h5(f'{sn}-rg.h5') 
            
            """            
            if opt == '2.35': 
                # get long objects
                
            elif opt == '2.351':
                # affinity is not reliable
                gid = np.in1d(rg[0][:,0], bid) * np.in1d(rg[0][:,1], bid) * (rg[1]<=aff_thres)            
                mid = rg[0][gid]
            import pdb;pdb.set_trace()
            relabel = merge_id(mid[:, 0], mid[:, 1])
            mask[mask<len(relabel)] = relabel[mask[mask<len(relabel)]]
            write_h5(f'{sn}_aff{aff_thres}.h5', mask)
            """ 
            
            
            