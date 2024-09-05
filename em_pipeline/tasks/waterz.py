from .task import Task
import os
import numpy as np
import h5py
import pickle
import waterz
import zwatershed
import cc3d, fastremap
import mahotas
from scipy.ndimage import zoom
from em_util.io import read_image, mkdir, write_h5, read_h5, read_vol, compute_bbox

class WaterzSoma2DTask(Task):
    def __init__(self, conf_file, name='waterz-soma'):
        super().__init__(conf_file, name)
        
    def get_job_num(self):
        return self.get_zchunk_num()
    
    def run(self, job_id, job_num):
        # convert affinity to waterz and region graph        
        output_name = self.get_output_name(file_name=f"{job_id}_{job_num}_soma2d.h5")
        if not os.path.exists(output_name):
            im_size = self.conf['im']['shape']
            output_folder = os.path.dirname(output_name)
            border_width = self.conf['mask']['border_width'] 
            num_z = self.param['num_z']
            z0 = num_z * job_id
            z1 = min(num_z * (job_id + 1), im_size[0])
            do_rebuild = True
            soma_fid = h5py.File(self.conf['mask']['soma'], 'r')['main']
            soma_ratio = self.conf['mask']['soma_ratio']
            print('\t get affinity')
            dask_array = pickle.load(open(self.conf['aff']['path'], "rb"))
            aff = dask_array[:3, z0: z1].compute()
            aff[aff < self.param['low']] = 0           
            
            seg_out = np.zeros(aff.shape[1:], np.uint32)
            max_id = 0
            soma_id0 = self.conf['mask']['soma_id0'] 
            for z in range(z0, z1):                
                print(f'{z:04d}')

                print('\t remove by blood vessel and myeline mask')
                # blood vessel
                mask_bv = read_image(self.conf['mask']['blood_vessel'] % (z)) == 0                
                # border
                bd = np.loadtxt(self.conf['mask']['border'] % (z)).astype(int)
                mask_bv[:bd[0]+border_width] = 0
                mask_bv[bd[1]-border_width:] = 0
                mask_bv[:, :bd[2]+border_width] = 0
                mask_bv[:, bd[3]-border_width:] = 0
                aff[:, z - z0] = aff[:, z - z0] * mask_bv
    
                print('\t run initial waterz')
                seg = waterz.waterz(aff[:, z-z0 : z-z0+1], self.param['thres'], merge_function = self.param['mf'], \
                                    aff_threshold = [self.param['low'], self.param['high']], \
                                    fragments_opt = self.param['opt_frag'], fragments_seed_nb = self.param['nb'],\
                                    bg_thres = self.param['bg_thres'], rebuild=do_rebuild)
                do_rebuild = False               
                # remap the 2D seg
                seg, _ = fastremap.renumber(seg[0][0], in_place=True)
                seg = seg.astype(np.uint32)
                seg_max = seg.max()                
                
                print('\t merge/split by soma mask')
                z_soma = int(np.round(z / soma_ratio[0]))
                mask_soma = zoom(np.array(soma_fid[z_soma]), soma_ratio[1:], order=0)[:im_size[1], :im_size[2]]
                
                if mask_soma.max() > 0:
                    relabel = np.arange(seg.max() + 1).astype(seg.dtype)
                    soma_ids = np.unique(mask_soma)
                    soma_ids = soma_ids[soma_ids>0]
                    seg_ids = [None] * len(soma_ids)
                    # fix false split
                    for i, soma_id in enumerate(soma_ids):
                        uid = np.unique(seg[mask_soma == soma_id])
                        seg_ids[i] = uid[uid > 0]
                        relabel[seg_ids[i]] = soma_id0 + soma_id
                    
                    # fix false merge
                    ui, uc = np.unique(np.hstack(seg_ids), return_counts=True)
                    for i in ui[uc > 1]:
                        soma_split = [soma_ids[j] for j in range(len(soma_ids)) if i in seg_ids[j]]
                        bb = compute_bbox(seg == i)
                        seg_split = seg[bb[0]: bb[1]+1, bb[2]: bb[3]+1]
                        seed_split = np.zeros(seg_split.shape, dtype=seg.dtype)
                        for j in soma_split:
                            seed_split[mask_soma[bb[0]: bb[1]+1, bb[2]: bb[3]+1] == j] = j
                        aff_split = aff[1:, z - z0, bb[0]: bb[1]+1, bb[2]: bb[3]+1]
                        boundary = 1.0 - 0.5*(aff_split[0].astype(np.float32) + aff_split[1]) / 255.0
                        out_split = mahotas.cwatershed(boundary, seed_split)
                        out_split[seg_split != i] = 0
                        print(f'\t split soma {soma_split}')
                        for j in soma_split:
                            seg_split[out_split == j] = soma_id0 + j
                        
                    seg[seg < soma_id0] = relabel[seg[seg < soma_id0]]                
                seg[(seg > 0) * (seg < soma_id0)] += max_id
                max_id += seg_max
                seg_out[z - z0] = seg
            
            print('\tcreate region graph') 
            rg = waterz.getRegionGraph(aff, seg_out, 1, self.param['mf'], rebuild=False)
            print('\t save output')  
            write_h5(output_name, [seg_out, rg[0], rg[1]], ['seg', 'id', 'score'])
        
class WaterzTask(Task):
    def __init__(self, conf_file, name='waterz'):
        super().__init__(conf_file, name)
        
    def get_job_num(self):
        return self.get_zchunk_num()
    
    def run(self, job_id, job_num):
        # convert affinity to waterz and region graph        
        output_name = self.get_output_name(job_id=job_id, job_num=job_num)
        if not os.path.exists(output_name):
            print('get affinity')
            num_z = self.param['num_z']
            z0 = num_z * job_id
            dask_array = pickle.load(open(self.conf['aff']['path'], "rb"))
            #aff = dask_array[:3, z0:z0+num_z, :500, :500].compute()
            aff = dask_array[:3, z0:z0+num_z].compute()
            aff[aff < self.param['low']] = 0
           
            print('apply mask')
            border_width = self.conf['mask']['border_width'] 
            for z in range(aff.shape[1]):
                # blood vessel
                mask = read_image(self.conf['mask']['blood_vessel'] % (z0+z)) == 0                
                # border
                bd = np.loadtxt(self.conf['mask']['border'] % (z0+z)).astype(int)
                mask[:bd[0]+border_width] = 0
                mask[bd[1]-border_width:] = 0
                mask[:, :bd[2]+border_width] = 0
                mask[:, bd[3]-border_width:] = 0
                aff[:,z] = aff[:,z] * mask
            del mask
        
            print('run waterz')
            seg, rg = waterz.waterz(aff, self.param['thres'], merge_function = self.param['mf'], \
                                        aff_threshold = [self.param['low'], self.param['high']], \
                                        fragments_opt = self.param['opt_frag'], fragments_seed_nb = self.param['nb'],\
                                        bg_thres = self.param['bg_thres'], return_rg=True, rebuild=True)                        
            if self.param['debug'] == 1:
                # debug mode
                write_h5(output_name, seg[0])
            else:
                seg = seg[0].astype(np.uint64)
                print('merge small seg crumb')            
                rg_id = rg[0][0].astype(np.uint64)
                rg_sc = rg[0][1].astype(np.float32) / 255.
                ui, uc = np.unique(seg, return_counts=True)
                uc_rl = np.zeros(int(ui.max()) + 1, np.uint64)
                uc_rl[ui] = uc
                gid = rg_id.max(axis=1) <= ui.max()
                rg_id = rg_id[gid]
                rg_sc = rg_sc[gid]
                zwatershed.zw_merge_segments_with_function(seg, 1-rg_sc, rg_id[:,0], rg_id[:,1], uc_rl, \
                    self.param['small_size'], self.param['small_aff'], self.param['small_dust'], -1)
                
                print('connected component')
                seg = cc3d.connected_components(seg)
                
                # problem: soma can be near border
                # print('remove border seg')
                # rl_seg = np.arange(seg.max()+1, dtype=seg.dtype)
                # for z in range(seg.shape[0]):                    
                #     bd = np.loadtxt(self.conf['mask']['border'] % (z0+z)).astype(int)
                #     ii = [None]*4
                #     ii[0] = np.unique(seg[z, bd[0]+border_width])
                #     ii[1] = np.unique(seg[z, bd[1]-border_width-1])
                #     ii[2] = np.unique(seg[z, :, bd[2]+border_width])
                #     ii[3] = np.unique(seg[z, :, bd[3]-border_width-1])
                #     ii = np.unique(np.hstack(ii))
                #     rl_seg[ii] = 0    
                # seg = rl_seg[seg]
                
                print('relabel seg')
                seg, _ = fastremap.renumber(seg, in_place=True)
                
                print('create region graph') 
                rg = waterz.getRegionGraph(aff, seg, 1, self.param['mf'], rebuild=False)
            
                print('save output')  
                write_h5(output_name, [seg, rg[0], rg[1]], ['seg', 'id', 'score'])
            
class WaterzStatsTask(Task):
    def __init__(self, conf_file, name='waterz'):
        super().__init__(conf_file, name)
        
    def get_job_num(self):        
        return 1

    def run(self, job_id, job_num):
        # convert affinity to waterz and region graph
        waterz_name = self.get_output_name(job_id=job_id, job_num=job_num)
        output_name = self.get_output_name(file_name='stats.txt')
        if not os.path.exists(output_name):       
            num_chunk = self.get_zchunk_num()
            count = np.zeros(num_chunk+1, np.uint32)
            for z in range(num_chunk):
                rg = read_h5(waterz_name % (z, num_chunk), 'id')
                count[z+1] = rg.max()
            count = np.cumsum(count)
            np.savetxt(output_name, count, '%d')            
             
