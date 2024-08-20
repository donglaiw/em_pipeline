from .task import Task
import os
import numpy as np
import h5py
import pickle
import waterz
import zwatershed
import cc3d, fastremap
from em_util.io import read_image, mkdir, write_h5, read_h5, read_vol

        
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
             