import os
import numpy as np
import h5py
import zarr
import yaml
from em_util.io import read_image, mkdir, write_h5, read_h5
import waterz
import zwatershed
import cc3d, fastremap

class Task(object):
    def __init__(self, conf_file, name=''):        
        with open(conf_file, 'r') as file:
            self.conf = yaml.safe_load(file)
        self.name = name
        if name in self.conf:
            self.param = self.conf[name]
    
    def get_job_num(self):
        pass

    def get_zchunk_num(self):
        num_z = self.param['num_z']
        total_z = self.conf['im']['shape'][0]
        return (total_z + num_z - 1) // num_z      
    
    def get_output_name(self, folder_name=None, file_name=None, job_id=0, job_num=0):
        if folder_name is None:
            folder_name = self.name
        if file_name is None:
            file_name = f'{job_id}_{job_num}.h5'
        output_name = os.path.join(self.conf['output']['path'], folder_name, file_name)
        mkdir(output_name, 'parent')
        return output_name
   
    def run(self, job_id, job_num):
        pass


        
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
            aff = zarr.open(self.conf['aff']['path'], mode='r')[:3, z0:z0+num_z]
            aff[aff < self.param['low']] = 0
            
            print('apply mask')
            border_width = self.conf['mask']['border-width'] 
            for z in range(aff.shape[1]):
                # blood vessel
                mask = read_image(self.conf['mask']['blood-vessel'] % (z0+z)) == 0
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
                                        return_rg=True, rebuild=True)
            
            print('merge small seg crumb')
            rg_id = rg[0][0].astype(np.uint64)
            rg_sc = rg[0][1].astype(np.float32) / 255.
            ui, uc = np.unique(seg[0], return_counts=True)
            uc_rl = np.zeros(int(ui.max()) + 1, np.uint64)
            uc_rl[ui] = uc
            gid = rg_id.max(axis=1) <= ui.max()
            rg_id = rg_id[gid]
            rg_sc = rg_sc[gid]
            zwatershed.zw_merge_segments_with_function(seg[0], 1-rg_sc, rg_id[:,0], rg_id[:,1], uc_rl, \
                self.param['small_size'], self.param['small_aff'], self.param['small_dust'], -1)
            
            print('connected component')
            seg = cc3d.connected_components(seg[0])
            
            print('remove border seg')
            rl_seg = np.arange(seg.max()+1, dtype=seg.dtype)
            for z in range(seg.shape[0]):                    
                bd = np.loadtxt(self.conf['mask']['border'] % (z0+z)).astype(int)
                ii = [None]*4
                ii[0] = np.unique(seg[z, bd[0]+border_width])
                ii[1] = np.unique(seg[z, bd[1]-border_width-1])
                ii[2] = np.unique(seg[z, :, bd[2]+border_width])
                ii[3] = np.unique(seg[z, :, bd[3]-border_width-1])
                ii = np.unique(np.hstack(ii))
                rl_seg[ii] = 0    
            seg = rl_seg[seg]
            
            print('relabel seg')
            seg = fastremap.renumber(seg, in_place=True)
            
            print('create region graph') 
            rg = waterz.getRegionGraph(aff, seg, 1, self.param['mf'], rebuild=False)
           
            print('save output')  
            write_h5(output_name, [seg] + rg, ['seg', 'id', 'score'])
            
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
             
class RegionGraphInitTask(Task):
    def __init__(self, conf_file, name='rg'):
        super().__init__(conf_file, name)
        
    def get_job_num(self):        
        return self.get_zchunk_num() - 1
    
    def run(self, job_id, job_num):
        # convert affinity to waterz and region graph
        output_name = self.get_output_name(file_name='border', job_id=job_id, job_num=job_num)
        if not os.path.exists(output_name):
            waterz_seg_name = self.get_output_name(folder_name='waterz', file_name='%d_%d.h5')
            waterz_stats_name = self.get_output_name(folder_name='waterz', file_name='stats.txt')             
            count = np.loadtxt(waterz_stats_name).astype(int)
            seg0 = np.array(h5py.File(waterz_seg_name % (job_id, job_num), 'r')['main'][-1]).astype(np.uint64)
            seg0[seg0 > 0] += count[job_id]
            seg1 = np.array(h5py.File(waterz_seg_name % (job_id+1, job_num), 'r')['main'][0]).astype(np.uint64)
            seg1[seg1 > 0] += count[job_id+1]
            
            
            aff = zarr.open(self.conf['aff']['path'], mode='r')[:3, (job_id+1) * self.param['num_z']]
            aff[aff < self.param['low']] = 0

            rg = waterz.getRegionGraph(aff, np.stack([seg0, seg1], axis=0), 2, self.conf['waterz']['mf'], rebuild=False)
            
            write_h5(output_name, rg, ['id','score'])
            
class RegionGraphAllTask(Task):
    def __init__(self, conf_file, name='rg'):
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

def get_class(task, conf):
    task0 = task if '-' not in task else task[:task.find('-')]
    if task == "waterz":
        return WaterzTask(conf, task0)
    elif task == "waterz-stats":
        return WaterzStatsTask(conf, task0)
    elif task == 'rg-border':
       return RegionGraphBorderTask(conf, task0) 
    elif task == 'rg-all':
       return RegionGraphAllTask(conf, task0) 
        