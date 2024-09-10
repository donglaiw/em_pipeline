import os
import numpy as np
from .task import Task

from em_util.io import read_h5, write_h5
from waterz.region_graph import merge_id


def somaBFS(rg_id, soma_id0):
    # ii: Nx2
    # soma_id0: ids that are bigger than this are soma ids
    
    ii = rg_id.copy()
    # remove self-linking pairs
    ii = ii[ii[:,0] != ii[:,1]]
    
    # 1: soma-others that can't be linked
    nolink = np.zeros(ii.shape[0])
    mapping = np.arange(ii[ii<soma_id0].max()+1).astype(np.uint32)

    while True:
        # find ids that are merged to soma
        # remove pairs that connect somas
        soma_id_count = (ii > soma_id0).sum(axis=1)                
        if ((soma_id_count == 1) * (nolink == 0)).sum() == 0:
            # no more soma linked seg ids        
            break
        else:
            # soma-others: find unique rows
            # move small index in front
            ii[:,0], ii[:,1] = ii.min(axis=1), ii.max(axis=1)
            ii_g, ii_g_pos = np.unique(ii[soma_id_count == 1], axis=0,  return_inverse=True)
            
            # find seg ids that connect to different soma ids
            ui, uc = np.unique(ii_g[:, 0], return_counts=True) 
            # mark nolink
            bid = np.where(np.in1d(ii_g[:,0], ui[uc>1]))[0]
            nolink[np.where(soma_id_count == 1)[0][np.in1d(ii_g_pos, bid)]] = 1
                        
            # merge to soma
            gid = np.in1d(ii_g[:,0], ui[uc==1])
            mapping[ii_g[gid,0]] = ii_g[gid,1]
            
            ii[ii<soma_id0] = mapping[ii[ii<soma_id0]]
        print(nolink.sum(), ((ii > soma_id0).sum(axis=1)==1).sum(), ((ii > soma_id0).sum(axis=1)==2).sum())
    # merge ids that are not connected to soma ids
    mapping2 = merge_id(rg_id[soma_id_count==0,0], rg_id[soma_id_count==0,1], id_thres=soma_id0)
    mid = np.unique(rg_id[soma_id_count==0])
    mapping[mid] = mapping2[mid]
    
    bid = rg_id[nolink==1]
    return mapping, bid


class RegionGraphChunkTask(Task):
    # for debugging purpose
    def __init__(self, conf_file, name='rg'):
        super().__init__(conf_file, name)
        
    def get_job_num(self):        
        return self.get_zchunk_num() - 1
    
    def run(self, job_id, job_num):
        # convert affinity to waterz and region graph
        output_name = self.get_output_name(file_name=f'chunk_{job_id}_{job_num}.h5')
        if not os.path.exists(output_name):
            waterz_seg_name = self.get_output_name(folder_name='waterz', file_name=f'{job_id}_{job_num}_soma2d.h5')
            rg_id, rg_score = read_h5(waterz_seg_name,['id','score'])            
            bad_id = [np.zeros([0,2], np.uint32)] * (self.param['thres_z'] + 1)
            for i, thres in enumerate(range(self.param['thres_z'] + 1)):                
                if thres in rg_score: 
                    mapping, bad_id[i] = somaBFS(rg_id[rg_score==thres], self.conf['mask']['soma_id0'])
                    rg_id = mapping[rg_id]
            write_h5(output_name, [mapping, np.vstack(bad_id)], ['mapping', 'bid'])
            

class RegionGraphBorderTask(Task):
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
