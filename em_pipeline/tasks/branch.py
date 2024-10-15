import os
import numpy as np
from .task import Task

from em_util.io import *
from em_util.seg import *
from waterz.region_graph import merge_id


class BranchChunkIoUTask(Task):
    # for debugging purpose
    def __init__(self, conf_file, name='rg'):
        super().__init__(conf_file, name)
        
    def get_job_num(self):        
        return self.get_zchunk_num(self.conf['waterz']['num_z'])
    
    def run(self, job_id, job_num):
        # convert affinity to waterz and region graph
        output_name = self.get_output_name(file_name=f'chunk_{job_id}_{job_num}_%s.h5')
        if not os.path.exists(output_name % 'iouB'):
            waterz_seg_name = self.get_output_name(folder_name='waterz', file_name=f'{job_id}_{job_num}_soma2d.h5')
            mask = read_h5(waterz_seg_name, ['seg'])
            get_seg = lambda x: mask[x]

            if not os.path.exists(output_name % 'iouF'):
                iou = segs_to_iou(get_seg, range(mask.shape[0]))
                write_h5(output_name % 'iouF', iou)
            
            iou = segs_to_iou(get_seg, range(mask.shape[0])[::-1])
            write_h5(output_name % 'iouB', iou)
            
class BranchChunkS1Task(Task):
    # for debugging purpose
    def __init__(self, conf_file, name='rg'):
        super().__init__(conf_file, name)
        
    def get_job_num(self):        
        return self.get_zchunk_num(self.conf['waterz']['num_z'])
    
    def run(self, job_id, job_num):
        # convert affinity to waterz and region graph
        output_name = self.get_output_name(file_name=f'chunk_{job_id}_{job_num}_s1.h5')
        soma_id0 = self.conf['mask']['soma_id0']
        iou_thres = self.conf['branch']['s1_iou']
        sz_thres = self.conf['branch']['s1_sz'] 
        rg_thres = self.conf['branch']['s1_rg'] 
        if not os.path.exists(output_name):
            waterz_seg_name = self.get_output_name(folder_name='waterz', file_name=f'{job_id}_{job_num}_soma2d.h5')
            mask = read_h5(waterz_seg_name, ['seg'])

            iou = read_h5(self.get_output_name(file_name=f'chunk_{job_id}_{job_num}_iouF.h5'))
            iou = np.vstack(iou)
            # ids that are not related to soma
            iou = iou[(iou[:,:2] > soma_id0).max(axis=1) == 0]
            
            # step 1            
            score = iou[:, 4].astype(float) / (iou[:, 2] + iou[:, 3] - iou[:, 4])
            gid = iou[score >= iou_thres, :2].astype(np.uint32)

            # step 2: singleton (not in the pairs above)
            score2 = iou[:, 2].astype(float) / iou[:, 3]
            gid2 = iou[(score2 >= sz_thres)* (score2 <= 1/sz_thres), :2].astype(np.uint32)
            singleton = np.in1d(gid2.ravel(), gid.ravel(), invert=True).reshape(gid2.shape).max(axis=1)
            gid = np.vstack([gid, gid2[singleton]])
            
            # step 3: rg check
            rg_id, rg_score = read_h5(waterz_seg_name, ['id', 'score'])
            arr2str = lambda x: f'{min(x)}-{max(x)}'            
            score = get_query_count_dict(rg_id, arr2str, rg_score, gid)
            gid = gid[score < rg_thres]
                         
            relabel = merge_id(gid[:, 0], gid[:, 1])
            mask[mask<len(relabel)] = relabel[mask[mask<len(relabel)]]
            
            write_h5(output_name, [mask, relabel, gid], ['seg', 'relabel', 'gid'])

class BranchBorderTask(Task):
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
            
class BranchAllTask(Task):
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
