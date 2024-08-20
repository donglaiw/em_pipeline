from .task import Task


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
