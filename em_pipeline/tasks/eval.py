from .task import Task
import numpy as np
from em_util.io import read_vol
from em_erl.eval import compute_segment_lut_tile_z


class ERLTask(Task):
    def __init__(self, conf_file, name='rg'):
        super().__init__(conf_file, name)        
        
    def get_job_num(self):        
        return 1

    def run(self, job_id, job_num):
        # convert affinity to waterz and region graph
        waterz_name = self.get_output_name(folder_name='waterz', file_name=f'%d_{num_chunk}.h5')
        if self.name == 'erl-chunk':
            # naively stack all chunks        
            seg_count = np.loadtxt(self.get_output_name(file_name='stats.txt')).astype(int)
            # compute lut
            pts = read_vol(self.conf['gt'])
            num_chunk = self.get_zchunk_num()
            zran = range(num_chunk)[job_id::job_num]
            compute_segment_lut_tile_z(waterz_name, zran, pts, self.conf['waterz']['num_z'], seg_oset=seg_count[zran])            

