import os
import yaml
from em_util.io import mkdir

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