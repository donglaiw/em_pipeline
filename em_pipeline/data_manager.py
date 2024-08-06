import os
import yaml




class DataManager(object):
    def __init__(self, config_file=None):
        if config_file is not None:
            self.load_config(config_file)   
    
    def load_config(self, config_file):
        with open(config_file, encoding='utf-8') as file:
            self.param = yaml.load(file)
    
    def get_file_path(self, pref, x, y, z):
        return os.path.join(self.param.folder, pref, str(x), str(y), str(z))