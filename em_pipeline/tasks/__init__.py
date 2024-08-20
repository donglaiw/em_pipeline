from .waterz import *
from .region_graph import *
from .eval import *


def get_task(conf, task):
    task0 = task if '-' not in task else task[:task.find('-')]
    if task == "waterz":
        return WaterzTask(conf, task0)
    elif task == "waterz-stats":
        return WaterzStatsTask(conf, task0)
    elif task == 'rg-border':
       return RegionGraphBorderTask(conf, task0) 
    elif task == 'rg-all':
       return RegionGraphAllTask(conf, task0) 
    elif task0 == 'erl':
       return ERLTask(conf, task) 