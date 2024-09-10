from .waterz import *
from .region_graph import *
from .eval import *


def get_task(conf, task):
    task0 = task if '-' not in task else task[:task.find('-')]
    if task == "waterz":
        return WaterzTask(conf, task0)
    elif task == "waterz-soma2d":
        return WaterzSoma2DTask(conf, task0)
    elif task == "waterz-soma2d-iou":
        return WaterzSoma2DIOUTask(conf, task0)
    elif task == 'rg-chunk':
       return RegionGraphChunkTask(conf, task0) 
    elif task == 'rg-border':
       return RegionGraphBorderTask(conf, task0) 
    elif task == 'rg-all':
       return RegionGraphAllTask(conf, task0) 
    elif task0 == 'erl':
       return ERLTask(conf, task) 
