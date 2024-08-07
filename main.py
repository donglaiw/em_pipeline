import os
import argparse
import yaml
from em_pipeline.task import *
from em_util.cluster import slurm
from em_util.io import mkdir

def get_arguments():
    """
    Get command line arguments for converting ground truth segmentation to a graph of skeleton.

    Returns:
        argparse.Namespace: Parsed command line arguments including seg_path, seg_resolution, output_path, and num_thread.
    """
    parser = argparse.ArgumentParser(
        description="EM pipeline for 3D neuron segmentation"
    )
    parser.add_argument(
        "-c",
        "--conf-file",
        type=str,
        help="path to the configuration file",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        help="task name",
        required=True,
    )
    parser.add_argument(
        "-i",        
        "--job-id",
        type=int,
        help="job id",
        default=0,
    )
    parser.add_argument(
        "-n",        
        "--job-num",
        type=int,
        help="job num",
        default=0,
    )
    parser.add_argument(
        "-p",        
        "--partition",
        type=str,
        help="cluster partition name",
        default="shared",
    )
    
    return parser.parse_args()        
    
    
if __name__== "__main__":
    # python main.py -c em_pipeline/data/j0126.yaml -t waterz -i 0 -n 10
     
    args = get_arguments()
    task = get_class(args.task, args.conf_file)
        
    if args.job_num == 0: # write cluster cmd files
        cmd = 'python /data/projects/weilab/weidf/lib/em_pipeline/main.py -c {args.conf_file} -t {args.task} -i %d - n %d'      
        output_file = task.get_output_name('slurm', '%d_%d.sh')
        num_cpu, num_gpu = 1, 0
        memory = 100000
        job_num = task.get_job_num()
        slurm.write_slurm_all(cmd, output_file, job_num, args.partition, num_cpu, num_gpu, memory)
    else:
        task.run(args.job_id, args.job_num)