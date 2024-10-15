import argparse
from em_pipeline.tasks import get_task
from em_util.io import read_yml
from em_util.cluster import slurm

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
        "-s",
        "--cluster-file",
        type=str,
        help="path to the cluster configuration file",
        default='conf/cluster.yml',        
    )
    parser.add_argument(
        "-c",
        "--conf-file",
        type=str,
        help="path to the project configuration file",
        default='conf/j0126.yml',        
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
    parser.add_argument(
        "-nc",        
        "--num-cpu",
        type=int,
        help="number of cpu",
        default=1,
    )
    parser.add_argument(
        "-e",
        "--eval",
        type=str,
        help="evaluation data",
        default="valid",
    )
    
    
    return parser.parse_args()        
    
    
if __name__== "__main__":
    # sa zf
    # python main.py -t waterz-soma2d -nc 8
    # python main.py -t branch-chunk-s1
    # python main.py -t waterz-soma2d -i 0 -n 57
     
    args = get_arguments()
    task = get_task(args.conf_file, args.task)
        
    if args.job_num == 0: # write cluster cmd files
        conf = read_yml(args.cluster_file)
        #cmd = '\n'.join(conf['env'])
        cmd = f'\ncd {conf["folder"]}\n'
        cmd += f'\n{conf["python"]}/python main.py -c {args.conf_file} -t {args.task} -i %d -n %d'
        output_file = task.get_output_name('slurm', task.name)
        job_num = task.get_job_num()
        slurm.write_slurm_all(cmd, output_file, job_num, args.partition, \
            args.num_cpu, conf['num_gpu'], conf['memory'])
    else:
        task.run(args.job_id, args.job_num)
