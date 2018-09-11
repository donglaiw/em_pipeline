import numpy as np
import cPickle
import json
import itertools
import os,sys
import tqdm
import traceback
import rh_logger
import yaml

from T_util import bfly,writeh5

# module load cuda/9.0-fasrc02 cudnn/7.0_cuda9.0-fasrc01
# get parameter
D0 = './data/'
param = yaml.load(open(D0+'run_100-100-100.yaml'))
#param = yaml.load(open(D0+'run_20-20-20_v1.yaml'))

p_vol = param['data']
tile_sz = p_vol['tile-size']
z0=p_vol['z0'];z1=p_vol['z1']
y0=p_vol['y0'];y1=p_vol['y1']
x0=p_vol['x0'];x1=p_vol['x1']

# input volume: z,y,x
p_aff = param['aff']
in_sz = [p_aff['mz'], p_aff['my'], p_aff['mx']] # model output
vol_sz = [p_aff['vz'], p_aff['vy'], p_aff['vx']] # read more chunk
pad_sz = [p_aff['pz'], p_aff['py'], p_aff['px']] # read more chunk

# classifier
Dp = '/n/coxfs01/donglai/cerebellum/' 
classifier_path = Dp + "pipeline/model_%d_%d_%d.pkl" % tuple(in_sz)
destination = D0+"aff/"
bfly_path = D0+"bfly_v2-2.json"
bfly_db = json.load(open(bfly_path))

def get_dest_path(x, y, z):
    return os.path.join(destination, str(x), str(y), str(z))

def process_message(classifier, x0, x1, y0, y1, z0, z1):
    p1 = get_dest_path(x0, y0, z0)
    redo = False
    for i,k in enumerate('zyx'):
        path = os.path.join(p1, k+'_min.h5')
        if not os.path.exists(path):
            redo = True
            break
    if redo:
        rh_logger.logger.report_event("process: %d,%d,%d" % (x0,y0,z0))
        out = 255*np.ones([3,z1-z0,y1-y0,x1-x0],dtype=np.uint8)
        data = bfly(bfly_db, x0-pad_sz[2], x1+pad_sz[2], y0-pad_sz[1], y1+pad_sz[1], z0-pad_sz[0], z1+pad_sz[0], tile_sz)
        for xflip, yflip, zflip, transpose in itertools.product(
                        (False, ), (False, True), (False, True), (False, )):
            extension = ""
            if transpose:
                extension += "t"
            if zflip:
                extension += "z"
            if yflip:
                extension += "y"
            if xflip:
                extension += "x"
            volume = data.copy() 
            if xflip:
                volume = volume[:, :, ::-1]
            if yflip:
                volume = volume[:, ::-1]
            if zflip:
                volume = volume[::-1]
            if transpose:
                volume = volume.transpose(0, 2, 1)
            # aff: 3*z*y*x 
            vout = classifier.classify(volume, 0, 0, 0)

            if transpose: # swap x-/y-affinity
                vout = vout.transpose(0, 1, 3, 2)
                vout[[1,2]] = vout[[2,1]]
            if zflip:
                vout = vout[:,::-1]
            if yflip:
                vout = vout[:, :, ::-1]
            if xflip:
                vout = vout[:, :, :, :, ::-1]
            out = np.minimum(out,vout)
        for i,k in enumerate('zyx'):
            path = os.path.join(p1, k+'_min.h5')
            writeh5(path, 'main', out[i])

def main_db(jobId, jobNum):
    rh_logger.logger.start_process("Worker %d" % jobId, "starting", [])
    theano_flags="device=cuda0,dnn.enabled=True,base_compiledir=/n/home04/donglai/.theano/compile_%d/"%(jobId)
    os.environ["THEANO_FLAGS"] = theano_flags
    import keras
    classifier = cPickle.load(open(classifier_path))
    classifier.model_path = Dp+classifier.model_path
    count = 0
    for x0a in range(x0, x1, vol_sz[2]):
        x1a = x0a + vol_sz[2]
        for y0a in range(y0, y1, vol_sz[1]):
            y1a = y0a + vol_sz[1]
            for z0a in range(z0, z1, vol_sz[0]):
                z1a = z0a + vol_sz[0]
                dir_path = get_dest_path(x0a, y0a, z0a)
                if count % jobNum == jobId:
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    process_message(classifier, x0a, x1a, y0a, y1a, z0a, z1a)
                count += 1

def check_done():
    count = 0
    for x0a in range(x0, x1, vol_sz[2]):
        for y0a in range(y0, y1, vol_sz[1]):
            for z0a in range(z0, z1, vol_sz[0]):
                p1 = get_dest_path(x0a, y0a, z0a)
                count+=1
                for i,k in enumerate('x'):
                    path = os.path.join(p1, k+'_min.h5')
                    if not os.path.exists(path):
                        print "undone: %d,%d,%d" % (x0a, y0a, z0a)
    print 'total jobs: ',count
if __name__== "__main__":
    # python classify4-jwr_20um.py 0 0,1,2,3,4,5,6,7,9
    jobId = int(sys.argv[1])
    jobNum = int(sys.argv[2])
    main_db(jobId, jobNum) # single thread 
    #check_done() # single thread 
