import numpy as np
import yaml
import h5py
import os,sys

from zwatershed import zwatershed
from waterz import waterz
from em_segLib.seg_util import relabel
from em_segLib.io_util import writeh5

# get parameter
D0 = '/n/coxfs01/donglai/jwr/20um_20180720/'

param = yaml.load(open(D0+'run_100-100-100.yaml'))

p_vol = param['data']
z0=p_vol['z0'];z1=p_vol['z1']
y0=p_vol['y0'];y1=p_vol['y1']
x0=p_vol['x0'];x1=p_vol['x1']

p_aff = param['aff']
vol_sz = [p_aff['vz'], p_aff['vy'], p_aff['vx']] # read more chunk

p_zw = param['zwatershed']
zw_low = p_zw['find-basins']['a-low']
zw_high = p_zw['find-basins']['a-high']
zw_rel = p_zw['find-basins']['a-rel']==1
zw_dust = p_zw['aggregate']['s-dust']
zw_thres = p_zw['aggregate']['s-thres']
zw_dust_merge = p_zw['aggregate']['s-dust-merge']
zw_mst_merge = p_zw['aggregate']['s-mst-merge']
zw_chunk = p_zw['chunkZ']

p_wz = param['waterz']
wz_mf = p_wz['mf']
wz_thres = [float(x) for x in p_wz['thres'].split(',')]
wz_chunk = p_wz['chunkZ']

Do=param['files']['result-path']
paths=[Do+'waterz/',Do+'zwatershed/']
for p in paths:
    if not os.path.exists(p):
        os.mkdir(p)
for thres in wz_thres:
    sn=Do+'waterz/%s_%.2f'%(wz_mf,thres)
    if not os.path.exists(sn):
        os.mkdir(sn)

def get_aff(x, y, z):
    return os.path.join(Do,'aff', str(x), str(y), str(z))

def get_zwatershed(x,y,z):
    return os.path.join(Do, 'zwatershed', str(x), str(y), 'zwatershed_%04d.h5'%(z))

def get_waterz(x, y, z, thres):
    return os.path.join(D0, 'waterz', '%s_%.2f'%(wz_mf, thres), str(x), str(y), 'waterz_%04d.h5'%(z))

def process_waterz(z0a, z1a, y0a, y1a, x0a, x1a):
    out_path = get_waterz(x0a, y0a, z0a, wz_thres[-1])
    if not os.path.exists(out_path):
        # naive 
        zstep = min(z1a-z0a, vol_sz[0])
        if zstep == vol_sz[0]: # fit at least a chunk
            zz = range(z0a, z1a, vol_sz[0])
        else:
            zz = z0+vol_sz[0]*((z0a-z0)//vol_sz[0])
        numZ = z1a-z0a
        xx = range(x0a, x1a, vol_sz[1])
        yy = range(y0a, y1a, vol_sz[2])
        print xx,yy,zz
        aff = np.zeros((3, numZ, vol_sz[1]*len(yy), vol_sz[2]*len(xx)), dtype=np.float32) 
        
        print 'load aff'
        for xi,x in enumerate(xx):
            for yi,y in enumerate(yy):
                for kid,k in enumerate('zyx'):
                    if zstep == vol_sz[0]: # fit at least a chunk
                        for zi,z in enumerate(zz):
                            aff[kid, zi*vol_sz[1]:(zi+1)*vol_sz[0], yi*vol_sz[1]:(yi+1)*vol_sz[1], xi*vol_sz[2]:(xi+1)*vol_sz[2]] = \
                                    np.array(h5py.File(get_aff(x, y, z)+"/%s_min.h5" % (k))['main']).astype(np.float32)/255.0
                    else: # fit part of a chunk
                        aff[kid, :, yi*vol_sz[1]:(yi+1)*vol_sz[1], xi*vol_sz[2]:(xi+1)*vol_sz[2]] = \
                                np.array(h5py.File(get_aff(x, y, zz)+"/%s_min.h5" % (k))['main'])[z0a-zz:z1a-zz].astype(np.float32)/255.0

        print 'load 2d seg'
        if zstep == vol_sz[0]: # fit at least a chunk
            z2d = np.zeros((numZ, vol_sz[1]*len(yy), vol_sz[2]*len(xx)), dtype=np.uint64) 
            for zid in range(len(zz)):
                sn = get_zwatershed(x0a, y0a, z0a+zid*vol_sz[0])
                z2d[zid*vol_sz[0]:(zid+1)*vol_sz[0]] = np.array(h5py.File(sn)['main']).astype(np.uint64)
        else:
            sn = get_zwatershed(x0a, y0a, zz)
            z2d = np.array(h5py.File(sn)['main'])[z0a-zz:z1a-zz].astype(np.uint64)

        # unique and relabel ids
        next_id=np.uint64(0);
        # change value by reference
        for zid in range(numZ):
            tile = z2d[zid]
            tile[np.where(tile>0)] += next_id
            next_id = tile.max()
        
        print 'do segmentation'
        seg = waterz(affs=aff, thresholds=wz_thres, fragments=z2d, merge_function=wz_mf)
        for tid,tt in enumerate(wz_thres):
            out_path = get_waterz(x0a, y0a, z0a, tt)
            writeh5(out_path, 'main', relabel(seg[tid]).astype(np.uint32))

def process_zwatershed(z0a, z1a, y0a, y1a, x0a, x1a):
    # zwatershed: need to relabel final seg for storage
    out_path = get_zwatershed(x0a,y0a,z0a)
    if not os.path.exists(out_path):
        xx = range(x0a,x1a,vol_sz[1])
        yy = range(y0a,y1a,vol_sz[2])
        print 'load tiles:',z0a, z1a, y0a, y1a, x0a, x1a
        aff = np.zeros((3, vol_sz[0], vol_sz[1]*len(yy), vol_sz[2]*len(xx)), dtype=np.float32) 
        for xi,x in enumerate(xx):
            for yi,y in enumerate(yy):
                for kid,k in enumerate('yx'):
                    aff[1+kid,:, yi*vol_sz[1]:(yi+1)*vol_sz[1],xi*vol_sz[2]:(xi+1)*vol_sz[2]] = \
                            np.array(h5py.File(get_aff(x, y, z0a)+"/%s_min.h5" % (k))['main']).astype(np.float32)/255.0
        
        print 'do segmentation'
        seg = np.zeros((vol_sz[0], vol_sz[1]*len(yy), vol_sz[2]*len(xx)), dtype=np.uint16) 
        for zid in range(vol_sz[0]):
            seg[zid] = relabel(zwatershed(aff[:,zid:zid+1], T_threshes=[zw_thres], T_dust=zw_dust, T_aff=[zw_low,zw_high,zw_dust_merge], T_aff_relative=zw_rel, T_merge=zw_mst_merge)[0][0][0]).astype(np.uint16)
        writeh5(out_path, 'main', seg)

def main_db(opt, jobId, jobNum):
    n_work_items = 0
    # 100um x 100um
    #xx=list(44+np.array([0,5,10,15,21,27])*464)
    #yy=list(44+np.array([0,5,10,15,21,27])*464)
    xx=list(44+np.array([0,9,18,27])*464)
    yy=list(44+np.array([0,9,18,27])*464)
    #xx=list(44+np.array([2,10])*464)
    #yy=list(44+np.array([9,15])*464)

    zchunk = [zw_chunk,wz_chunk][opt]
    for xi in range(len(xx)-1):
        x0a = xx[xi];x1a = xx[xi+1]
        for yi in range(len(yy)-1):
            y0a = yy[yi];y1a = yy[yi+1]
            if opt == 0: 
                out_folder = get_zwatershed(x0a, y0a, z0)
            elif opt == 1: 
                out_folder = get_waterz(x0a, y0a, z0, wz_thres[-1])
            out_folder = out_folder[:out_folder.rfind('/')]
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)

            for z0a in range(z1-zchunk, z0-zchunk, -zchunk):
            #for z0a in range(z0, z1, zchunk):
                z1a = z0a + zchunk
                if n_work_items % jobNum == jobId:
                    if opt == 0: 
                        process_zwatershed(z0a, z1a, y0a, y1a, x0a, x1a)
                    elif opt == 1: 
                        process_waterz(z0a, z1a, y0a, y1a, x0a, x1a)
                n_work_items += 1
    
if __name__== "__main__":
    # python seg-jwr_20um.py 0 0 1
    opt = int(sys.argv[1])
    jobId = int(sys.argv[2])
    jobNum = int(sys.argv[3])
    if jobNum>0:
        main_db(opt, jobId, jobNum) 
    else: # check results
        for i in range(z0,z1,vol_sz[0]):
            if not os.path.exists(get_zwatershed(i)):
                print i
