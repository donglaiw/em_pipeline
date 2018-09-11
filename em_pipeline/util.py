import numpy as np
import h5py

def writeh5(filename, datasetname, dtarray):
    fid=h5py.File(filename,'w')
    ds = fid.create_dataset(datasetname, dtarray.shape, compression="gzip", dtype=dtarray.dtype)
    ds[:] = dtarray
    fid.close()

def write_bfly(sz, numT, imN, zPad=0, im_id=None, outName=None):                                     
    # one tile for each section                                                                      
    dim={'depth':sz[0]+2*zPad, 'height':sz[1], 'width':sz[2],                                        
         'dtype':'uint8', 'n_columns':numT[1], 'n_rows':numT[0]}                                     
    # 1-index                                                                                        
    if im_id is None:                                                                                
        im_id = range(zPad+1,1,-1)+range(1,sz[0]+1)+range(sz[0]-1,sz[0]-zPad-1,-1)                   
    sec=[imN % x for x in im_id]                                                                     
    out={'sections':sec, 'dimensions':dim}                                                           
    if outName is None:                                                                              
        return out                                                                                   
    else:                                                                                            
        import json                                                                                  
        with open(outName,'w') as fid:                                                               
            json.dump(out, fid)
