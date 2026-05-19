import h5py                                                                                    
import numpy as np
import pdb
f = h5py.File('/Users/lehmer/python/lightningpy/lightning/data/models/POSYDON/POSYDON_fullgrid_g_single.h5','r')                                       
print('age.shape', f['age'][:].shape)                                                            
print('Zstars.shape', f['Zstars'][:].shape, f['Zstars'][:])                                      
print("'lines/lum' present:", 'lines' in f and 'lum' in f['lines'])                              
if 'lines' in f and 'lum' in f['lines']:                                                         
    arr = np.array(f['lines/lum'][:])                                                            
    print("lines/lum.shape", arr.shape)                                                          
if 'spec' in f and 'noneb' in f['spec']:                                                         
    print('spec/noneb.shape', np.array(f['spec/noneb'][:]).shape)                                
if 'spec' in f and 'neb' in f['spec']:                                                           
    print('spec/neb.shape', np.array(f['spec/neb'][:]).shape)                                    

breakpoint()
f.close()   

