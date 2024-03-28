# Testing memory efficient methods for mapping data
# This code combines chunking with memory mapping, a method described frequently by neurodata without borders folk
import tifffile as tf
import numpy as np
import os
import xmltodict
import matplotlib.pyplot as plt
import time
import psutil

# TODO: RECHECK AND VALIDATE ALL FUNCTIONS IN CONVERT
# TODO: CHECK 4D, Validate suite2p and max-proj

# Minimal ram usage
filepath = r"C:\Users\spell\Desktop\John\cleanLines1_img\Image_001_001.raw"
self=RawToTif(filepath)
class RawToTif():
    '''
    This code writes your .raw file to .tif using various mechanisms.

    Writing suite2p style provides options to chunk write your data, performing very fast.

    This code is rather expensive on memory if you attempt to write one large file and so we must
    write separate files!
    
    '''

    def __init__(self, filepath: str):
        '''
        Loads and converts .raw file while skipping the flyback frame. Provides options to process data.

            Args:
                >>> filepath: path to .raw file

        '''
        print("Starting at",str(psutil.virtual_memory()[2]),"<%> RAM utility")
        code_start = time.process_time()

        # figure root
        rootpath = os.path.split(filepath)[0]

        # get metadata
        root_contents = os.listdir(rootpath)
        metadata_file = [i for i in root_contents if '.xml' in i][0]
        metadata_path = os.path.join(rootpath,metadata_file)
        file = xmltodict.parse(open(metadata_path,"r").read()) # .xml file

        # define frame rate based on metadata
        fr = float(file['ThorImageExperiment']['LSM']['@frameRate'])

        # get dimensions of recorded data
        x=int(file['ThorImageExperiment']['LSM']['@pixelX'])
        y=int(file['ThorImageExperiment']['LSM']['@pixelY'])
        t=int(file['ThorImageExperiment']['Timelapse']['@timepoints']) # this is how the thorlabs code works
        z=int(file['ThorImageExperiment']['ZStage']['@steps']) # check this variable
        dims=(z,t,y,x)

        # instead of pulling all of that into memory, lets write it immediately, then call the mapped data
        offset=0; vector_list = []; counter = 0; offset_list = []
        try:
            for ti in range(t):
                for zi in range(z):     
                    np.memmap(filepath, dtype='int16', offset=offset, mode='r', shape=(x,y))
                    offset_list.append(offset) 
                    offset+=int(x*y*16/8) # bytes (16bit/8)
                    counter+=1                      
                # skip the flyback frame
                offset+=int(x*y*16/8)
        except:
            print("Aborting loop at:",str(ti),"/",str(t))          

        # store this for later
        self.vector_list = vector_list
        self.dims = (z,int(len(offset_list)/z),y,x)
        self.fr = fr
        self.filepath = filepath
        self.rootpath = rootpath
        self.root_contents = root_contents
        self.metadata = file
        self.idx_offset_np = offset_list # this is really important for indexing from the np.memmap .raw file

        print("rootpath:",self.rootpath)

    def convert(self, method: str = 'suite2p', split_file = True):

        '''
        Mechanism to convert data
        
        Args:
            >>> method: method on how to format your data
                    '4D': Preserves your z-dimension and saves your file as a 4D array (z,t,y,x)
                    'suite2p': preserves your z-dimension but saves your file as a 3D array (t,y,x) as such:
                                frame0 = time0_plane0_channel0
                                frame1 = time0_plane1_channel0
                                frame2 = time0_plane2_channel0
                            Assuming a 3 plane video (code is agnostic to number of planes)
                    'max_proj': maximum projection taken over the z-plane to generate a 3D file (t,y,x)
            
            >>> split_file: method on whether to split read-outs into separate files (recommended)
            >>> chunk_samples: number of samples by which you write data
                    default = 500 samples. You could up this to speed up saving but at the cost of memory.
        '''
        print("This code does not support multi-channel recordings")

        print("Starting at",str(psutil.virtual_memory()[2]),"<%> RAM utility")
        code_start = time.process_time()        

        # get dimensions
        z,t,y,x = self.dims

        # check for faults
        assert z*t == len(self.idx_offset_np), "Mismatched Dimensions: expected dimensions (z and t) do not match the number of collected samples"

        # chunky writing variables
        total_count = t*z; # get total count of timepoints and amount of samples to chunk data by
        count_range = list(range(total_count)) # define the range over which to sample data

        # create a memory mappable file, with vectorized data
        if '4D' in method:
            code_start = time.process_time()   
            print("method: 4D detected. Your file will be saved with dimensions (z,t,y,x):",z,t,y,x)
            print("Please wait while memory mapped file is created...")
            self.fname = fname_new(self.rootpath,'img_mmap_4D.tif')
            #self.fname = os.path.join(self.rootpath,'img_mmap_4D.tif')
            im = tf.memmap(
                self.fname,
                shape=(z,t,y,x),
                dtype=np.uint16,
                imagej=True
                #append=True
            )
            print(time.process_time() - code_start)
            print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")

            # Chunking by time, into a z-plane
            for zi in range(z):
                time_range = list(range(t)); chunker = 500; 

                # array of ca data in plane zi
                np_mem_list = []
                for idxi in self.idx_offset_np[zi::z]:
                    np_mem_list.append(np.memmap(self.filepath, dtype='int16', offset=idxi, mode='r', shape=(x,y)))

                # chunk write
                for timei in time_range[::chunker]:
                    im[zi,timei:timei+chunker,:,:] = np_mem_list[timei:timei+chunker] 
                    im.flush()
                    del im; im=tf.memmap(self.fname)
                    print("Run time for:",str(timei),"/",str(t), time.process_time() - code_start)
                    print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")   

                del np_mem_list

        elif 'suite2p' in method:

            print("method: suite2p detected. Your file will be saved with dimensions (t*z,y,x):",t*z,y,x)
            print("Please wait while memory mapped file is created...")
            self.fname = fname_new(self.rootpath,'img_mmap_suite2p_z.tif')
            #self.fname = os.path.join(self.rootpath,'img_mmap_suite2p_z.tif')
            im = tf.memmap(
                self.fname,
                shape=(t*z,y,x),
                dtype=np.uint16,
                imagej=True,
                #append=True
            )
            print("File mapped to disk")
            print(time.process_time() - code_start)
            print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")            

            # writing data to disk in chunks (500)
            print("Please wait while data are written to disk using a 'chunking' mechanism...")

            # beautiful thing about python is that if the loop exceeds the samples, python will grab the remaining samples, despite you requesting more than what exists!
            chunk_samples = int(500)
            chunk_loop = count_range[0::chunk_samples] # skip every chunk_samples samples
            assert chunk_loop[-1]+chunk_samples > total_count, "You will not write all samples! Looping mechanism exceeds the total count of samples! FIX ME!"
                        
            for framesi in chunk_loop:
                # temporarily load data
                np_mem_list = []
                idx_load = self.idx_offset_np[framesi:framesi+chunk_samples]
                for idxi in idx_load:
                    np_mem_list.append(np.memmap(self.filepath, dtype='int16', offset=idxi, mode='r', shape=(x,y)))

                # chunky write :) - no need to worry about the remainder bc slicing takes care of it!
                im[framesi:framesi+chunk_samples,:,:] = np_mem_list
                im.flush()
                del im; im=tf.memmap(self.fname) # clean up memory
                print("Run time for",str(framesi),"/",str(total_count),":",time.process_time() - code_start, "Memory:",str(psutil.virtual_memory()[2]),"<%> RAM utility")
            #print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")

        # TODO: Something is broken here
        elif 'max_proj' in method:

            print("method: max_proj detected. Your file will be saved with dimensions (t,y,x):",t,y,x)
            print("Please wait while memory mapped file is created...")
            self.fname = fname_new(self.rootpath,'img_mmap_maxproj_z.tif')
            im = tf.memmap(
                self.fname,
                shape=(t,y,x),
                dtype=np.uint16,
                imagej=True
                #append=True
            )
            print(time.process_time() - code_start)
            print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")

            # lets chunk it!
            # beautiful thing about python is that if the loop exceeds the samples, python will grab the remaining samples, despite you requesting more than what exists!
            chunker = z*z*100 # number of samples to save over
            chunk_loop = count_range[0::chunker] # skip every chunk_samples samples
            time_loop = list(range(t))
            assert chunk_loop[-1]+chunker > total_count, "You will not write all samples! Looping mechanism exceeds the total count of samples! FIX ME!"

            counter = 0
            for framesi in range(len(chunk_loop)):

                # get indices of data in .raw file
                np_mem_list = []
                idx_load = self.idx_offset_np[chunk_loop[framesi]:chunk_loop[framesi]+(chunker*z)]
                for idxi in idx_load:
                    np_mem_list.append(np.memmap(self.filepath, dtype='int16', offset=idxi, mode='r', shape=(x,y)))
                
                # separate data by plane
                planes = []
                for zi in range(z):
                    planes.append(np_mem_list[zi::z]) # automatically establish the sampling over planes and data to extract every plane as separate variables                   
                
                # keep track of true samples
                true_samples = len(planes[0])

                # chunky write :) - no need to worry about the remainder bc slicing takes care of it!
                if im.shape[0]-counter < chunker:
                    im[counter:counter+(im.shape[0]-counter),:,:] = np.max(np.array(planes),axis=0)
                else:
                    im[counter:counter+chunker,:,:] = np.max(np.array(planes),axis=0)
                im.flush()
                counter+=true_samples
                del im; im=tf.memmap(self.fname)
                print("Run time for",str((framesi/len(chunk_loop))*100),"% Complete",":",time.process_time() - code_start, "Memory:",str(psutil.virtual_memory()[2]),"<%> RAM utility")

            # fact check max projection
            #fig, ax = plt.subplots(nrows=1,ncols=4)
            #for i in range(3):
            #    ax[i].imshow(self.vector_list[i])
            #ax[-1].imshow(np.max(np.array(self.vector_list[0:0+3]),axis=0))
        print("Completed conversion")

class RawToNWB(RawToTif):
    # use the __init__ from RawToTif. We can write as np or tif to nwb
    pass

def fname_new(rootpath,fname):
    '''
    This code searches for existing fnames and updates the naming convention as to prevent overwrite
    
    Args:
        >>> rootpath: folder that you want your data saved to
        >>> fname: file name to save your data as
    '''
    root_contents = os.listdir(rootpath)
    next = False
    while next is False:
        if fname in root_contents:
            fullpath = os.path.join(rootpath,fname.split('.tif')[0]+'_new.tif')
            next = True
        else:
            fullpath = os.path.join(rootpath,fname)
            next = True

    return fullpath

