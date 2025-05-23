# Testing memory efficient methods for mapping data
# This code combines chunking with memory mapping, a method described frequently by neurodata without borders folk
import tifffile as tf
import numpy as np
import os
import xmltodict
import matplotlib.pyplot as plt
import time
import psutil

# ram usage
filepath = r"C:\Users\spell\Desktop\John\cleanLines1_img\Image_001_001.raw"
self = RawToTif(filepath=filepath)
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

        # identify offset, the number of timepoints
        print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")
        offset=0; vector_list = []
        try:
            for ti in range(t):
                for zi in range(z):       
                    vector_list.append(np.memmap(filepath, dtype='int16', offset=offset, mode='r', shape=(x,y)))
                    offset+=int(x*y*16/8) # bytes (16bit/8) 
                # skip the flyback frame
                offset+=int(x*y*16/8)
        except:
            print("Aborting loop at:",str(ti),"/",str(t))
        print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")
        print("Run time:",time.process_time() - code_start,"s")

        # memory map to prep space
        #self.fname = os.path.join(rootpath,'img_mmap.tif')
        print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")

        # store this for later
        self.vector_list = vector_list
        self.dims = (z,int(len(self.vector_list)/z),y,x)
        self.fr = fr
        self.filepath = filepath
        self.rootpath = rootpath
        self.root_contents = root_contents
        self.metadata = file

        print("rootpath:",self.rootpath)

    def convert(self, method: str = 'suite2p', split_file = True, chunk_write = True):

        '''
        Mechanism to convert data
        
        Args:
            >>> method:
                    '4D': Preserves your z-dimension and saves your file as a 4D array (z,t,y,x)
                    'suite2p': preserves your z-dimension but saves your file as a 3D array (t,y,x) as such:
                                frame0 = time0_plane0_channel0
                                frame1 = time0_plane1_channel0
                                frame2 = time0_plane2_channel0
                            Assuming a 3 plane video (code is agnostic to number of planes)
                    'max_proj': maximum projection taken over the z-plane to generate a 3D file (t,y,x)
        '''
        print("This code does not support multi-channel recordings")

        print("Starting at",str(psutil.virtual_memory()[2]),"<%> RAM utility")
        code_start = time.process_time()        

        # get dimensions
        z,t,y,x = self.dims

        # check for faults
        assert z*t == len(self.vector_list), "Mismatched Dimensions: expected dimensions (z and t) do not match the number of collected samples"

        # create a memory mappable file, with vectorized data
        if '4D' in method:
            print("method: 4D detected. Your file will be saved with dimensions (z,t,y,x):",z,t,y,x)
            print("Please wait while memory mapped file is created...")
            self.fname = os.path.join(self.rootpath,'img_mmap_4D.tif')
            im = tf.memmap(
                self.fname,
                shape=(z,t,y,x),
                dtype=np.uint16,
                append=True
            )
            print(time.process_time() - code_start)
            print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")

            # loop over data 
            print("Please wait while data are written to disk...")
            frame_counter = 0
            for timei in range(t):
                for zi in range(z):
                    im[zi,timei,:,:] = self.vector_list[frame_counter]
                    im.flush()
                    frame_counter+=1
                print("Run time for:",str(timei),"/",str(t), time.process_time() - code_start)
            print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")
        
        elif 'suite2p' in method:
            print("method: suite2p detected. Your file will be saved with dimensions (t*z,y,x):",t*z,y,x)
            print("Please wait while memory mapped file is created...")
            self.fname = os.path.join(self.rootpath,'img_mmap_suite2p_z.tif')
            im = tf.memmap(
                self.fname,
                shape=(t*z,y,x),
                dtype=np.uint16,
                append=True
            )
            print("File mapped to disk")
            print(time.process_time() - code_start)
            print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")            

            #chunk_write = True
            if chunk_write:

                # writing data to disk in chunks (500)
                print("Please wait while data are written to disk using a 'chunking' mechanism...")
                total_count = t*z; chunk_samples = 500; # get total count of timepoints and amount of samples to chunk data by
                count_range = list(range(total_count)) # define the range over which to sample data
                chunk_loop = count_range[0::chunk_samples] # skip every chunk_samples samples
                assert chunk_loop[-1]+chunk_samples > total_count, "You will not write all samples! Looping mechanism exceeds the total count of samples! FIX ME!"
                
                # beautiful thing about python is that if the loop exceeds the samples, python will grab the remaining samples, despite you requesting more than what exists!
                for framesi in chunk_loop:
                    print(framesi, framesi+chunk_samples)
                    # chunky write :) - no need to worry about the remainder bc slicing takes care of it!
                    im[framesi:framesi+chunk_samples,:,:] = self.vector_list[framesi:framesi+chunk_samples]
                    im.flush()
                    print("Run time for",str(framei),"/",str(total_count),":",time.process_time() - code_start)
                print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")
            
            else:        
                # writing data to disk iteratively
                print("Please wait while data are written to disk...")
                frame_counter = 0; total_count = t*z
                for framei in range(total_count):
                    im[framei,:,:] = self.vector_list[frame_counter]
                    im.flush()
                    frame_counter+=1
                    print("Run time for",str(framei),"/",str(total_count),":",time.process_time() - code_start)
                print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")

        elif 'max_proj' in method:

            print("method: max_proj detected. Your file will be saved with dimensions (t,y,x):",t,y,x)
            print("Please wait while memory mapped file is created...")
            self.fname = os.path.join(self.rootpath,'img_mmap_maxproj_z.tif')
            im = tf.memmap(
                self.fname,
                shape=(t,y,x),
                dtype=np.uint16,
                append=True
            )
            print(time.process_time() - code_start)
            print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")
                
            # max projection
            if chunk_write:
                # writing data to disk in chunks (500)
                print("Please wait while data are written to disk using a 'chunking' mechanism...")
                total_count = t*z; chunk_samples = 500; # get total count of timepoints and amount of samples to chunk data by
                count_range = list(range(total_count)) # define the range over which to sample data
                chunk_loop = count_range[0::chunk_samples] # skip every chunk_samples samples
                assert chunk_loop[-1]+chunk_samples > total_count, "You will not write all samples! Looping mechanism exceeds the total count of samples! FIX ME!"
                
                # beautiful thing about python is that if the loop exceeds the samples, python will grab the remaining samples, despite you requesting more than what exists!
                for framesi in chunk_loop:
                    print(framesi, framesi+chunk_samples)
                    # chunky write :) - no need to worry about the remainder bc slicing takes care of it!
                    im[framesi:framesi+chunk_samples,:,:] = self.vector_list[framesi:framesi+chunk_samples]
                    im.flush()
                    print("Run time for",str(framei),"/",str(total_count),":",time.process_time() - code_start)
                print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")

            else:
                total_count = t*z; count_range = list(range(total_count))
                frame_counter = 0; looper = count_range[0::chunk_samples]
                for framei in range((t*z)-2):
                    print(framei, frame_counter, frame_counter+3) 
                    frame_counter+=1               
                    # so python will keep running until it cant. If you say n:n+3 and there is only N+2, itll just get n+2
                    im[framei,:,:] = np.max(np.array(self.vector_list[frame_counter:frame_counter+3]),axis=0)
                    im.flush()
                    frame_counter+=1
                    print("Run time for",str(framei),"/",str(t*z-2),":",time.process_time() - code_start)
            # print("Run time:", time.process_time() - code_start)
                print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")

            # fact check max projection
            #fig, ax = plt.subplots(nrows=1,ncols=4)
            #for i in range(3):
            #    ax[i].imshow(self.vector_list[i])
            #ax[-1].imshow(np.max(np.array(self.vector_list[0:0+3]),axis=0))

    def genNWBfile():
        pass





# ITERARIVE
# writing data to disk in chunks (500)
print("Please wait while data are written to disk using a 'chunking' mechanism...")
chunk_loop = count_range[0::z] # skip every chunk_samples samples
assert chunk_loop[-1]+z == total_count, "You will not write all samples! Looping mechanism exceeds the total count of samples! FIX ME!"

# this works to write iteratively
frame_counter = 0
for framesi in chunk_loop:
    #print(frame_counter)
    #frame_counter+=1
    print("Start:stop",framesi, framesi+z, ", Counter:",frame_counter)
    # chunky write :) - no need to worry about the remainder bc slicing takes care of it!
    im[frame_counter,:,:] = np.max(np.array(self.vector_list[framesi:framesi+z]),axis=0)
    im.flush()
    frame_counter+=1
    print("Run time for",str(framesi),"/",str(total_count),":",time.process_time() - code_start, "Memory:",str(psutil.virtual_memory()[2]),"<%> RAM utility")
