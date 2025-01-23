# This module are functions that interface with thorlabs outputs and includes a RAM efficient version of converting data
#
# RAM friendly mechanisms:
#   -> memory mapping
#   -> chunk saving, then cleaning memory
#   -> Multi-file save (splits file into separate files) - NOT SUPPORTED
#
# Essentially, we map a file to disk via memory mapping, then write chunks of data to memory.
# The chunks of data take up memory and so we continuously delete what we previously used to conserve memory.
#
# John Stout

# packages
import numpy as np
import os
import xmltodict
import matplotlib.pyplot as plt
import time
import psutil
import tifffile as tf
import h5py
import scipy.stats as stat
import scipy.io as sio
#import dask.array as da
#from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import multiprocessing
import rawpy

# load imgfuns
try:
    import imgfuns
except:
    os.chdir(os.path.join(os.getcwd(),'code'))
    import imgfuns
# TODO: RECHECK AND VALIDATE ALL FUNCTIONS IN CONVERT
# TODO: CHECK 4D, Validate suite2p and max-proj
# TODO: Validate with single plane data
# TODO: Implement NWB as an option?

# Minimal ram usage
# the matlab version has a thresholding mechanism to detect potential LED artifacts
class RawToTif():
    '''
    This code writes your .raw file to .tif using various mechanisms.

    Writing suite2p style provides options to chunk write your data, performing very fast.

    This code is rather expensive on memory if you attempt to write one large file and so we must
    write separate files!

    10/17/2024: @JS discovered major issue with using np.memmap to index out the image files. Was 
                    generating incorrect max projection images as compared to both matlab and Fiji (watching the whole video)
                    - The updated method slices out the image planes
                    - - - This is really for a second reason: some LED artifacts are present in only one plane. This needs to be handled plane by plane.
    
    
    12/14/2024: @JS changed & bit operator to "and" logical operator.
    12/14/2024: Addition/fixing of suite2p method
    12/14/2024: Apparently, on our machine, parallel processing may actually increw time for numpy conversion. Now file is converted in __init__
    '''

    def __init__(self, filepath: str):
        '''
        Loads and converts .raw file while skipping the flyback frame. Provides options to process data.

            Args:
                >>> filepath: path to .raw file

        '''
        print("Please use method .convert(method='max_proj') rather than 'suite2p' and '4D'")
        print("Starting at",str(psutil.virtual_memory()[2]),"<%> RAM utility")
        code_start = time.process_time()

        # search for .raw file
        if '.raw' in os.path.split(filepath)[-1]:
            rootpath = os.path.split(filepath)[0]
        else:
            # define root path
            rootpath = filepath

            # discover your .raw imaging file
            filepath = [i for i in os.listdir(rootpath) if '.raw' in i and 'Image' in i]
            assert len(filepath)==1, "The code does not currently support multiple saved .raw files"
            print("Discovered", filepath[0])
            filepath = os.path.join(rootpath,filepath[0]) # save the result

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

        # data
        # Read the .raw file
        # Initialize an empty list to hold the chunks
        chunks = []
        chunk_shape = (512, 512)

        # Define the chunk size in bytes
        chunk_size = np.prod(chunk_shape) * np.dtype('int16').itemsize

        # Read the file in chunks
        print("Reading image data...")
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                chunks.append(np.frombuffer(chunk, dtype='int16').reshape(chunk_shape))

        # now using the chunks variable, split the data according to recording dimensions
        total_frames = int(len(chunks)/4) # 3 planes and a flyback
        #plane0 = chunks[::4]
        #plane1 = chunks[1::4]
        #plane2 = chunks[2::4]
        #flyback = chunks[3::4]

        # Remove every 4th element starting at element 4
        planes = [elem for i, elem in enumerate(chunks) if (i + 1) % 4 != 0]
        assert int(len(planes)/3) == total_frames, "Something is wrong with your removal of flyback frames"


        '''
        # instead of pulling all of that into memory, lets write it immediately, then call the mapped data
        offset=0; vector_list = []; counter = 0; offset_list = []
        try:
            for ti in range(t):
                for zi in range(z):     
                    vector_list.append(np.memmap(filepath, dtype='int16', offset=offset, mode='r', shape=(x,y)))
                    offset_list.append(offset) 
                    offset+=int(x*y*16/8) # bytes (16bit/8)
                    counter+=1                      
                # skip the flyback frame
                offset+=int(x*y*16/8)
        except:
            print("Aborting loop at:",str(ti),"/",str(t))          
        '''

        # store this for later
        self.planes = planes
        self.dims = (z,total_frames,y,x)
        self.fr = fr
        self.filepath = filepath
        self.rootpath = rootpath
        self.root_contents = root_contents
        self.metadata = file
        #self.idx_offset_np = offset_list # this is really important for indexing from the np.memmap .raw file

        print("rootpath:",self.rootpath)

    def convert(self, method: str = 'max_proj', chunker: int = 1000, led_artifacts: str = 'y', memmap_write: bool = False, wipe_and_replace: bool = False, run_parallel = True):

        '''
        Method to convert data
        
        Args:
            >>> method: method on how to format your data
                    '4D': Preserves your z-dimension and saves your file as a 4D array (z,t,y,x)
                    'suite2p': preserves your z-dimension but saves your file as a 3D array (t,y,x) as such:
                                frame0 = time0_plane0_channel0
                                frame1 = time0_plane1_channel0
                                frame2 = time0_plane2_channel0
                            Assuming a 3 plane video (code is agnostic to number of planes)
                    'max_proj': maximum projection taken over the z-plane to generate a 3D file (t,y,x)
            
            >>> chunker: how many images to save at once
            >>> led_artifacts: preset to 'n' but if set to 'y' performs interpolation of led artifact contaminated images
                                Method of interpolation:
                                        - First, the time resolved average is taken of each plane, the result is then z-scored and the
                                        absolute value is used to identify if avg pixel events exceed 7std. 7std was chosen after viewing datasets.
                                        - Thresholded events are then interpolated using linear interpolation of contaminated frame based on the immediately surrounding images

            >>> memmap_write: False. This can be removed. The imwrite method is better and the result is still memory mappable.

            IMPORTANT** led_artifacts is only functional for max_proj

            This code uses parallel processing to handle the large imaging dataset

        Written by John Stout

        # --- EDITS --- #
        # 10/9/2024: updated the max_proj method and defaulted the convert method to max_proj
                        - Included an option for artifact conversion
                        - Included an option for the user to control the scale of saving with "chunker"
        # 10/15/2024: Updated mechanism to perform computations in parallel using copilot
        # 12/18/2024: Finished updating the run_parallel mechanism
        # 1/10/2025: Fixed issue with shape. Must have edited the code in dec to handle numpy rather than list and didnt fix .shape attribute

        '''
        print("This code does not support multi-channel recordings")

        print("Starting at",str(psutil.virtual_memory()[2]),"<%> RAM utility")
        code_start = time.process_time()        

        # get dimensions
        z,t,y,x = self.dims

        # chunky writing variables
        total_count = t*z; # get total count of timepoints and amount of samples to chunk data by
        count_range = list(range(total_count)) # define the range over which to sample data

        # temporary solution to prevent use of other methods before making sure they follow the updated procedures set by 'max_proj' and the __init__
        assert method != '4D', "method=='4D' has not been validated. Please set method='max_proj' or method='suite2p' "
        
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
                time_range = list(range(t)); #chunker = 500; 

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

        # this is the suite2p method for 4D data
        elif 'suite2p' in method:

            # convert to numpy then save
            print("Converting array to numpy...")
            print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")  
            process_start = time.process_time()   
            self.planes = np.array(self.planes)
            print("Time to convert to numpy:",time.process_time() - process_start)
            print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")  

            # TODO: replace this with boolean
            if led_artifacts.lower() == 'y':
                print("Running image interpolation for LED artifacts...")
                ledArtifacts = dict(); meanData = dict(); meanXYzData = dict()
                for zi in range(z):
                    print("Working to correct artifacts in plane",zi)

                    # identify candidate artifact events
                    mean_pixels  = np.mean(np.mean(self.planes[zi::3],axis=1),axis=1) # get a pixel average over time
                    meanXYz      = np.abs(stat.zscore(mean_pixels,axis=0)) # zscore the averaged pixels
                    ledArtifact  = np.asarray(np.where(meanXYz > 7)).flatten()
                    ledArtifacts['Axis'+str(zi)] =  ledArtifact
                    meanData['Axis'+str(zi)]     =  mean_pixels
                    meanXYzData['Axis'+str(zi)]  =  meanXYz

                    # check by introducing artifacts and then interpolating them
                    # self.planes[9999] = np.full((512, 512), 1000)
                    # self.planes[10000] = np.full((512, 512), 1000)
                    # self.planes[10001] = np.full((512, 512), 1000)
                    # after you run the code below, plot
                    # plt.plot(meanXYz)

                    # interpolate missing data
                    for imgi in ledArtifact:
                        if imgi > 1 and imgi < len(meanXYz):
                            print("Interpolating artifact at index:",imgi)

                            # get data surrounding artifact
                            img_temp = np.moveaxis(self.planes[zi::3][imgi-1:imgi+2], 0, -1)
                            img_interp = imgfuns.interp_img(img=img_temp)

                            # reshape result
                            img_interp = np.moveaxis(img_interp,-1,0)

                            # replace data
                            self.planes[zi+imgi*3] = img_interp[1]

                # save array
                print("Saving ledArtifact data...")
                ledMat = {"ledArtifact": ledArtifacts,
                        "meanXY": meanData,
                        "meanXYz": meanXYzData,
                        "info": "ledArtifact is an index of artifacts. meanXY is the pixel average. meanXYz is |zscore(meanXY)|."}
                artFile = os.path.join(self.rootpath,'ledArtifactDataInterp.mat')
                sio.savemat(artFile, ledMat)

            # quicker write
            self.fname = os.path.join(self.rootpath, 'imgPlaneZ.tif')

            # convert to numpy then save
            print(f'Writing imgPlaneZ.tif to: {self.fname}')
            tf.imwrite(self.fname, self.planes, dtype=self.planes.dtype, bigtiff=True)

            '''
            print("method: suite2p detected. Your file will be saved with dimensions (t*z,y,x):",t*z,y,x)
            print("Please wait while memory mapped file is created...")
            self.fname = fname_new(self.rootpath,'imgPlaneZ.tif')
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
            chunk_samples = int(chunker)
            chunk_loop = count_range[0::chunk_samples] # skip every chunk_samples samples
            assert chunk_loop[-1]+chunk_samples > total_count, "You will not write all samples! Looping mechanism exceeds the total count of samples! FIX ME!"
                        
            for framesi in chunk_loop:
                # temporarily load data
                np_mem_list = []
                idx_load = self.idx_offset_np[framesi:framesi+chunk_samples]
                for idxi in idx_load:
                    np_mem_list.append(np.memmap(self.filepath, dtype='int16', offset=idxi, mode='r', shape=(x,y)))
                
                # inversion
                #np_mem_list = [np.rot90(np.fliplr(i)) for i in np_mem_list]

                # chunky write :) - no need to worry about the remainder bc slicing takes care of it!
                im[framesi:framesi+chunk_samples,:,:] = np_mem_list
                im.flush()
                del im; im=tf.memmap(self.fname) # clean up memory
                print("Run time for",str(framesi),"/",str(total_count),":",time.process_time() - code_start, "Memory:",str(psutil.virtual_memory()[2]),"<%> RAM utility")
            #print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")
            '''
        
        # here is the max projection method that the lab prefers
        elif 'max_proj' in method:

            print("method: max_proj detected. Your file will be saved with dimensions (t,y,x):",t,y,x)
            print("Please wait while memory mappable file is created...")
            
            # might save time by replacing this with a more efficient approach
            if wipe_and_replace == True:
                imgFound = len([i for i in os.listdir(self.rootpath) if 'img.tif' in i])
                if imgFound > 0:
                    print("Wiping img.tif file to replace it.")
                    os.remove(os.path.join(self.rootpath,'img.tif'))
            self.fname = fname_new(self.rootpath,'img.tif')

            # lets chunk it!
            # beautiful thing about python is that if the loop exceeds the samples, python will grab the remaining samples, despite you requesting more than what exists!
            #chunker = 1000 # number of samples to save over
            time_loop = list(range(t)); time_chunker = time_loop[0::chunker]
            assert time_loop[-1]+chunker > t, "You will not write all samples! Looping mechanism exceeds the total count of samples! FIX ME!"

            # I fed copilot my simpler code and it spit out a code with better error statements and so I kept that
            # Initialize timing
            process_start = time.process_time()
            print(f'Parallel processing set to: {run_parallel}')
            if run_parallel == True:

                # Function to process a chunk
                def process_chunk(start_idx, chunk_size, planes, stride):
                    return [(i, planes[i]) for i in range(start_idx, min(start_idx + chunk_size * stride, len(planes)), stride)]

                # Assuming planes is already defined
                total_length = len(self.planes)
                chunk_size = 1000  # You may adjust this as needed
                stride = z
                num_chunks = total_length // stride

                # Initialize the output array
                output_shape = (z, num_chunks, y, x)
                separated_planes = np.zeros(output_shape, dtype=self.planes[0].dtype)

                with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
                    futures = {executor.submit(process_chunk, i + chunk_idx * stride, chunk_size, self.planes, stride): (i, chunk_idx)
                            for i in range(stride)
                            for chunk_idx in range(0, num_chunks, chunk_size)}

                    completed_results = []

                    for future in concurrent.futures.as_completed(futures):
                        i, chunk_idx = futures[future]
                        chunk = future.result()
                        if chunk:
                            completed_results.append((i, chunk_idx, chunk))
                        progress = (len(completed_results) / len(futures)) * 100
                        print(f"{progress:.2f}% Completed")

                    # Sort the results to maintain order
                    completed_results.sort(key=lambda x: (x[1], x[0]))

                    for i, chunk_idx, chunk in completed_results:
                        for idx, frame in chunk:
                            plane_idx = idx % stride
                            chunk_start = (idx - plane_idx) // stride
                            separated_planes[plane_idx, chunk_start, :, :] = frame
                print("Time to separate and reshape:",time.process_time() - process_start)
                print("Shape of the resulting 4D array:", separated_planes.shape)

                # very critical to ensure that the order matches the original data order
                # assert
                process_start = time.process_time()
                plane0=self.planes[0::3]
                plane1=self.planes[1::3]
                plane2=self.planes[2::3]

                # randomly test for misaligned frames
                randTest = np.random.randint(0, separated_planes.shape[1], 1000)

                for ri in randTest:
                    tempTest0 = plane0[ri]-separated_planes[0,ri,:,:]
                    tempTest1 = plane1[ri]-separated_planes[1,ri,:,:]
                    tempTest2 = plane2[ri]-separated_planes[2,ri,:,:]
                    
                    assert np.max(tempTest0) == 0 and np.max(tempTest1) == 0 and np.max(tempTest2) == 0 and np.min(tempTest0) == 0 and np.min(tempTest1) == 0 and np.min(tempTest2) == 0, "Misaligned frames"
                print("Time to check frame sequence:",time.process_time() - process_start,"sec")

            else:

                # convert to numpy
                print("Converting to numpy. Please wait... BUT If this step is taking too long consider running parallel")
                self.planes = np.array(self.planes)

                assert self.planes.shape[0] % 3 == 0, 'array is improperly divided into frames'

                # separate planes
                separated_planes = np.zeros((3, int(self.planes.shape[0]/3), self.planes.shape[1], self.planes.shape[2]), dtype=self.planes.dtype)    
                print(f'Separating list planes into Z: {separated_planes.shape[0]}, t: {separated_planes.shape[1]}, y: {separated_planes.shape[2]}, x: {separated_planes.shape[3]}')            
                separated_planes[0, :, :, :] = self.planes[0::3] 
                separated_planes[1, :, :, :] = self.planes[1::3] 
                separated_planes[2, :, :, :] = self.planes[2::3]
                print("Time to convert img data to numpy and separate:",time.process_time() - process_start)

            # artifact detection
            if led_artifacts.lower() == 'y':
                print("Running image interpolation for LED artifacts...")
                ledArtifacts = dict(); meanData = dict(); meanXYzData = dict()
                for zi in range(z):
                    print("Working to correct artifacts in plane",zi)

                    # identify candidate artifact events
                    mean_pixels  = np.mean(np.mean(separated_planes[zi],axis=1),axis=1) # get a pixel average over time
                    meanXYz      = np.abs(stat.zscore(mean_pixels,axis=0)) # zscore the averaged pixels
                    ledArtifact  = np.asarray(np.where(meanXYz > 7)).flatten()
                    ledArtifacts['Axis'+str(zi)] =  ledArtifact
                    meanData['Axis'+str(zi)]     =  mean_pixels
                    meanXYzData['Axis'+str(zi)]  =  meanXYz

                    # interpolate missing data
                    for imgi in ledArtifact:
                        if imgi > 1 and imgi < len(meanXYz):
                            print("Interpolating artifact at index:",imgi)

                            # get data surrounding artifact
                            img_temp = np.moveaxis(separated_planes[zi][imgi-1:imgi+2], 0, -1)
                            img_interp = imgfuns.interp_img(img=img_temp)

                            # reshape result
                            img_interp = np.moveaxis(img_interp,-1,0)

                            # replace data
                            separated_planes[zi][imgi] = img_interp[1]

                            # fact check - these are blank arrays as expected
                            # plt.imshow(img_interp[2]-max_proj[imgi+1])
                            # plt.imshow(img_interp[0]-max_proj[imgi-1])

                # save array
                print("Saving ledArtifact data...")
                ledMat = {"ledArtifact": ledArtifacts,
                        "meanXY": meanData,
                        "meanXYz": meanXYzData,
                        "info": "ledArtifact is an index of artifacts. meanXY is the pixel average. meanXYz is |zscore(meanXY)|."}
                artFile = os.path.join(self.rootpath,'ledArtifactDataInterp.mat')
                sio.savemat(artFile, ledMat)

            # max projection
            print("Calculating max projection. This may take a moment...")
            process_start = time.process_time()        
            max_proj = np.max(separated_planes,axis=0) # rewriting same array to help memory
            max_proj = max_proj.astype('int16')
            del separated_planes # clean up memory
            print("Time to calculate max projection:",time.process_time() - process_start)
            print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")

            # ensure that we have the appropriate bit precision
            if max_proj.dtype != 'int16':
                max_proj = max_proj.astype('int16')

            # save the max-projection image
            print("Writing imaging data to:", self.fname)
            #im[:] = max_proj[:] # writes full file to disk. Might cause crashing of VScode
            #im.flush() # write to disk

            # mechanisms to write files to disk
            if memmap_write is True:
                # memory map write
                im = tf.memmap(
                    self.fname,
                    shape=(t,y,x),
                    dtype=np.uint16,
                    imagej=True
                    #append=True
                )
                print("Time (sec):",time.process_time() - code_start)
                print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")

                # This is an alternative method to write iteratively and is less prone to crashing
                for framesi in time_chunker:
                    #temp = []
                    #for zi in range(z):
                    #    temp.append(planes[zi][framesi:framesi+chunker])
                    #max_proj = np.max(np.array(temp),axis=0) # convert to numpy
                    im[framesi:framesi+chunker,:,:] = max_proj[framesi:framesi+chunker,:,:]
                    im.flush() # write to disk
                    del im; im=tf.memmap(self.fname) # clean up memory
                    print("Run time for",str(framesi),"/",str(t),"::: Time (s):",time.process_time() - code_start, "::: Memory:",str(psutil.virtual_memory()[2]),"<%> RAM utility")

                    # validated
                    #np_array = np.array(temp)
                    #fig, ax = plt.subplots(nrows=1,ncols=4)
                    #ax[0].imshow(np_array[0,0,:,:])
                    #ax[0].set_title("Axis1")
                    #ax[1].imshow(np_array[1,0,:,:])
                    #ax[1].set_title("Axis2")
                    #ax[2].imshow(np_array[2,0,:,:])
                    #ax[2].set_title("Axis3")
                    #ax[3].imshow(max_proj[0,:,:])
                    #ax[3].set_title("Max_proj")
            else:
                # quicker write
                tf.imwrite(self.fname, max_proj, dtype=max_proj.dtype, bigtiff=True)
        process_end = time.process_time()
        print(f"Total time spent converting: {(process_end - code_start)/60:.2f} minutes")

    def split_file():
        '''
        This function will be called internally to split the .raw file into multiple separate .raw files which then repopulate fname to then write out the data as needed
        '''
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

# function to delete a .tif file
def remTif(fname):
    '''
    remTif: removes/deletes an img.tif file

    Args:
        >>> fname: path/to/your/img.tif
    
    '''
    # Delete the file if it exists
    if os.path.exists(fname):
        os.remove(fname)
        print(f"{fname} deleted")
    else:
        print(f"{fname} does not exist")

# dysfunctional for now
#TODO Make functional
def importThorsync(bpath):
    '''
    importThorSync
        Equivalent to the MATLAB version. Written by John Stout
        Additions:
            Handles times when your imaging and behavioral data are misaligned by using the piezo monitor

    Args:
        >>> bpath: path to behavioral data, including the .h5 extension

    John Stout merged written code with copilot  
    '''
    # [bData,frameData,trialData]=importThorsync(fileName, subsamp, saveData)

    import h5py
    import os
    import numpy as np

    # Default parameters
    def check_and_set_defaults(subsamp=None, saveData=None):
        if subsamp is None:
            subsamp = [1, 1]
        if saveData is None:
            saveData = True
        return subsamp, saveData

    subsamp, saveData = check_and_set_defaults()

    # Get the extension on the fileName path
    #bpath = "E:\L6 Experiments\L608\FOV1\SEDS_day10_FOV1_optoRec_LBC0\SEDS_day10_FOV1_optoRec_LBC0_beh"
    #bpath = r"F:\John\L6 Experiments\recordings_L5CT\L6-05\FOV1\SD2_odor_day6_FOV1_optoRec\SD2_odor_day6_FOV1_optoRec_restart_beh001"

    bpath = os.path.abspath(bpath)
    ext = os.path.splitext(bpath)[1]

    # Search for .h5 file extension - copilot
    if not ext.endswith('.h5'):
        print("Searching for .h5 file")
        dirFiles = os.listdir(bpath)  # directory contents
        fnames = [f for f in dirFiles if f.endswith('.h5')]  # file names in directory
        fileName = os.path.join(bpath, fnames[0])
        print(f"Discovered and loading: {fnames[0]}")

    # Reading behavioral data
    print("Reading behavioral data from:",fileName)
    dataIn = h5py.File(fileName,'r')
    bData  = dict()
    for i in dataIn['DI'].keys():
        bData[i] = dataIn['DI'][i][:]
        if np.max(bData[i]) > 0:
            bData[i] = np.ravel(bData[i]/np.max(bData[i]))

    # Index of frame times for behavior
    frameTimes = np.where(np.diff(bData['FrameOut'],axis=0)==1)[0]

    # piezo cycles
    piezo = dataIn['AI']['PiezoMonitor'][:]
    piezo_norm = np.ravel(piezo/np.max(piezo)) # scale to 1 and convert to 1D

    # when recordings start, the piezo kicks on
    idx_offrec = np.where(piezo_norm < 0.3)[0]

    # make sure that the first and last value of idx_offrec are outside of frameIdx
    good_rec = ( (idx_offrec[0] < frameTimes[0]) & (idx_offrec[-1] > frameTimes[-1]) )
    #assert good_rec==True, "Your behavioral data are misaligned with your imaging data. Use this session to modify the code"
    if good_rec == False:

        # if the first frame index is less than the first flatlined piezo, that means the experimenter started recording their
        # imaging data before starting thorsync. This happens because the piezo turns on when you hit start on the img software. So there
        # was never a flatlined piezo
        img_too_soon = frameTimes[0] < idx_offrec[0] # you started recording img too soon

        # the experimenter stopped thorsync before the img rec. The last index of frameIdx > last index of piezo.
        img_too_late = frameTimes[-1] > idx_offrec[-1] # you stopped recording img too late

        # if img_too_late
        if img_too_late:
            print("Experimenter turned off 1) ThorSync then 2) ThorImg. Trim end of imaging data.")
            save_tag = "_trimImgEnd"
        elif img_too_soon:
            print("Experimenter turned on 1) ThorImg then 2) ThorSync. Trim the start of imaging data.")
            save_tag = "_trimImgStart"
    
    # Extract velocity data from treadmill rotations
    if 'RotaryA' in bData and 'RotaryB' in bData:
        bData['RotaryA'][bData['RotaryA'] == 4] = 1
        bData['RotaryB'][bData['RotaryB'] == 128] = 1
        position = []
        counter = 0
        for i in range(len(bData['RotaryA']) - 1):
            aState = bData['RotaryA'][i]
            aNextState = bData['RotaryA'][i + 1]
            bNextState = bData['RotaryB'][i + 1]
            if aState != aNextState and aNextState == 1:
                if bNextState != aNextState:
                    counter += 1
                elif bNextState == aNextState:
                    counter -= 1
            position.append(counter)
        position.append(counter)
        position = np.array(position) * -1 * (38/250)  # flip direction and convert to cm
        bData['Velocity'] = np.diff(np.convolve(position, np.ones(100)/100, mode='same')) * 1000  # convert to cm/sec assuming 1kHz sample rate

    # Fit the other behavioral variables based on frameTimes
    frameData = {k: v[frameTimes] for k, v in bData.items()}

    # Get trial data
    trialStartTimes = np.where(np.diff(bData['trialOut']) == 1)[0]
    trialEndTimes = np.where(np.diff(bData['trialOut']) == -1)[0]

    if trialEndTimes[0] < trialStartTimes[0]:
        trialStartTimes = np.insert(trialStartTimes, 0, 0)
    if len(trialStartTimes) > len(trialEndTimes):
        trialEndTimes = np.append(trialEndTimes, len(bData['trialOut']))

    trialData = {'trial': [], 'trialLR': [], 'irrelLR': [], 'setID': [], 'lickNumL': [], 'lickNumR': [], 'trialCorrect': [], 'resDir': [], 'opto': [], 'info': []}
    trialData['info'] = 'Behavioral data from recording session. Note that in python version, zero-indexing is applied. MATLAB saved data add +1 to any indices.'
    for x in range(len(trialStartTimes)):
        trialData['trial'].append(x)
        trialData['trialLR'].append(1 if np.max(bData['trialLROut'][trialStartTimes[x]:trialEndTimes[x]]) == 1 else -1)
        trialData['irrelLR'].append(1 if np.max(bData['irrelLROut'][trialStartTimes[x]:trialEndTimes[x]]) == 1 else -1)
        trialData['setID'].append(1 if np.max(bData['setIDOut'][trialStartTimes[x]:trialEndTimes[x]]) == 1 else -1)
        trialData['lickNumL'].append(len(np.where(np.diff(bData['lickingLOut'][trialStartTimes[x]:trialEndTimes[x]]) == 1)[0]))
        trialData['lickNumR'].append(len(np.where(np.diff(bData['lickingROut'][trialStartTimes[x]:trialEndTimes[x]]) == 1)[0]))
        trialData['trialCorrect'].append(np.max(bData['rewardOut'][trialStartTimes[x]:trialEndTimes[x]]))
        trialData['resDir'].append(trialData['trialLR'][x] if trialData['trialCorrect'][x] == 1 else trialData['trialLR'][x] * -1)
        
        if x < len(trialStartTimes)-1:
            trialData['opto'].append(1 if np.max(bData['LEDStim'][trialEndTimes[x]:trialStartTimes[x+1]]) == 1 else 0)
        else:
            trialData['opto'].append(0)
    
    # use diff to identify select events
    lickingL = np.diff(bData['lickingLOut'])
    lickingR = np.diff(bData['lickingROut'])
    rewarded = np.diff(bData['rewardOut'])
    opto     = np.diff(bData['LEDStim'])

    # get the absolute index, the index of the events, irrespective of your imaging data
    timesData = {'trialStartTimes': trialStartTimes, 'trialEndTimes': trialEndTimes,
                  'lickTimesL': [], 'lickTimesR': [], 'rewardTimes': [], 'optoOnTimes': [], 'info': []}   
    timesData['info'] = 'Behavioral timestamp indices from recording session. Note that these are indices to the behavioral frames. You would have to align these with frameTimes. Note that in python version, zero-indexing is applied. MATLAB saved data add +1 to any indices.'
    for ti in range(len(trialStartTimes)):

        # Find indices where lickingLOut equals 1 within the trial range
        idx_lickL = [i for i in range(trialStartTimes[ti], trialEndTimes[ti]) if lickingL[i] == 1]
        idx_lickR = [i for i in range(trialStartTimes[ti], trialEndTimes[ti]) if lickingR[i] == 1]

        # you only need the first index for this variable because it will return all times when digital pulse==1
        idx_rew = [i for i in range(trialStartTimes[ti], trialEndTimes[ti]) if rewarded[i] == 1]
        if len(idx_rew) > 0:
            idx_rew = idx_rew[0]
        else:
            idx_rew = np.nan

        # opto happens in between trials, during the ITI, so search throughout trial onset to trial onset+1
        if ti < len(trialStartTimes) - 1:
            idx_opto = [i for i in range(trialStartTimes[ti], trialStartTimes[ti+1]) if opto[i] == 1]
        else:
            # search from the start of the last trial to the end of the session
            idx_opto = [i for i in range(trialStartTimes[ti], len(lickingL)) if opto[i] == 1]
        
        # Append found indices to the list
        timesData['lickTimesL'].append(idx_lickL)
        timesData['lickTimesR'].append(idx_lickR)
        timesData['rewardTimes'].append(idx_rew)
        timesData['optoOnTimes'].append(idx_opto)

    # sanity check - thanks copilot
    def assert_same_length(data_dict, dict_name):
        lengths = [len(data_dict[v]) for v in data_dict.keys() if v != 'info']
        assert len(set(lengths)) == 1, f"Not all lists in {dict_name} are of the same length. Lengths found: {lengths}"

    # Check the lengths of timesData and trialData
    assert_same_length(timesData, "timesData")
    assert_same_length(trialData, "trialData")

    # make into a dict
    beh_dict = {'timesData': timesData,
                'trialData': trialData,
                'frameData': frameData,
                'bData': bData,
                'frameTimes': frameTimes}

    # save the variable as .npy file
    #np.save(os.path.join(bpath,'behPy.npy'), beh_dict, allow_pickle = True)

    # in the case for MATLAB analysis, we want to add a sample to relevant arrays
    timesData['rewardTimes'] = [i+1 if not np.isnan(i) else i for i in timesData['rewardTimes']]
    timesData['trialStartTimes'] = timesData['trialStartTimes'] + 1
    timesData['trialEndTimes'] = timesData['trialEndTimes'] + 1
    trialData['trial'] = [i+1 if not np.isnan(i) else i for i in trialData['trial']]
    frameTimes = frameTimes + 1

    # save
    if good_rec == False:
        sio.savemat(os.path.join(bpath,'beh'+save_tag+'.mat'), {
            'timesData': timesData,
            'trialData': trialData,
            'bData': bData,
            'frameData': frameData,
            'frameTimes': frameTimes,
            }
        )
    else:
        sio.savemat(os.path.join(bpath,'beh.mat'), {
            'timesData': timesData,
            'trialData': trialData,
            'bData': bData,
            'frameData': frameData,
            'frameTimes': frameTimes,
            }
        )



'''
    animals = "L6_A03"
    session = "SD1_d1"
    recording_type = "recordings_L5CT" #"recordings_panneuronal"

    animals = "L6-02"
    session = "SD1_whisker_d1"
    recording_type = 'recordings_panneuronal' #"recordings_L5CT" #"recordings_panneuronal"

    animals = "L6-02"
    session = "SH5"
    recording_type = 'recordings_panneuronal' #"recordings_L5CT" #"recordings_panneuronal"

    fname = os.path.join(r"F:\John\L6 Experiments",recording_type,animals,"sessions",session,"img","img.tif")
    bname = os.path.join(r"F:\John\L6 Experiments",recording_type,animals,"sessions",session,"beh")
    bfile = [i for i in os.listdir(bname) if '.h5' in i]
    assert len(bfile)==1, "This code does not support multiple episodes"
    bname = os.path.join(bname,bfile[0])
    
    data = h5py.File(bname,'r')

    print("Reading behavioral data from:",bname)
    bData = dict()
    for i in data['DI'].keys():
        bData[i] = data['DI'][i][:]
        if np.max(bData[i]) > 0:
            bData[i] = np.ravel(bData[i]/np.max(bData[i]))
    
    # Index of frame times for behavior
    frameIdx = np.where(np.diff(bData['FrameOut'],axis=0)==1)

    if check_piezo and len(imgpath) > 0:

        # piezo cycles
        piezo=data['AI']['PiezoMonitor'][:]
        piezo_norm = np.ravel(piezo/np.max(piezo)) # scale to 1 and convert to 1D

        assert bData['FrameOut'].shape[0]==piezo.shape[0], "Your piezo monitor and behavioral data are misaligned"

        # frames with piezo - this is where you find misalignments
        piezo_frames = np.ravel(piezo_norm[frameIdx])

        # can add one frame to this data
        img = tf.memmap(fname, mode='r')
        img.shape

        if len(piezo_frames) != (img.shape[0]*4):
            offset = len(piezo_frames)-(img.shape[0]*4)
            print("Your pre-downsampled data are off by",offset,"samples")

            # bc the piezo motor is mechanical, our rescaling to 0 and 1 should always give us similar answers
            # as such, here is our thresholding technique to find problematic time points
            idxMiss = np.ravel(np.where(piezo_frames < 0.35))

            fig, ax = plt.subplots(nrows=3,ncols=1)
            ax[0].plot(piezo_frames); ax[0].set_title("Full frame")
            ax[1].plot(piezo_frames); ax[1].set_xlim((0,100)); ax[1].set_title("First samples")
            ax[2].plot(piezo_frames); ax[2].set_xlim((len(piezo_frames)-100,len(piezo_frames))); ax[1].set_title("Last samples")

            shave_out = input("Should we shave off time points at the start or end? [start/end]")
            if shave_out == 'end':
                piezo_frames = piezo_frames[:-offset]
                frameIdx     = np.ravel(frameIdx)[:-offset]
            elif shave_out == 'start':
                piezo_frames = piezo_frames[offset::] # check
                frameIdx     = np.ravel(frameIdx)[offset::] # check

    # now get variables to save
    frameTimes = frameIdx[0::4] # every 4th datapoint because max_projection
    LEDStim    = bData['LEDStim'][frameTimes]
    irrelLR    = bData['irrelLROut'][frameTimes]
    trialLR    = bData['trialLROut'][frameTimes]
    setID      = bData['setIDOut'][frameTimes]
    trial      = bData['trialOut'][frameTimes]
    reward     = bData['rewardOut'][frameTimes]
    lickTimesL = bData['lickingLOut'][frameTimes]
    lickTimesR = bData['lickingROut'][frameTimes]
    behCam     = bData['BehaviorCam'][frameTimes]

    # build other parts of the matlab code here


    # SCRAP


    # if you convert with matlab, this matches perfectly, but if you convert with python it doesnt
    # this is because the very last frame is a blank framein matlab and python tosses it. This must have something
    # to do with the way the thor software works
    len_match = len(frameIdx) == img.shape[0] * 4
    if len_match == False:
        len_match = len(frameIdx) == (img.shape[0]+1) * 4
        # reassess whether the data are now a match
        if len_match == True:
            converter = 'python'
            rec_shape = (img.shape[0]+1) * 4

            # correct for offset
            img_offset = img.shape[0] * 4 - len(frameIdx)

            # correct the frameIdx
            print("Removing",img_offset,"samples from frameIdx to correct for blank frame in thorSync")
            frameIdx = frameIdx[:img_offset]
    else:
        converter = 'matlab'
        rec_shape = img.shape[0] * 4

    # TODO STOPPED HERE
    if len_match == False:
        print("Attempting to align your behavioral and imaging data")


        

        # frames with piezo - this is where you find misalignments
        piezo_frames = np.ravel(piezo_norm[frameIdx])







        if converter == 'python':
            # trim off n number of samples from piezeo_frames
            piezo_frames = piezo_frames + img_offset

        if len(piezo_frames) != rec_shape:
            offset = len(piezo_frames)-(img.shape[0]*4)
            print("Your pre-downsampled data are off by",offset,"samples")

            # bc the piezo motor is mechanical, our rescaling to 0 and 1 should always give us similar answers
            # as such, here is our thresholding technique to find problematic time points
            idxMiss = np.ravel(np.where(piezo_frames < 0.35))

            fig, ax = plt.subplots(nrows=3,ncols=1)
            ax[0].plot(piezo_frames); ax[0].set_title("Full frame")
            ax[1].plot(piezo_frames); ax[1].set_xlim((0,100)); ax[1].set_title("First samples")
            ax[2].plot(piezo_frames); ax[2].set_xlim((len(piezo_frames)-100,len(piezo_frames))); ax[1].set_title("Last samples")

            shave_out = input("Should we shave off time points at the start or end? [start/end]")
            if shave_out == 'end':
                piezo_frames = piezo_frames[:-offset]
                frameIdx     = np.ravel(frameIdx)[:-offset]
            elif shave_out == 'start':
                piezo_frames = piezo_frames[offset::] # check
                frameIdx     = np.ravel(frameIdx)[offset::] # check

'''
