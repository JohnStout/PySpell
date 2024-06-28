

# write to disk
z_counter = 0; temp_array = np.zeros((z,x,y))
print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")
for frame in range(len(vector_list)): # loop over every memory mapped file in the list
    # store data temporarily as a 3D array
    temp_array[z_counter,:,:]=vector_list[frame]
    # add to the counter
    z_counter+=1
    # reset the counter to keep multiples of 3 (z-plane)
    if z_counter == 3:
        im[:,frame,:,:]=temp_array
        im.flush()
        # reset the counter and temporary zeros array    
        z_counter = 0; temp_array = np.zeros((z,x,y))
    print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")
print("Run time:",time.process_time() - code_start,"s")
print(time.process_time() - code_start)



# try to write data to disk in an imagej friendly format
fname = os.path.join(rootpath,'img_memmap.tif')
with tf.tifffile.TiffWriter(fname, imagej=True) as tif:
    print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")
    z_counter = 0; temp_array = np.zeros((z,x,y))
    print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")
    for frame in vector_list: # loop over every memory mapped file in the list
        # store data temporarily as a 3D array
        temp_array[z_counter,:,:]=frame
        # add to the counter
        z_counter+=1
        # reset the counter to keep multiples of 3 (z-plane)
        if z_counter == 3:
            # write to disk in chunks
            tif.memmap(
                temp_array,
                contiguous=True,
                metadata={'fps': fr, 'finterval': 1 / fr},
            )   
            # reset the counter and temporary zeros array    
            z_counter = 0; temp_array = np.zeros((z,x,y))
        print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")
print("Run time:",time.process_time() - code_start,"s")



# now we will append to memory mapped file
print("Please wait while data is mapped to:",fname)
offset=0
try:
    for ti in range(t):
        for zi in range(z):   
            # load in vectorized frame    
            single_frame = np.memmap(filepath, dtype='int16', offset=offset, mode='r', shape=x*y)
            # invert
            temp_frame = np.reshape(single_frame, newshape=(x,y))

            offset+=int(x*y*16/8) # bytes (16bit/8) https://stackoverflow.com/questions/16149803/working-with-big-data-in-python-and-numpy-not-enough-ram-how-to-save-partial-r
        # skip the flyback frame
        offset+=int(x*y*16/8)
except:
    print("Aborting loop at:",str(ti),"/",str(t))

# not really worth moving forward here



#filesize.flush()


scaled_imshow(filein[0,0,:,:])















# figure root
rootpath = os.path.split(filepath)[0]

# get metadata
root_contents = os.listdir(rootpath)
metadata_file = [i for i in root_contents if '.xml' in i][0]
metadata_path = os.path.join(rootpath,metadata_file)
file = xmltodict.parse(open(metadata_path,"r").read()) # .xml file

# define frame rate based on metadata
fr = float(file['ThorImageExperiment']['LSM']['@frameRate'])

# define .raw file fname
raw_files = [os.path.join(rootpath,i) for i in root_contents if '.raw' in i]

# get dimensions of recorded data
x=int(file['ThorImageExperiment']['LSM']['@pixelX'])
y=int(file['ThorImageExperiment']['LSM']['@pixelY'])
t=int(file['ThorImageExperiment']['Timelapse']['@timepoints']) # this is how the thorlabs code works
z=int(file['ThorImageExperiment']['ZStage']['@steps']) # check this variable
dims=(z,t,y,x)

# matlab code loops over time, then over z, then makes the file. Likewise, it assumes the number of data points using metadata, then just fills in
# read a single frame

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
fname = os.path.join(rootpath,'img_memmap.tif')
print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")

# create a memory mappable file, with vectorized data
im = tf.memmap(
    fname,
    shape=(z,ti,y,x),
    dtype=np.uint16,
    append=True
)
print(time.process_time() - code_start)
print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")

# loop over data 
frame_counter = 0
for timei in range(ti):
    for zi in range(z):
        im[zi,timei,:,:] = vector_list[frame_counter]
        im.flush()
        frame_counter+=1
    print("Run time for:",str(timei),"/",str(ti), time.process_time() - code_start)
print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")

# suite2p wants it differently
frame_counter = 0
total_count = ti*z
for framei in range(total_count):
    im[framei,:,:] = vector_list[frame_counter]
    im.flush()
    frame_counter+=1
    print("Run time:", time.process_time() - code_start)
print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")

for timei in range(ti):
    for zi in range(z):
        im[zi,timei,:,:] = vector_list[frame_counter]
        im.flush()
        frame_counter+=1
    print("Run time for:",str(timei),"/",str(ti), time.process_time() - code_start)
print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")



            # loop over data - DELETE ME
            print("Please wait while data are written to disk...")
            frame_counter = 0
            for timei in range(t):
                for zi in range(z):
                    im[zi,timei,:,:] = self.vector_list[frame_counter]
                    im.flush()
                    frame_counter+=1
                print("Run time for:",str(timei),"/",str(t), time.process_time() - code_start)
            print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")






# WORKED TO CONVERT FROM NP
self.fname = fname_new(rootpath,'img_mmap_suite2p_z.tif')
im = tf.memmap(
    self.fname,
    shape=(counter,y,x),
    dtype=np.uint16,
    imagej=True,
    #append=True
)    

# writing data to disk in chunks (500)
print("Please wait while data are written to disk using a 'chunking' mechanism...")

# beautiful thing about python is that if the loop exceeds the samples, python will grab the remaining samples, despite you requesting more than what exists!
count_range = list(range(counter)) # define the range over which to sample data
chunk_samples = int(500)
chunk_loop = count_range[0::chunk_samples] # skip every chunk_samples samples
assert chunk_loop[-1]+chunk_samples > counter, "You will not write all samples! Looping mechanism exceeds the total count of samples! FIX ME!"

# instead of pulling all of that into memory, lets write it immediately, then call the mapped data
offset=0; vector_list = []; counter = 0; offset_list = []
try:
    for ti in range(t):
        for zi in range(z):     
            im[counter,:,:]=np.memmap(filepath, dtype='int16', offset=offset, mode='r', shape=(x,y))
            offset+=int(x*y*16/8) # bytes (16bit/8)
            offset_list.append(offset)
            counter+=1 
            del im; im=tf.memmap(self.fname)                      
        # skip the flyback frame
        offset+=int(x*y*16/8)
except:
    print("Aborting loop at:",str(ti),"/",str(t))