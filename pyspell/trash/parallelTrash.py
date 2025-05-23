
# List to store the data
np_mem_list = []

# Number of chunks to load
num_chunks = len(self.idx_offset_np)

# Use ProcessPoolExecutor for parallel processing
workers = os.cpu_count()
workers = workers-2

# Function to process a batch of indices
'''
bufferedIO = False # doesn't really work
if bufferedIO:
    # Function to process a batch of indices with buffered I/O
    def load_chunk_batch(idxi_batch, filepath, shape):
        results = []
        buffer_size = 64 * 1024  # 64KB buffer size
        for idxi in idxi_batch:
            try:
                with open(filepath, 'rb', buffering=buffer_size) as f:
                    f.seek(idxi)
                    data = np.frombuffer(f.read(shape[0] * shape[1] * np.dtype('int16').itemsize), dtype='int16').reshape(shape)
                    results.append(data)
            except Exception as e:
                print(f"Error processing chunk at offset {idxi}: {e}")
        return
else:
    def load_chunk_batch(idxi_batch, filepath, shape):
        results = []
        for idxi in idxi_batch:
            try:
                results.append(np.array(np.memmap(filepath, dtype='int16', offset=idxi, mode='r', shape=shape)))
            except Exception as e:
                print(f"Error processing chunk at offset {idxi}: {e}")
        return results

# List to store the data
np_mem_list = []

# Batch size
batch_size = chunker  # Adjust batch size based on performance testing

# Number of chunks to load
num_chunks = len(self.idx_offset_np)

# Initialize timing
process_start = time.process_time()

# Use ThreadPoolExecutor for parallel processing
with ThreadPoolExecutor(max_workers=workers) as executor:  # Adjust max_workers based on your CPU cores
    futures = []
    for i in range(0, num_chunks, batch_size):
        batch = self.idx_offset_np[i:i + batch_size]
        futures.append(executor.submit(load_chunk_batch, batch, self.filepath, (x, y)))
        progress = (i + batch_size) / num_chunks * 100
        print(f"{progress:.2f}% submitted for processing.")

    completed_batches = 0
    for future in as_completed(futures):
        start_time = time.time()
        try:
            results = future.result(timeout=60)  # Add timeout to each future result
            np_mem_list.extend(results)
            completed_batches += 1
            progress = (completed_batches * batch_size / num_chunks) * 100
            print(f"{progress:.2f}% complete loading/converting to numpy.")
            elapsed_time = time.time() - start_time
            print(f"{progress:.2f}% complete loading/converting to numpy. Time: {elapsed_time:.2f} seconds")
            print("Update:", str(psutil.virtual_memory()))            
        
        except TimeoutError:
            print("A chunk took too long to process and was skipped.")
        except Exception as e:
            print(f"Error occurred: {e}")

    # Periodically restart the executor to free up resources
    executor.shutdown(wait=True)

# Terminate leftover processes explicitly
multiprocessing.active_children()
for p in multiprocessing.active_children():
    p.terminate()
'''

# Function to process a batch of indices with buffered I/O
def load_chunk_batch(idxi_batch, filepath, shape):
    results = []
    buffer_size = 64 * 1024  # 64KB buffer size
    for idxi in idxi_batch:
        try:
            with open(filepath, 'rb', buffering=buffer_size) as f:
                f.seek(idxi)
                data = np.frombuffer(f.read(shape[0] * shape[1] * np.dtype('int16').itemsize), dtype='int16').reshape(shape)
                results.append(data)
        except Exception as e:
            print(f"Error processing chunk at offset {idxi}: {e}")
    return results

# List to store the data
np_mem_list = []

# Number of chunks to load
num_chunks = len(self.idx_offset_np)

# Batch size
batch_size = chunker  # Adjust batch size based on performance testing

# Initialize timing
process_start = time.process_time()

# shut down parallel processes if active
active_children = multiprocessing.active_children()
for p in active_children:
    p.terminate()

# Use ThreadPoolExecutor for parallel processing
with ThreadPoolExecutor(max_workers=os.cpu_count()-2) as executor:  # Adjust max_workers based on your CPU cores
    futures = []
    for i in range(0, num_chunks, batch_size):
        batch = self.idx_offset_np[i:i + batch_size]
        futures.append(executor.submit(load_chunk_batch, batch, self.filepath, (x, y)))
        progress = (i + batch_size) / num_chunks * 100
    print("Data submitted for processing.")

    # this is the time consuming step. There seems to be an IO issue.
    completed_batches = 0
    for future in as_completed(futures):
        start_time = time.time()
        try:
            results = future.result(timeout=60)  # Add timeout to each future result
            if results:
                np_mem_list.extend(results)
            completed_batches += 1
            progress = (completed_batches * batch_size / num_chunks) * 100
            elapsed_time = time.time() - start_time
            print(f"{progress:.2f}% complete loading/converting to numpy. Time: {elapsed_time:.2f} seconds")
            print("Update:", str(psutil.virtual_memory()))
        except TimeoutError:
            print("A chunk took too long to process and was skipped.")
            # Retry logic can be added here if needed
        except Exception as e:
            print(f"Error occurred: {e}")

process_end = time.process_time()
print("Complete loading/converting to numpy.")
print(f"Total time: {process_end - process_start:.2f} seconds")

# Explicitly shutdown remaining processes
executor.shutdown(wait=True)
active_children = multiprocessing.active_children()
for p in active_children:
    p.terminate()

# update time records
process_end = time.process_time()
print("Complete loading/converting to numpy.")
print(f"Total time: {process_end - process_start:.2f} seconds")









            def reorganize_chunk(zi, z, planes, t, y, x):
                return zi, np.array(planes[zi::z]).reshape(t, y, x)

            if __name__ == '__main__':
                # Load and convert the .Raw data to np.array using parallel processing
                np_mem_list = []
                num_chunks = len(self.planes)
                chunk_size = 1000  # Adjust chunk size based on performance testing

                process_start = time.process_time()
                with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
                    #load_futures = [
                    #    executor.submit(load_and_convert_chunk, i, i + chunk_size, self.vector_list) 
                    #    for i in range(0, num_chunks, chunk_size)
                    #]
                    
                    #for future in concurrent.futures.as_completed(load_futures):
                    #    try:
                    #        result = future.result()
                    #        np_mem_list.extend(result)
                    #        progress = len(np_mem_list) / num_chunks * 100
                    #        print(f"{progress:.2f}% complete loading/converting to numpy. Update: {psutil.virtual_memory()[2]}% RAM utility")
                    #    except Exception as e:
                    #        print(f"Error processing chunk: {e}")
                
                #process_end = time.process_time()

                # Reorganize the data into a 4D array using parallel processing
                imgData = np.zeros(shape=(z, t, y, x))
                process_start = time.process_time()
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
                    reorganize_futures = [
                        executor.submit(reorganize_chunk, zi, z, planes, t, y, x) 
                        for zi in range(z)
                    ]
                    
                    for future in concurrent.futures.as_completed(reorganize_futures):
                        try:
                            zi, chunk_data = future.result()
                            planes[zi, :, :, :] = chunk_data
                            progress = (zi + 1) / z * 100
                            print(f"{progress:.2f}% Completed")
                            print("Update:", str(psutil.virtual_memory()[2]), "<%> RAM utility")
                        except Exception as e:
                            print(f"Error processing slice {zi}: {e}")
                
                print("Time to reorganize (sec):", time.process_time() - process_start)

            # split data into z number of planes
            planeZ = []
            for zi in range(z):
                planeZ.append(planes[zi::3]) # planeZ[0] has t number of data points
            assert len(planeZ[0])==t, "The expected time points do not match the extracted time points"

            # convert to numpy
            if runParallel == False:

                # This converts the .Raw data into np.arrays img by img
                np_mem_list = []
                counter = 0
                num_chunks = len(self.planes)
                for idxi in range(len(self.planes)):#self.idx_offset_np:
                    try:
                        #np_mem_list.append(np.array(np.memmap(self.filepath, dtype='int16', offset=idxi, mode='r', shape=(x, y))))
                        np_mem_list.append(np.array(self.vector_list[idxi])) 
                        counter += 1
                        progress = (counter / num_chunks) * 100
                        print(f"{progress:.2f}% complete loading/converting to numpy. Update: {psutil.virtual_memory()[2]}% RAM utility")
                    except Exception as e:
                        print(f"Error processing chunk at offset {idxi}: {e}")
                process_end = time.process_time()
                print("Complete loading/converting to numpy.")

                # this reorganizes the data into a 4D array
                process_start = time.process_time()        
                planes = np.zeros(shape=(z,t,y,x)); 
                for zi in range(z):
                    planes[zi,:,:,:] = np_mem_list[zi::z]
                    print(f"{(zi + 1) / z * 100:.2f}% Completed")
                    print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")
                print("Time to reorganize (sec):",time.process_time() - process_start)                

            #np_mem_list = []; counter = 0
            #for idxi in self.idx_offset_np:
            #    np_mem_list.append(np.array(np.memmap(self.filepath, dtype='int16', offset=idxi, mode='r', shape=(x,y))))
            #    counter += 1
            #    print("% Complete loading/converting to numpy:",(counter/len(self.idx_offset_np))*100)
                #np_mem_list.append(np.load(self.filepath, dtype='int16', offset=idxi, mode='r', shape=(x,y)))

            else:


                # TODO: FACT CHECK AGAINST MATLABS02
                def load_and_convert_chunk(start_idx, end_idx, vector_list):
                    return [np.array(vector_list[i]) for i in range(start_idx, min(end_idx, len(vector_list)))]

                def reorganize_chunk(zi, z, np_mem_list, t, y, x):
                    return zi, np.array(np_mem_list[zi::z]).reshape(t, y, x)

                if __name__ == '__main__':
                    # Load and convert the .Raw data to np.array using parallel processing
                    np_mem_list = []
                    num_chunks = len(self.vector_list)
                    chunk_size = 1000  # Adjust chunk size based on performance testing

                    process_start = time.process_time()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
                        load_futures = [
                            executor.submit(load_and_convert_chunk, i, i + chunk_size, self.vector_list) 
                            for i in range(0, num_chunks, chunk_size)
                        ]
                        
                        for future in concurrent.futures.as_completed(load_futures):
                            try:
                                result = future.result()
                                np_mem_list.extend(result)
                                progress = len(np_mem_list) / num_chunks * 100
                                print(f"{progress:.2f}% complete loading/converting to numpy. Update: {psutil.virtual_memory()[2]}% RAM utility")
                            except Exception as e:
                                print(f"Error processing chunk: {e}")
                    
                    process_end = time.process_time()
                    print("Complete loading/converting to numpy.")
                    print(f"Total time: {process_end - process_start:.2f} seconds")
                    
                    # Reorganize the data into a 4D array using parallel processing
                    planes = np.zeros(shape=(z, t, y, x))
                    process_start = time.process_time()
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
                        reorganize_futures = [
                            executor.submit(reorganize_chunk, zi, z, np_mem_list, t, y, x) 
                            for zi in range(z)
                        ]
                        
                        for future in concurrent.futures.as_completed(reorganize_futures):
                            try:
                                zi, chunk_data = future.result()
                                planes[zi, :, :, :] = chunk_data
                                progress = (zi + 1) / z * 100
                                print(f"{progress:.2f}% Completed")
                                print("Update:", str(psutil.virtual_memory()[2]), "<%> RAM utility")
                            except Exception as e:
                                print(f"Error processing slice {zi}: {e}")
                    
                    print("Time to reorganize (sec):", time.process_time() - process_start)
