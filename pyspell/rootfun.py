import os
import sys

def dropbox_root(dropbox_folder: str = 'timspellman'):
    root = os.getcwd().split(dropbox_folder)[0]
    return root

def addcode_paths(code_root: str):
    local_root = code_root.split('PySpell')[0]
    local_packages = os.listdir(local_root) # packages in John folder

    # get lab root
    lab_root = os.path.split(os.path.split(local_root)[0])[0]
    lab_folders = os.listdir(lab_root) # all folders in timspellman dropbox

    # get matlab and python folder
    matlab_folder = os.path.join(lab_root,'MATLAB')
    python_folder = os.path.join(lab_root,'Python')

    # add the local packages to path
    [sys.path.append(os.path.join(local_root,i,'code')) for i in local_packages]
    print("Added the following packages to path:",local_packages)

def create_path(fname):
    if os.path.isdir(fname)==False:
        os.mkdir(fname)

def list_all_subdirs(phile_name):
    '''
    Recursive method to list all subdirectories. 

    Written by Alex M and converted to Python from MATLAB via copilot
    
    '''

    # Initialize the list
    dir_paths = []

    # Start by listing all subdirectories of the main folder
    subdirs = [d for d in os.scandir(phile_name) if d.is_dir()]

    # Loop through each subdirectory found
    for subdir in subdirs:
        # Get the full path of the subdirectory
        sub_dir_path = os.path.join(phile_name, subdir.name)

        # Weed out any empty directories
        if len(os.listdir(sub_dir_path)) > 0:
            # Add this subdirectory path to the list
            dir_paths.append(sub_dir_path)

            # Recursively call this function to get sub-subdirectories
            sub_folder_paths = list_all_subdirs(sub_dir_path)

            # Append any found sub-subdirectories to the list
            dir_paths.extend(sub_folder_paths)

    return dir_paths

def is_folder_busy(folder_path):

    '''
    Copilot wrote this to test if a folder is busy
    
    '''
    try:
        # Try to create a temporary file in the folder
        temp_file_path = os.path.join(folder_path, 'temp_file')
        with open(temp_file_path, 'w') as temp_file:
            temp_file.write('Test')
        
        # Try to remove the temporary file
        os.remove(temp_file_path)
        
        # If no exception was raised, the folder is not busy
        return False
    except (OSError, IOError):
        # If an exception was raised, the folder is busy
        return

