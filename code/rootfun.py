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