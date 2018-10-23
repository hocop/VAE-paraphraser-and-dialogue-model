# Easy Filesystem Routines
# list all folder names in given directory
import os
def onlyFolders(path, recursive=False, full_path=False):
        path_list=[]
        if os.path.lexists(path):
            if recursive is False:
                if full_path is False:
                    for filename in os.listdir(path):
                        if os.path.isdir(os.path.join(path,filename)):
                            path_list.append(filename)
                    return path_list
                else:
                    for filename in os.listdir(path):
                        if os.path.isdir(os.path.join(path,filename)):
                            path_list.append(os.path.normpath(os.path.join(path, filename)))
                    return path_list
            else:
                if full_path is False:
                    for root, dirs, filename in os.walk(path, topdown=False):
                        for name in dirs:
                            if os.path.isdir(os.path.join(root,name)):
                                path_list.append(name)
                    return path_list
                else:
                    for root, dirs, filename in os.walk(path, topdown=False):
                        for name in dirs:
                            fp=os.path.realpath(os.path.join(root, name))
                            path_list.append(fp)
                    return path_list                   
        else:
            print ('Path does not exist')
	

# print all file names in given directory
def onlyFiles(path, recursive=False, full_path=False):
        path_list=[]
        if os.path.lexists(path):
            if recursive is False:
                if full_path is False:
                    for filename in os.listdir(path):
                        if os.path.isfile(os.path.join(path,filename)):
                            path_list.append(filename)
                    return path_list
                else:
                    for filename in os.listdir(path):
                        if os.path.isfile(os.path.join(path,filename)):
                            path_list.append(os.path.normpath(os.path.join(path, filename)))
                    return path_list
            else:
                if full_path is False:
                    for root, dirs, filename in os.walk(path, topdown=False):
                        for name in filename:
                                path_list.append(name)
                    return path_list
                else:
                    for root, dirs, filename in os.walk(path, topdown=False):
                        for name in filename:
                            fp=os.path.realpath(os.path.join(root, name))
                            path_list.append(fp)
                    return path_list                   
        else:
            print ('Path does not exist')
	

