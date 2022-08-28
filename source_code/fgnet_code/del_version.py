__author__ = 'dk'
#rename 版本号
import os
import shutil
for _root,_dirs,_files in os.walk("./fsnet/") :
    for file in _files:
        nfile = file.split('-')[0]+".num"
        os.rename(_root+file,_root+nfile)
        print('{0}->{1}'.format(_root+file,_root+nfile))