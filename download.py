import glob
import shutil

# I prefer to set path and mask as variables, but of course you can use values
# inside glob() and move()

source_files='data/Image/*.jpg'
target_folder='data/lateral'

# retrieve file list
filelist=glob.glob(source_files)
for single_file in filelist:
     # move file with full paths as shutil.move() parameters
    shutil.move(single_file,target_folder) 