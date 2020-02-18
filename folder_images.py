import glob
import shutil
import os

source_files='data/Image/*.jpg'

filelist=glob.glob(source_files)
for single_file in filelist:
    patient_id=single_file.split()[1]
    file_name=single_file[11:].replace(' ','_')
    target_folder = f'data/Image/{patient_id}'
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    shutil.move(single_file,target_folder+'/'+file_name)