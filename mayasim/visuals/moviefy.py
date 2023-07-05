import glob
import os
import shutil
import sys
from subprocess import call


def moviefy(input_folder, output_folder=None, rmold=False, namelist=['image_']):
    """
    type: (str, str) -> object 
    """

    framerate = 8
    if output_folder is None:
        output_folder = input_folder

    input_folder = '/' + input_folder.strip('/') + '/'
    output_folder = '/' + output_folder.strip('/') + '/'

    # namelist = ['AG_', 'bca_', 'es_',
    #             'forest', 'influence_cropped_b_',
    #             'npp_', 'pop_grad_', 'rain_',
    #             'soil_deg_', 'trade_network_',
    #             'waterflow_']

    for name in namelist:
        input_string = input_folder + name + "%03d.png"
        del_string = input_folder + name + "*.png"
        output_string = output_folder + name.strip('_') + '.mp4'
        call(["ffmpeg", "-y", "-r", repr(framerate), "-i", input_string, output_string])
        if rmold:
            for fl in glob.glob(del_string):
                os.remove(fl)

if __name__ == '__main__':

    print(sys.argv)

    if len(sys.argv) < 3:
        print('usage is python moviefy.py [input folder] [output folder]')

    if os.path.exists(sys.argv[2]):
        if sys.argv[1] != sys.argv[2]:
            shutil.rmtree(sys.argv[2])
            os.makedirs(sys.argv[2])

    moviefy(sys.argv[1], sys.argv[2])
