from utils import create_data_lists
import os
if __name__ == '__main__':
    create_data_lists(voc07_path='C:\VOCdevkit\VOC2007',
                      output_folder='C:\VOCdevkit')
# print(os.path.isdir('C:\VOCdevkit'))