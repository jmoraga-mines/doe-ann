'''

  DOE Project : 
  Task        : 
  File        : 
  
    This is the main program to create, train and use an ANN to classify
  regions based on geothermal potential.

'''
from __future__ import print_function
'''
# In case we need to append directories to Python's path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
#sys.path.append(os.path.join(ROOT_DIR, 'a_directory'))
'''

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

''' Class definitions '''
class doe_ann_object(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir


''' Main program '''

if __name__ == 'main':
  ''' Main instructions '''
