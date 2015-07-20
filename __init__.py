__all__ = ["core","visualization","intfun"]
from core.polfun import *
from core.corefun import *
from core.matrices import *
from visualization import visfun
from fileutils.gpri_files import gammaDataset
import subprocess as _sp
import os as _os
#Set environment variables
_os.environ['GAMMA_HOME']='/usr/local/GAMMA_SOFTWARE-20130717'
_os.environ['ISP_HOME']=_os.environ['GAMMA_HOME'] + '/ISP'
_os.environ['DIFF_HOME']=_os.environ['GAMMA_HOME'] + '/DIFF'
_os.environ['LD_LIBRARY_PATH']=_os.environ['GAMMA_HOME'] +'/lib'