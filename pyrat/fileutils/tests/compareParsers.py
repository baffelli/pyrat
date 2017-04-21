import datetime as dt
from pyrat.fileutils.parsers import  FasterParser, ParameterParser
from pyrat.fileutils.parameters import ParameterFile
import timeit



fast = ParameterFile.from_file('../default_slc_par.par', parser=FasterParser)
# print(fast)
slow = ParameterFile.from_file('../default_slc_par.par', parser=ParameterParser)
