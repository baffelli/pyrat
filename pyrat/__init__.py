import glob as _glob
import os as _os

# This loads all rules files to the package for them to be accessible
all_rules = _glob.glob(_os.path.dirname(__file__) + '/rules/*.snake')
rules = {_os.path.splitext(_os.path.basename(rule_path))[0]: rule_path
         for rule_path in all_rules}
