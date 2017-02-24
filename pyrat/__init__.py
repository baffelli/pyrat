import glob as _glob
import os as _os
import fnmatch

# This loads all rules files to the package for them to be accessible
all_rules = _glob.glob(_os.path.dirname(__file__) + '/rules/*.snake')
rules = {_os.path.splitext(_os.path.basename(rule_path))[0]: rule_path
         for rule_path in all_rules}

#Construct wrappers
wrappers_base = _os.path.dirname(__file__) + '/rules/wrappers/'

wrappers = {}
for filenames in _glob.glob(wrappers_base + '/*/*[!\.].py'):
    dirname = _os.path.basename(_os.path.normpath(_os.path.dirname(filenames)))
    rulename = _os.path.splitext(_os.path.basename(filenames))[0]
    new_dict = {rulename: filenames}
    if dirname in wrappers:
        wrappers[dirname].update(new_dict)
    else:
        wrappers[dirname] = new_dict


