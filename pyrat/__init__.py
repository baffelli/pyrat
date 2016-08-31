import glob
import os

# This loads all rules files to the package for them to be accessible
all_rules = glob.glob(os.path.dirname(__file__) + '/rules/*.snake')
rules = {os.path.splitext(os.path.basename(rule_path))[0]: rule_path
         for rule_path in all_rules}
