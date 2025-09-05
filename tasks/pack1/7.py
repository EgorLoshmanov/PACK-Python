from os import *

for root, dirs, files in walk('.'):
    for i in files:
        print(path.abspath(path.join(root, i)))