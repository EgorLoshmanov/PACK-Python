import os

for root, dirs, files in os.walk('.'):
    for i in files:
        print(os.path.abspath(os.path.join(root, i)))