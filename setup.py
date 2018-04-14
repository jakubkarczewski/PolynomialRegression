import sys
import shutil
import os

shebang = '#! ' + str(sys.executable) + '\n'
from_file = open("polynomial.py") 
line = from_file.readline()

to_file = open("polynomial.py",mode="w")
to_file.write(shebang)
shutil.copyfileobj(from_file, to_file)

os.system('mv polynomial.py polynomial')
os.system('chmod +x polynomial')
