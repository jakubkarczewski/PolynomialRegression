import sys
import os

shebang = '#! ' + str(sys.executable) + '\n'
with open('polynomial.py', 'r+') as f:
    file_data = f.read()
    f.seek(0, 0)
    f.write(shebang.rstrip('\r\n') + '\n' + file_data)

os.system('mv polynomial.py polynomial')
os.system('chmod +x polynomial')
