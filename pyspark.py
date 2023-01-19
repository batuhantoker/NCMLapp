import os
import shutil
import jdk
if shutil.which("java"):
    print("Java is installed on the system")
else:
    print('not installed')
    jdk.install('9')

os.system("java -version")

#os.path.realpath(shutil.which("java"))