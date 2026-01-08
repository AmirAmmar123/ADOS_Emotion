import os
import shutil
def check_create(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
def check_delete(dir):
    if os.path.exists(dir):
        if os.path.isdir(dir):
            shutil.rmtree(dir)
        if os.path.isfile(dir):
            os.remove(dir)