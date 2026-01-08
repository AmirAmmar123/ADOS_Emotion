#!/usr/bin/env python3

import os
import zipfile
import re
import sys
import datetime
from Files import get_folder



NAME = str(get_folder())
HW_NUM = str(2)

wet_hw_path = os.path.dirname(NAME) + "\\" + HW_NUM + "\\wet_hw"
dry_hw_path = os.path.dirname(NAME) + "\\" + HW_NUM + "\\dry_hw"

os.mkdir(os.path.dirname(NAME) + "\\" + HW_NUM)
os.mkdir(wet_hw_path)
os.mkdir(dry_hw_path)

ids = []
total_ids = []

with zipfile.ZipFile(NAME) as main_zip:
	for hw_zip_name in main_zip.namelist():
		hw_zip_basename = os.path.basename(hw_zip_name)

		with main_zip.open(hw_zip_name) as temp:
			try:
				with zipfile.ZipFile(temp) as hw_zip:
					real_name = os.path.splitext(hw_zip_basename)[0].removeprefix(HW_NUM + "_")
					ids = [int(s) for s in hw_zip_basename.split('_file_')[-1].split('_') if s.isdigit()]
					ids = [str(s) for s in ids]
					
					if len(ids) == 0:
						os.mkdir(os.path.join(wet_hw_path, real_name))
						print("[-] WARNING! file " + real_name + " was not added to the list")
					else:
						total_ids.extend(ids)
						if len(ids) == 1:
							total_ids.append("\n")
						real_name = '_'.join(ids)
						os.mkdir(os.path.join(wet_hw_path, real_name))
					
					for fileName in hw_zip.namelist():
						if fileName.endswith('.cpp'):
							hw_zip.extract(fileName, os.path.join(wet_hw_path, real_name))

						if fileName.endswith('.h'):
							hw_zip.extract(fileName, os.path.join(wet_hw_path, real_name))

						if fileName.endswith('.txt'):
							hw_zip.extract(fileName, os.path.join(wet_hw_path, real_name))
							
						if fileName.endswith('.pdf'):
							dry = hw_zip.getinfo(fileName)
							dry.filename = f'{real_name}.pdf'
							try:
								hw_zip.extract(dry, dry_hw_path)
							except:
								print("error in " + real_name + " PDF file")
			except:
				print("[-] ERROR! cant extract " + hw_zip_basename) 
				
with open(os.path.dirname(NAME) + '\\ids.txt', 'w') as f:
    f.write('\n'.join(total_ids))

print("[+] DONE! output in " + os.path.dirname(NAME))
