from Files import *
from Excel import *

dict = read_file_line_to_dict(get_file("Open excel file"))
print(dict['excel_dict'])