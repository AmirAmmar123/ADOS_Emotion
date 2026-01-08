from operator import ge
import openpyxl
from openpyxl.reader.excel import load_workbook
from openpyxl.workbook import workbook
from openpyxl import Workbook
from Files import *
import os.path
import csv
from pickle import load
#Written by Omer Lerman
def append_write(file,data):
    if not os.path.isfile(file):
        titles = ['Type','Start','End']
        wb = Workbook()
        ws=wb.active
        ws.append(titles)
        wb.save(file)
        wb.close()
    wb = load_workbook(file)
    ws = wb.active
    for row in data:
        ws.append(row)
    wb.save(file)
    wb.close()
def colnum_string(n):
    string = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        string = chr(65 + remainder) + string
    return string
def convert_to_xlsx(file):
    wb = Workbook()
    ws = wb.active
    with open(file, 'r') as f:
        for row in csv.reader(f):
            ws.append(row)
def read_file_line_to_dict(file):
    if not file:
        return None
    if not os.path.isfile(file):
        print("Not a file!")
        return None
    if(file.split(".")[-1]=='pickle'):
        with open(file, "rb") as input_file:
            pic = load(input_file)
            if(type(pic) != type(dict())):
                print('ERROR, pickle file is not a dict')
                return None
            return pic

    if file.split(".")[-1]=='csv':
        print("convert to xlsx")
        wb = Workbook()
        ws = wb.active
        with open(file, 'r') as f:
            for row in csv.reader(f):
                ws.append(row)
    else:
        wb = load_workbook(file)
        ws = wb.active
    first_row = ord('A')
    excel_dict = dict()
    keys = list()
    row = 1
    
    for i in range(0,len(list(ws.columns))):
        # print(colnum_string(i+1)+str(row),colnum_string(len(list(ws.columns))))
        if  ws[colnum_string(i+1)+str(row)].value is None:
            # print("end at "+colnum_string(i+1)+str(row))
            break
        key=ws[colnum_string(i+1)+str(row)].value
        keys.append(key)
        excel_dict[key]=dict()
    for row in range(2,len(list(ws.rows))+1):
        for i,key in enumerate(keys):
            cell=colnum_string(i+1)+str(row)
            excel_dict[key][row] = ws[cell].value
    return {'keys':keys,'first_row':2,'last_row':row,'excel_dict':excel_dict}
def readToPickle():
    save_pickle_file(read_file_line_to_dict(get_file("select excel file to pickle")))
# file = get_file()
# if file.split(".")[-1]=='csv':
#     print("convert to xlsx")
#     wb = Workbook()
#     ws = wb.active
#     with open(file, 'r') as f:
#         for row in csv.reader(f):
#             ws.append(row)
# else :
#     wb = load_workbook(file)
#     ws = wb.active
# print(ws['A2'].value)
# readToPickle()