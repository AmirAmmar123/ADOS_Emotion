import tkinter as tk
from tkinter import filedialog
import time
from datetime import datetime
from pickle import dump,load
from os import listdir, rename
#Written by Omer Lerman
def get_file(title = "Please select a file:"):
    root = tk.Tk()
    root.withdraw()

    # Make it almost invisible - no decorations, 0 size, top left corner.
    root.overrideredirect(True)
    root.geometry('0x0+0+0')

    # Show window again and lift it to top so it can get focus,
    # otherwise dialogs will end up behind the terminal.
    root.deiconify()
    root.lift()
    root.focus_force()
    xl_file_path = filedialog.askopenfilename(title=title,parent=root)
    root.destroy()
    return xl_file_path
def get_files(title = "Please your files:"):
    root = tk.Tk()
    root.withdraw()

    # Make it almost invisible - no decorations, 0 size, top left corner.
    root.overrideredirect(True)
    root.geometry('0x0+0+0')

    # Show window again and lift it to top so it can get focus,
    # otherwise dialogs will end up behind the terminal.
    root.deiconify()
    root.lift()
    root.focus_force()
    files = filedialog.askopenfilenames(title=title,parent=root)
    root.destroy()
    return files
def get_folder(title= "Please select a Directory:"):
    root = tk.Tk()
    root.withdraw()
    root.focus_force()
    folder_path = filedialog.askdirectory(title= title)
    return folder_path   
def write_list_to_file(list,file):
    with open(file, 'w') as f:
        for item in list:
            f.write("%s\n" % item)
def now_file_format():
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H-%M-%S")
    return dt_string
def save_pickle_file(to_pickle,title = "Please save your file",ext = 'pickle',where = None):
    if where:
        f = open(str(where), 'wb')
        dump(to_pickle, f)
        f.close()
        return
    root = tk.Tk()
    root.withdraw()
    root.focus_force()
    out_file = filedialog.asksaveasfile(mode='w', defaultextension=ext,title=title)
    print('saving to ',out_file.name," ...")
    f = open(str(out_file.name), 'wb')
    dump(to_pickle, f)
    f.close()
    if out_file is None:
        return
    return out_file
def save_get_file(title = "Please save your file",ext =None):
    root = tk.Tk()
    root.withdraw()
    root.focus_force()
    out_file = filedialog.asksaveasfile(mode='w', defaultextension=ext,title=title)
    if out_file is None:
        return None
    return out_file.name
def rename_pickle_files():
    folder = get_folder()
    for f in listdir(folder):
        old_name = f.split(" ")
        new_name = old_name[2]+" "+old_name[1]+" "+old_name[0]
        rename(folder+"/"+f,folder+"/"+new_name)
def load_pickle(file = None):
    if not file:
        file = get_file()
        val = None
    with open(file,'rb') as f:
            val = load(f)
    if not val : 
        print("ERROR with pickle load")
        exit(1)
    return val
# rename_pickle_files()
