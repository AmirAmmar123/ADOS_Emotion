import re
import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog
import tkinter

from pandastable import Table, TableModel,config

class TestApp():
        """Basic test frame for the table"""
        def __init__(self,list_of_results, parent=None):
            root = tkinter.Tk()
            print(len(list_of_results))
            if type(list_of_results) is not list:
                list_of_results = [list_of_results]
            self._df_list = list_of_results
            self._current = 0 
            #old table type support:
            root.title(list_of_results[0][0])

            frame = tk.Frame(root)
            self._root = root
            self._frame = frame
            frame.pack(fill='both', expand=True)

            pt = Table(frame,dataframe=list_of_results[0][1],showstatusbar=True,showtoolbar=True)
            self._table = pt
            pt.showIndex()
            pt.show()
            nbutton = tk.Button(root, text='Next Table', command=self.load_next_table)
            nbutton.pack()
            pbutton = tk.Button(root, text='Prev Table', command=self.load_prev_table)
            pbutton.pack()
            root.mainloop()
        def load_next_table(self):
            self._current+=1
            self._root.title(self._df_list[self._current%len(self._df_list)][0])
            self._table.model.df = self._df_list[self._current%len(self._df_list)][1]
            self._table.redraw()
        def load_prev_table(self):
            self._current-=1
            self._table.model.df = self._df_list[self._current%len(self._df_list)][1]
            self._root.title(self._df_list[self._current%len(self._df_list)][0])
            self._table.redraw()
def get_string(title='please enter your input'):
    application_window = tk.Tk()

    answer = simpledialog.askstring("Input", title,
                                    parent=application_window)
    return answer


def load_dataFrame(df):
    app = TestApp(data=df)
    #launch the app
    app.mainloop()