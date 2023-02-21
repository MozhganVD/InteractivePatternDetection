import tkinter as tk
from tkinter import *
import tkinter.filedialog as filedialog
from tkinter import ttk

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

root = tk.Tk()
root.title("Settings")
# set window size fixed
root.resizable(0, 0)
# set window size
root.geometry("900x600")


# add welcome message
welcome_label = tk.Label(root, text="Welcome to the Interactive Process Pattern Discovery Tool")
welcome_label.pack(pady=5)
welcome_label.config(font=("Courier", 15))


def select_file():
    global column_frame
    # get file path for only csv files
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")],
                                           title="Select an event log (.cvs format)")
    # check if filepath exists and in csv format
    if file_path and file_path.endswith('.csv'):
        # update the file label
        file_label.config(text=file_path)
        # remove any existing checkbutton frame
        # if column_frame.winfo_exists():
        #     column_frame.destroy()
        # create a new frame to hold the checkbuttons
        # column_frame = tk.Frame(column_canvas)
        # column_canvas.create_window((0, 0), window=column_frame, anchor='nw')
        # read in the CSV file
        df = pd.read_csv(file_path)
        show_columns(df)
        # update the scroll region of the canvas
        # column_canvas.update_idletasks()
        # column_canvas.config(scrollregion=column_canvas.bbox('all'))

def show_columns(df):
    # create a canvas for the column frame
    column_canvas = tk.Canvas(root, borderwidth=1, highlightthickness=0, width=800)
    column_canvas.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
    # Create comboboxes
    combo1 = ttk.Combobox(column_canvas, state="readonly", values=list(df.columns))
    combo2 = ttk.Combobox(column_canvas, state="readonly", values=list(df.columns))
    combo3 = ttk.Combobox(column_canvas, state="readonly", values=list(df.columns))
    combo4 = ttk.Combobox(column_canvas, state="readonly", values=list(df.columns))

    # Add comboboxes to window
    combo1.pack(side=tk.LEFT, padx=10, pady=10)
    combo2.pack(side=tk.LEFT, padx=10, pady=10)
    combo3.pack(side=tk.LEFT, padx=10, pady=10)
    combo4.pack(side=tk.LEFT, padx=10, pady=10)

    # set a value for hint text
    combo1.set("select Case ID")
    combo2.set("select Activity")
    combo3.set("select Timestamp")
    combo4.set("select Process Outcome")

    #make it manadatroy to select a value from combobox
    combo1.config(validate="key")
    combo2.config(validate="key")
    combo3.config(validate="key")
    combo4.config(validate="key")

    # create checkbuttons for each column
    # create a canvas for the column frame
    column_canvas2 = tk.Canvas(root, borderwidth=1, highlightthickness=0, width=800)
    column_canvas2.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
    columns = list(df.columns)
    # columns.remove(combo1.get())
    # columns.remove(combo2.get())
    # columns.remove(combo3.get())
    # columns.remove(combo4.get())
    for col in columns:
        var = tk.IntVar()
        c = tk.Checkbutton(column_canvas2, text=col, variable=var)
        c.pack(side=tk.LEFT, padx=10, pady=10)


file_select_button = tk.Button(root, text="Select EventLog", command=select_file)
file_select_button.pack(pady=5)

file_label = tk.Label(root, text="No file selected", borderwidth=1, relief="solid",
                      width=50, background="white", anchor="w", padx=5, pady=5, justify="left")
file_label.pack(pady=5)


#
# # add a horizontal scrollbar
# column_scrollbar = tk.Scrollbar(root, orient=tk.HORIZONTAL, command=column_canvas.xview)
# column_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
# column_canvas.configure(xscrollcommand=column_scrollbar.set)

root.mainloop()
