import threading
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import os
import sys
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class GUI_IMOPD_IKNL_tool():
    def __init__(self, master=None):
        super().__init__()
        self.master = master
        self.with_categorical = False
        self.with_numerical = False
        # add welcome message
        self.welcome_label = tk.Label(self.master, text="Welcome to the Interactive Process Pattern Discovery Tool")
        self.welcome_label.pack(pady=5)
        self.welcome_label.config(font=("Courier", 15))

        # add file label and button
        self.file_label = tk.Label(self.master, text="No file selected", borderwidth=1, relief="solid",
                              width=50, background="white", anchor="w", padx=5, pady=5, justify="left")
        self.file_label.pack(side=tk.TOP, padx=10, pady=10)
        self.select_file_button = tk.Button(self.master, text="Select file", command=self.select_file)
        self.select_file_button.pack(side=tk.TOP, padx=10, pady=10)

        # create a frame for comboboxes
        self.setting_frame = tk.Frame(self.master)
        self.setting_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # create combobox for selecting the case id column
        self.case_id_label = tk.Label(self.setting_frame, text="case id column: ")
        self.case_id_label.pack(side=tk.LEFT, padx=10, pady=10)
        self.case_id_combobox = ttk.Combobox(self.setting_frame, state="readonly")
        self.case_id_combobox.pack(side=tk.LEFT, padx=10, pady=10)

        # create combobox for selecting the activity column
        self.activity_label = tk.Label(self.setting_frame, text="activity column: ")
        self.activity_label.pack(side=tk.LEFT, padx=10, pady=10)
        self.activity_combobox = ttk.Combobox(self.setting_frame, state="readonly")
        self.activity_combobox.pack(side=tk.LEFT, padx=10, pady=10)

        # create combobox for selecting the timestamp column
        self.timestamp_label = tk.Label(self.setting_frame, text="timestamp column: ")
        self.timestamp_label.pack(side=tk.LEFT, padx=10, pady=10)
        self.timestamp_combobox = ttk.Combobox(self.setting_frame, state="readonly")
        self.timestamp_combobox.pack(side=tk.LEFT, padx=10, pady=10)

        # create combobox for selecting the outcome column
        self.outcome_label = tk.Label(self.setting_frame, text="outcome column: ")
        self.outcome_label.pack(side=tk.LEFT, padx=10, pady=10)
        self.outcome_combobox = ttk.Combobox(self.setting_frame, state="readonly")
        self.outcome_combobox.pack(side=tk.LEFT, padx=10, pady=10)

        # creat a menu with checkbuttons for selecting the categorical attributes
        self.menubutton = tk.Menubutton(self.master, text="Choose categorical attributes",
                                   indicatoron=True, borderwidth=1, relief="raised")
        self.menu = tk.Menu(self.menubutton, tearoff=False)
        self.menubutton.configure(menu=self.menu)
        self.menubutton.pack(side=tk.LEFT, padx=10, pady=10)

        # creat a menu with checkbuttons for selecting the numerical attributes
        self.menubutton_num = tk.Menubutton(self.master, text="Choose numerical attributes",
                                   indicatoron=True, borderwidth=1, relief="raised")
        self.menu_num = tk.Menu(self.menubutton_num, tearoff=False)
        self.menubutton_num.configure(menu=self.menu_num)
        self.menubutton_num.pack(side=tk.RIGHT, padx=10, pady=10)

        # create a button for starting the detection
        self.save_setting_button = tk.Button(self.master, text="Save Setting", command=self.save_setting)
        self.save_setting_button.pack(side=tk.TOP, padx=10, pady=10)
        # unable the button until all comboboxes are filled
        self.save_setting_button.config(state="disabled")

    def select_file(self):
        # get file path for only csv files
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")],
                                               title="Select an event log (.cvs format)")

        # check if filepath exists and in csv format
        if file_path and file_path.endswith('.csv'):
            # update the file label
            self.file_label.config(text=file_path)
            self.df = pd.read_csv(file_path)
            self.show_setting(self.df)
        else:
            messagebox.showerror("Error", "Please select a csv file")

    def show_setting(self, df):
        # add column names to the comboboxes
        self.case_id_combobox.config(values=list(df.columns))
        self.activity_combobox.config(values=list(df.columns))
        self.timestamp_combobox.config(values=list(df.columns))
        self.outcome_combobox.config(values=list(df.columns))
        self.choices = {}
        for choice in list(df.columns):
            self.choices[choice] = tk.IntVar(value=0)
            self.menu.add_checkbutton(label=choice, variable=self.choices[choice],
                                 onvalue=1, offvalue=0, indicatoron=True)

        self.choices_num = {}
        for choice in list(df.columns):
            self.choices_num[choice] = tk.IntVar(value=0)
            self.menu_num.add_checkbutton(label=choice, variable=self.choices_num[choice],
                                 onvalue=1, offvalue=0, indicatoron=True)

        # enable the button for starting the detection
        self.save_setting_button.config(state="normal")

    def save_setting(self):
        # get the selected values from the comboboxes
        self.case_id = self.case_id_combobox.get()
        self.activity = self.activity_combobox.get()
        self.timestamp = self.timestamp_combobox.get()
        self.outcome = self.outcome_combobox.get()
        self.categorical_attributes = []
        self.numerical_attributes = []
        for name, var in self.choices.items():
            if var.get() == 1:
                self.categorical_attributes.append(name)
        for name, var in self.choices_num.items():
            if var.get() == 1:
                self.numerical_attributes.append(name)

        # check if all comboboxes are filled
        if self.case_id and self.activity and self.timestamp and self.outcome:
            # create a button for starting the detection
            self.start_detection_button = tk.Button(self.master, text="Start Pattern Discovery",
                                                    command=self.start_detection)

            self.start_detection_button.pack(side=tk.BOTTOM, padx=10, pady=10)
        else:
            messagebox.showerror("Error", "Please fill all the combo boxes")

    def start_detection(self):
        # create a new window for showing the results
        self.result_window = tk.Toplevel(self.master)
        self.result_window.title("Results")
        self.result_window.geometry("900x700")
        self.result_window.resizable(False, False)
        self.result_window.grab_set()
        self.result_window.focus_set()
        self.result_window.transient(self.master)

        # add text holder for recieving input from user
        self.text_holder = tk.Text(self.result_window, height=1, width=50)
        self.text_holder.pack(side=tk.TOP, padx=10, pady=10)

        # add button for getting input from user
        self.get_input_button = tk.Button(self.result_window, text="Pattern Extension", command=self.extension)
        self.get_input_button.pack(side=tk.TOP, padx=10, pady=10)

        # add a text box for showing the results vertically in left side of the window
        self.result_text = tk.Text(self.result_window, height=30, width=10)
        self.result_text.pack(side=tk.LEFT, padx=10, pady=10)

        # add vertical scrollbar for the text box
        self.result_text_scrollbar = tk.Scrollbar(self.result_window, orient=tk.VERTICAL)
        self.result_text_scrollbar.config(command=self.result_text.yview)
        self.result_text_scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.result_text.config(yscrollcommand=self.result_text_scrollbar.set)

        # add horizontal scrollbar for the text box
        self.result_text_scrollbar_h = tk.Scrollbar(self.result_window, orient=tk.HORIZONTAL)
        self.result_text_scrollbar_h.config(command=self.result_text.xview)
        self.result_text_scrollbar_h.pack(side=tk.BOTTOM, fill=tk.X)
        self.result_text.config(xscrollcommand=self.result_text_scrollbar_h.set)

        # create a frame for showing the results
        self.result_frame = tk.Frame(self.result_window)
        self.result_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # create a scrollbar for the result frame
        self.result_scrollbar = tk.Scrollbar(self.result_frame)
        self.result_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # create a picture canvas for showing the results
        self.result_canvas = tk.Canvas(self.result_frame, yscrollcommand=self.result_scrollbar.set)
        self.result_canvas.pack(fill=tk.BOTH, expand=True)

        # configure the scrollbar
        self.result_scrollbar.config(command=self.result_canvas.yview)

        # check if any categorical and numerical attributes are selected
        if len(self.categorical_attributes) > 0 :
            self.with_categorical = True
        if len(self.numerical_attributes) > 0 :
            self.with_numerical = True

        # add a frame for showing the more results beside canvas
        self.table_result_frame = tk.Frame(self.result_window)
        self.table_result_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # add tree widget for showing the results
        self.tree = ttk.Treeview(self.table_result_frame, columns=("pattern", "case coverage",
                                                                   "pattern frequnecy", "outcome correlation",
                                                                   "Case distance"), show="headings", height=20)
        self.tree.column("pattern", width=100, anchor=tk.CENTER)
        self.tree.column("case coverage", width=100, anchor=tk.CENTER)
        self.tree.column("pattern frequnecy", width=100, anchor=tk.CENTER)
        self.tree.column("outcome correlation", width=100, anchor=tk.CENTER)
        self.tree.column("Case distance", width=100, anchor=tk.CENTER)
        self.tree.heading("pattern", text="Pattern")
        self.tree.heading("case coverage", text="Case Coverage")
        self.tree.heading("pattern frequnecy", text="Pattern Frequency")
        self.tree.heading("outcome correlation", text="Outcome Correlation")
        self.tree.heading("Case distance", text="Case Distance")
        self.tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # add vertical scrollbar for the tree widget
        self.tree_scrollbar = tk.Scrollbar(self.table_result_frame, orient=tk.VERTICAL)
        self.tree_scrollbar.config(command=self.tree.yview)
        self.tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.config(yscrollcommand=self.tree_scrollbar.set)

        # create a new thread for running the detection
        self.thread = threading.Thread(target=self.run_detection)
        self.thread.start()

    def run_detection(self):
        if self.with_numerical:
            self.patient_data = self.df[self.numerical_attributes]
        if self.with_categorical:
            self.patient_data = self.df[self.categorical_attributes]

        self.patient_data[self.case_id] = self.df[self.case_id]
        self.patient_data[self.outcome] = self.df[self.outcome]
        self.patient_data = self.patient_data.drop_duplicates(subset=[self.case_id], keep='first')

        self.patient_data.sort_values(by=self.case_id, inplace=True)
        self.patient_data.reset_index(inplace=True, drop=True)
        # create a new thread for running the detection
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        z = [1, 2, 3, 4, 5]
        ax.scatter(x, y, z)

        canvas = FigureCanvasTkAgg(fig, master=self.result_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # insert the unique activities in dataframe to the text box for showing the results i none column
        for activity in self.df[self.activity].unique():
            self.result_text.insert(tk.END, activity + "\n")


    def extension(self):
        # get the input from the user
        Core_activity = self.text_holder.get("1.0", tk.END)

        Core_activity = Core_activity[:-1]

        # check if the input is not empty
        if Core_activity == "":
            messagebox.showerror("Error", "Please enter a core activity")

        # check if the input is not empty
        elif Core_activity not in self.df[self.activity].unique():
            messagebox.showerror("Error", "Please enter a valid core activity")

        # if the input is valid
        else:
            # create a new windows for the results of extension
            self.extension_window = tk.Toplevel(self.master)
            self.extension_window.title("Extension Results")
            self.extension_window.geometry("900x700")
            self.extension_window.resizable(False, False)
            self.extension_window.grab_set()
            self.extension_window.focus_set()
            # add multiple tab for showing the results
            self.tab_control = ttk.Notebook(self.extension_window)
            self.tab_control.pack(expand=1, fill="both")
            # add a tab for showing the results of extension
            self.tab1 = ttk.Frame(self.tab_control)
            self.tab_control.add(self.tab1, text="Extension_1")
            # add a tab for showing the results of extension
            self.tab2 = ttk.Frame(self.tab_control)
            self.tab_control.add(self.tab2, text="Extension_2")
            # add a tab for showing the results of extension
            self.tab3 = ttk.Frame(self.tab_control)
            self.tab_control.add(self.tab3, text="Extension_3")


root = tk.Tk()
app = GUI_IMOPD_IKNL_tool(master=root)
app.master.mainloop()