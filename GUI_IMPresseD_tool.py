import pickle
import random
import threading
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import os
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from paretoset import paretoset
import pyperclip
from Auto_IMPID import AutoStepWise_PPD
from IMIPD import VariantSelection, create_pattern_attributes, calculate_pairwise_case_distance, Trace_graph_generator, \
    Pattern_extension, plot_patterns, Single_Pattern_Extender, plot_dashboard


class GUI_IMOPD_IKNL_tool:
    def __init__(self, master=None):
        super().__init__()
        self.numerical_attributes = []
        self.categorical_attributes = []
        self.master = master
        self.with_categorical = False
        self.with_numerical = False
        # add welcome message
        self.welcome_label = tk.Label(self.master, text="Welcome to IMPresseD")
        self.welcome_label.pack(pady=5)
        self.welcome_label.config(font=("Courier", 15))

        # add file label and button
        frame_file = tk.Frame(self.master)
        frame_file.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5, expand=True)
        self.file_label = tk.Label(frame_file, text="No file selected", borderwidth=0.5, relief="solid",
                                   width=100, background="white", anchor="w", padx=5, pady=5, justify="left")
        self.file_label.pack(side=tk.LEFT, padx=10, pady=5, expand=True)
        self.select_file_button = tk.Button(frame_file, text="Select file", command=self.select_file)
        self.select_file_button.pack(side=tk.LEFT, padx=10, pady=5)

        # create a frame for comboboxes
        self.setting_frame = tk.Frame(self.master)
        self.setting_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # create combobox for selecting the case id column
        self.case_id_label = tk.Label(self.setting_frame, text="Case ID column: ")
        self.case_id_label.pack(side=tk.LEFT, padx=10, pady=5)
        self.case_id_combobox = ttk.Combobox(self.setting_frame, state="readonly")
        self.case_id_combobox.pack(side=tk.LEFT, padx=10, pady=5)

        # create combobox for selecting the activity column
        self.activity_label = tk.Label(self.setting_frame, text="Activity column: ")
        self.activity_label.pack(side=tk.LEFT, padx=10, pady=5)
        self.activity_combobox = ttk.Combobox(self.setting_frame, state="readonly")
        self.activity_combobox.pack(side=tk.LEFT, padx=10, pady=5)

        # create frame for outcome settings
        outcome_frame = tk.Frame(self.master)
        outcome_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        self.outcome_label = tk.Label(outcome_frame, text="Outcome column: ")
        self.outcome_label.pack(side=tk.LEFT, padx=10, pady=5)
        self.outcome_combobox = ttk.Combobox(outcome_frame, state="readonly")
        self.outcome_combobox.pack(side=tk.LEFT, padx=10, pady=5)
        self.outcome_type_label = tk.Label(outcome_frame, text="Outcome type: ")
        self.outcome_type_label.pack(side=tk.LEFT, padx=10, pady=5)
        self.outcome_type_combobox = ttk.Combobox(outcome_frame, state="readonly")
        self.outcome_type_combobox.pack(side=tk.LEFT, padx=10, pady=5)

        # create a frame for time settings
        time_frame = tk.Frame(self.master)
        time_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        self.timestamp_label = tk.Label(time_frame, text="Timestamp column: ")
        self.timestamp_label.pack(side=tk.LEFT, padx=10, pady=5)
        self.timestamp_combobox = ttk.Combobox(time_frame, state="readonly")
        self.timestamp_combobox.pack(side=tk.LEFT, padx=10, pady=5)
        # create a entry widget for getting delta time
        self.delta_time_label = tk.Label(time_frame, text="Delta time (in second): ")
        self.delta_time_label.pack(side=tk.LEFT, padx=10, pady=5)
        self.delta_time_entry = tk.Entry(time_frame)
        self.delta_time_entry.pack(side=tk.LEFT, padx=10, pady=5)
        # create an entry widget for getting the maximum gap between events
        eventual_frame = tk.Frame(self.master)
        eventual_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        eventual_label = tk.Label(eventual_frame, text="Max gap between events: ")
        eventual_label.pack(side=tk.LEFT, padx=10, pady=5)
        self.eventual_holder = tk.Entry(eventual_frame, width=5)
        self.eventual_holder.pack(side=tk.LEFT, padx=10, pady=5)

        # create a frame for buttons
        button_frame = tk.Frame(self.master)
        button_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        # create a button for selecting the categorical attributes
        self.select_categorical_button = tk.Button(button_frame, text="Categorical attributes",
                                                   command=self.select_categorical, state=tk.DISABLED)
        self.select_categorical_button.pack(side=tk.LEFT, padx=10, pady=5)
        # create a button for selecting the numerical attributes
        self.select_numerical_button = tk.Button(button_frame, text="Numerical attributes",
                                                 command=self.select_numerical, state=tk.DISABLED)
        self.select_numerical_button.pack(side=tk.LEFT, padx=10, pady=5)

        # create a button for starting the detection
        self.save_setting_button = tk.Button(button_frame, text="Save Attributes", command=self.save_setting)
        self.save_setting_button.pack(side=tk.LEFT, padx=10, pady=5)
        self.save_setting_button.config(state=tk.DISABLED)

        # create a button for starting the detection
        self.start_detection_button = tk.Button(self.master, text="Interactive Pattern Discovery",
                                                command=self.start_detection)
        self.auto_detection_button = tk.Button(self.master, text="Automatic Pattern Discovery",
                                               command=self.Automatic_detection)
        self.interest_function_frame = tk.Frame(self.master)
        self.interest_function_frame_2 = tk.Frame(self.master)
        self.interest_function_frame_3 = tk.Frame(self.master)
        self.visualization_frame = tk.Frame(self.master)

    def select_categorical(self):
        # creat a window for selecting categorical attributes
        window = tk.Toplevel(self.master)
        window.title("Categorical attributes")
        yscrollbar = tk.Scrollbar(window)
        yscrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        label = tk.Label(window,
                         text="Select the categorical attributes :  ",
                         padx=5, pady=5)
        label.pack(side=tk.TOP)
        self.listbox_cat = tk.Listbox(window, selectmode=tk.MULTIPLE, yscrollcommand=yscrollbar.set)
        self.listbox_cat.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        for item in list(self.df.columns):
            self.listbox_cat.insert(tk.END, item)

        self.listbox_cat.config(bg="white")
        # add a button for saving the selection and closing the window
        button = tk.Button(window, text="Save", command=lambda: self.save_destroy_cat(window))
        button.pack(side=tk.BOTTOM)

    def select_numerical(self):
        window = tk.Toplevel(self.master)
        window.title("Numerical attributes")
        yscrollbar = tk.Scrollbar(window)
        yscrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        label = tk.Label(window,
                         text="Select the numerical attributes :  ",
                         padx=10, pady=5)
        label.pack(side=tk.TOP)
        self.listbox_num = tk.Listbox(window, selectmode=tk.MULTIPLE, yscrollcommand=yscrollbar.set)
        self.listbox_num.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        for item in list(self.df.columns):
            self.listbox_num.insert(tk.END, item)

        self.listbox_num.config(bg="white")
        # add a button for saving the selection and closing the window
        button = tk.Button(window, text="Save", command=lambda: self.save_destroy_num(window))
        button.pack(side=tk.BOTTOM)

    def save_destroy_cat(self, window):
        self.categorical_attributes = []
        for i in self.listbox_cat.curselection():
            self.categorical_attributes.append(self.listbox_cat.get(i))
        window.destroy()

    def save_destroy_num(self, window):
        self.numerical_attributes = []
        for i in self.listbox_num.curselection():
            self.numerical_attributes.append(self.listbox_num.get(i))
        window.destroy()

    def select_file(self):
        # get file path for only csv files
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")],
                                                    title="Select an event log (.cvs format)")

        # check if filepath exists and in csv format
        if self.file_path and self.file_path.endswith('.csv'):
            # update the file label
            self.file_label.config(text=self.file_path)
            self.df = pd.read_csv(self.file_path)

            # Print information about the DataFrame - added by Sabrina
            print("DataFrame Info:")
            print(self.df.info())

            self.show_setting()
        else:
            messagebox.showerror("Error", "Please select a csv file")

    def show_setting(self):
        # clear combo boxes
        self.case_id_combobox.set('')
        self.activity_combobox.set('')
        self.timestamp_combobox.set('')
        self.outcome_combobox.set('')
        self.outcome_type_combobox.set('')
        self.delta_time_entry.delete(0, tk.END)

        # Clear all buttons and text widgets
        self.start_detection_button.destroy()
        self.auto_detection_button.destroy()
        self.interest_function_frame.destroy()
        self.interest_function_frame_2.destroy()
        self.interest_function_frame_3.destroy()
        self.visualization_frame.destroy()

        # add column names to the comboboxes
        Options = list(self.df.columns)
        self.case_id_combobox.config(values=Options)
        self.activity_combobox.config(values=Options)
        self.timestamp_combobox.config(values=Options)
        self.outcome_combobox.config(values=Options)
        self.outcome_type_combobox.config(values=['binary', 'numerical'])
        self.select_categorical_button.config(state="normal")
        self.select_numerical_button.config(state="normal")

        # enable the button for starting the detection
        self.save_setting_button.config(state="normal")

    def save_setting(self):
        # create a dictionary for saving all the extended patterns
        self.all_extended_patterns = dict()
        # get the selected values from the comboboxes
        self.case_id = self.case_id_combobox.get()
        self.activity = self.activity_combobox.get()
        self.timestamp = self.timestamp_combobox.get()
        self.outcome = self.outcome_combobox.get()
        self.outcome_type = self.outcome_type_combobox.get()
        self.delta_time = self.delta_time_entry.get()
        self.Max_gap_between_events = self.eventual_holder.get()
        self.Max_gap_between_events = int(self.Max_gap_between_events)

        # check if all comboboxes are filled
        if self.case_id and self.activity and self.timestamp and self.outcome and self.delta_time\
                and self.outcome_type and self.Max_gap_between_events:
            # set the right format for the dataframe columns
            self.df[self.activity] = self.df[self.activity].str.replace("_", "-")
            self.df[self.timestamp] = pd.to_datetime(self.df[self.timestamp])
            self.df[self.case_id] = self.df[self.case_id].astype(str)
            self.df[self.case_id] = self.df[self.case_id].astype(str)

            color_codes = ["#" + ''.join([random.choice('0001123456789ABCDEFABC') for i in range(6)])
                           for j in range(len(self.df[self.activity].unique()))]

            self.color_act_dict = dict()
            counter = 0
            for act in self.df[self.activity].unique():
                self.color_act_dict[act] = color_codes[counter]
                counter += 1
            self.color_act_dict['start'] = 'k'
            self.color_act_dict['end'] = 'k'

            # create a button for starting the detection
            self.auto_detection_button = tk.Button(self.master, text="Automatic Pattern Discovery",
                                                   command=self.Automatic_detection)
            self.auto_detection_button.pack(side=tk.BOTTOM, padx=10, pady=5)

            self.start_detection_button = tk.Button(self.master, text="Interactive Pattern Discovery",
                                                    command=self.start_detection)
            self.start_detection_button.pack(side=tk.BOTTOM, padx=10, pady=5)

            # create a checkbox and combo box for three interest functions
            self.interest_function_frame = tk.Frame(self.master)
            self.interest_function_frame.pack(side=tk.BOTTOM, padx=10, pady=5)
            self.correlation_function = tk.IntVar()
            self.interest_function_checkbox_1 = tk.Checkbutton(self.interest_function_frame,
                                                               text="Correlation interest function",
                                                               variable=self.correlation_function)
            self.interest_function_checkbox_1.pack(side=tk.LEFT, padx=10, pady=5)
            self.direction_correlation_function = tk.StringVar()
            self.direction_combobox_1 = ttk.Combobox(self.interest_function_frame,
                                                     textvariable=self.direction_correlation_function, state="readonly")
            self.direction_combobox_1['values'] = ("Max", "Min")
            self.direction_combobox_1.pack(side=tk.RIGHT, padx=10, pady=5)
            self.direction_combobox_1.current(0)

            self.interest_function_frame_2 = tk.Frame(self.master)
            self.interest_function_frame_2.pack(side=tk.BOTTOM, padx=10, pady=5)
            self.frequency_function = tk.IntVar()
            self.interest_function_checkbox_2 = tk.Checkbutton(self.interest_function_frame_2,
                                                               text="Frequency interest function",
                                                               variable=self.frequency_function)
            self.interest_function_checkbox_2.pack(side=tk.LEFT, padx=10, pady=5)
            self.direction_frequency_function = tk.StringVar()
            self.direction_combobox_2 = ttk.Combobox(self.interest_function_frame_2,
                                                     textvariable=self.direction_frequency_function, state="readonly")
            self.direction_combobox_2['values'] = ("Max", "Min")
            self.direction_combobox_2.pack(side=tk.RIGHT, padx=10, pady=5)
            self.direction_combobox_2.current(0)

            self.interest_function_frame_3 = tk.Frame(self.master)
            self.interest_function_frame_3.pack(side=tk.BOTTOM, padx=10, pady=5)
            self.distance_function = tk.IntVar()
            self.interest_function_checkbox_3 = tk.Checkbutton(self.interest_function_frame_3,
                                                               text="Case Distance interest function",
                                                               variable=self.distance_function)
            self.interest_function_checkbox_3.pack(side=tk.LEFT, padx=10, pady=5)
            self.direction_distance_function = tk.StringVar()
            self.direction_combobox_3 = ttk.Combobox(self.interest_function_frame_3,
                                                     textvariable=self.direction_distance_function, state="readonly")
            self.direction_combobox_3['values'] = ("Max", "Min")
            self.direction_combobox_3.pack(side=tk.RIGHT, padx=10, pady=5)
            self.direction_combobox_3.current(1)

            # create a frame for visualization options
            self.visualization_frame = tk.Frame(self.master)
            self.visualization_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5, expand=True)
            self.visualization_label = tk.Label(self.visualization_frame, text="Visualization row number (min. 2):")
            self.visualization_label.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=5, expand=True)
            self.visualization_row_entry = tk.Entry(self.visualization_frame, width=10)
            self.visualization_row_entry.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=5, expand=True)
            self.visualization_col_label = tk.Label(self.visualization_frame, text="Visualization column number:")
            self.visualization_col_label.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=5, expand=True)
            self.visualization_col_entry = tk.Entry(self.visualization_frame, width=10)
            self.visualization_col_entry.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=5, expand=True)

        else:
            messagebox.showerror("Error", "You need to select case id, activity, timestamp, outcome,"
                                          " delta time, and max gap between events for eventually relations!")

    def start_detection(self):
        if self.correlation_function.get() == 0 and \
                self.frequency_function.get() == 0 and \
                self.distance_function.get() == 0:
            messagebox.showerror("Error", "You need to select at least one interest function!")
        else:
            # create a new window for showing the results
            self.result_window = tk.Toplevel(self.master)
            self.result_window.title("Results")
            self.result_window.geometry("1000x1000")
            self.result_window.resizable(False, False)
            self.result_window.grab_set()
            self.result_window.focus_set()
            self.result_window.transient(self.master)

            self.progress_bar = ttk.Progressbar(self.result_window, orient=tk.HORIZONTAL, length=900,
                                                mode='indeterminate')
            self.progress_bar.pack(side=tk.TOP, padx=10, pady=5)

            # add text holder for recieving input from user
            self.text_holder = tk.Text(self.result_window, height=1, width=50)
            self.text_holder.pack(side=tk.TOP, padx=10, pady=5)

            # add button for getting input from user
            self.get_input_button = tk.Button(self.result_window, text="Pattern Extension", command=self.extension)
            self.get_input_button.pack(side=tk.TOP, padx=10, pady=5)

            # create a frame for showing the results
            self.result_frame = tk.Frame(self.result_window)
            self.result_frame.pack(side=tk.TOP)
            # create a picture canvas for showing the results
            self.result_canvas = tk.Canvas(self.result_frame)
            self.result_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            # check if any categorical and numerical attributes are selected
            if len(self.categorical_attributes) > 0:
                self.with_categorical = True
            if len(self.numerical_attributes) > 0:
                self.with_numerical = True
            # add a frame for showing the more results beside canvas
            self.table_result_frame = tk.Frame(self.result_window)
            self.table_result_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            # add a message on top the table
            self.table_result_text = tk.Label(self.table_result_frame, text="Right click on your desired pattern to"
                                                                            " put in its name on the clipboard")
            self.table_result_text.pack(side=tk.TOP, padx=10, pady=5)
            # add tree widget for showing the results
            tree = self.creat_table(self.table_result_frame)
            # create a new thread for running the detection
            self.thread = threading.Thread(target=self.run_detection(tree))
            self.thread.start()

    def create_patient_data(self):
        patient_data = pd.DataFrame()
        if self.with_numerical:
            patient_data[self.numerical_attributes] = self.df[self.numerical_attributes]
        if self.with_categorical:
            patient_data[self.categorical_attributes] = self.df[self.categorical_attributes]

        patient_data[self.case_id] = self.df[self.case_id]
        patient_data[self.outcome] = self.df[self.outcome]
        patient_data = patient_data.drop_duplicates(subset=[self.case_id], keep='first')

        patient_data.sort_values(by=self.case_id, inplace=True)
        patient_data.reset_index(inplace=True, drop=True)

        # create feature for each activity
        patient_data[list(self.df[self.activity].unique())] = 0

        return patient_data

    def creat_pairwise_distance(self):
        # calculate the pairwise case distance or load it from the file if it is already calculated
        folder_address = os.path.dirname(self.file_path) + "/"
        distance_files = folder_address + "dist"
        if not os.path.exists(distance_files):
            X_features = self.patient_data.drop([self.case_id, self.outcome], axis=1)
            pairwise_distances_array = calculate_pairwise_case_distance(X_features, self.numerical_attributes)
            os.makedirs(distance_files)
            with open(distance_files + "/pairwise_case_distances.pkl", 'wb') as f:
                pickle.dump(pairwise_distances_array, f)

        else:
            with open(distance_files + "/pairwise_case_distances.pkl", 'rb') as f:
                pairwise_distances_array = pickle.load(f)

        pair_cases = [(a, b) for idx, a in enumerate(self.patient_data.index) for b in
                      self.patient_data.index[idx + 1:]]
        case_size = len(self.patient_data)
        i = 0
        start_search_points = []
        for k in range(case_size):
            start_search_points.append(k * case_size - (i + k))
            i += k

        return pairwise_distances_array, pair_cases, start_search_points

    def run_detection(self, tree):
        # create a progress bar at the bottom of the result window
        self.progress_bar.start(10)

        self.patient_data = self.create_patient_data()
        # variant selection
        selected_variants = VariantSelection(self.df, self.case_id, self.activity, self.timestamp)
        counter = 1
        for case in selected_variants["case:concept:name"].unique():
            counter += 1
            Other_cases = \
                selected_variants.loc[selected_variants["case:concept:name"] == case, 'case:CaseIDs'].tolist()[0]
            trace = self.df.loc[self.df[self.case_id] == case, self.activity].tolist()
            for act in np.unique(trace):
                Number_of_act = trace.count(act)
                for Ocase in Other_cases:
                    self.patient_data.loc[self.patient_data[self.case_id] == Ocase, act] = Number_of_act

        self.pairwise_distances_array, self.pair_cases, self.start_search_points = self.creat_pairwise_distance()

        self.visualization = ['Outcome_Interest', 'Frequency_Interest', 'Case_Distance_Interest']
        # check if the checkbox for interest function is checked
        self.pareto_features = []
        self.pareto_sense = []
        if self.correlation_function.get():
            self.pareto_features.append('Outcome_Interest')
            self.pareto_sense.append(self.direction_correlation_function.get())
        if self.frequency_function.get():
            self.pareto_features.append('Frequency_Interest')
            self.pareto_sense.append(self.direction_frequency_function.get())
        if self.distance_function.get():
            self.pareto_features.append('Case_Distance_Interest')
            self.pareto_sense.append(self.direction_distance_function.get())

        activity_attributes = create_pattern_attributes(self.patient_data, self.outcome,
                                                        list(self.df[self.activity].unique()),
                                                        self.pairwise_distances_array, self.pair_cases,
                                                        self.start_search_points, self.outcome_type)

        Objectives_attributes = activity_attributes[self.pareto_features]
        if 'Outcome_Correlation' in self.pareto_features:
            Objectives_attributes['Outcome_Correlation'] = np.abs(Objectives_attributes['Outcome_Correlation'])

        mask = paretoset(Objectives_attributes, sense=self.pareto_sense)
        paretoset_activities = activity_attributes[mask]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(self.visualization[0])
        ax.set_ylabel(self.visualization[1])
        ax.set_zlabel(self.visualization[2])
        for ticker, row in paretoset_activities.iterrows():
            ax.scatter(row[self.visualization[0]], row[self.visualization[1]], row[self.visualization[2]],
                       c=self.color_act_dict[row['patterns']])
            ax.text(row[self.visualization[0]], row[self.visualization[1]], row[self.visualization[2]],
                    row['patterns'])

        # make progress bar invisible
        self.progress_bar.stop()
        self.progress_bar.pack_forget()

        # create a new thread for running the detection
        canvas = FigureCanvasTkAgg(fig, master=self.result_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # update tree (tabel results) based on the values in pareto_activities
        for index, row in paretoset_activities.iterrows():
            # insert values into the treeview widget
            tree.insert("", "end",
                        values=(row['patterns'], row['Frequency_Interest'],
                                row['Outcome_Interest'], row['Case_Distance_Interest']))

    def creat_table(self, table_result_frame):
        tree = ttk.Treeview(table_result_frame, columns=("patterns",
                                                         "Frequency_Interest", "Outcome_Interest",
                                                         "Case_Distance_Interest"), show="headings", height=20)
        tree.column("patterns", width=100, anchor=tk.CENTER)
        tree.column("Frequency_Interest", width=100, anchor=tk.CENTER)
        tree.column("Outcome_Interest", width=100, anchor=tk.CENTER)
        tree.column("Case_Distance_Interest", width=100, anchor=tk.CENTER)
        tree.heading("patterns", text="Pattern")
        tree.heading("Frequency_Interest", text="Frequency_Interest")
        tree.heading("Outcome_Interest", text="Outcome_Interest")
        tree.heading("Case_Distance_Interest", text="Case_Distance_Interest")
        tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        tree.bind("<Button-3>", lambda event, arg=tree: self.copy_to_clipboard(event, tree))

        return tree

    def extension(self):
        # get the input from the user
        Core_activity = self.text_holder.get("1.0", tk.END)
        Core_activity = Core_activity[:-1]

        # check if the input is not empty
        if Core_activity == "":
            messagebox.showerror("Error", "Please enter a foundational pattern for extension!")

        # check if the input is not empty
        elif Core_activity not in self.df[self.activity].unique():
            messagebox.showerror("Error", "Selected pattern is not valid!")

        # if the input is valid
        else:
            # create a new windows for the results of extension
            extension_window = tk.Toplevel(self.master)
            extension_window.title("Extension Results: %s" % Core_activity)
            extension_window.geometry("1000x900")
            # extension_window.resizable(False, False)
            extension_window.grab_set()
            extension_window.focus_set()
            # add multiple tab for showing the results
            self.tab_control = ttk.Notebook(extension_window)
            self.tab_control.pack(expand=1, fill="both")
            # add a tab for showing the results of extension
            self.tab1 = ttk.Frame(self.tab_control)
            self.tab_control.add(self.tab1, text="Pareto Front")

            # add a progress bar for showing the progress of the extension
            self.progress_bar_2 = ttk.Progressbar(self.tab1, orient="horizontal", length=200, mode="indeterminate")
            self.progress_bar_2.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            # add text holder for recieving input from user
            text_holder = tk.Text(self.tab1, height=1, width=50)
            text_holder.pack(side=tk.TOP, padx=10, pady=5)

            # add button for getting input from user
            self.get_input_button = tk.Button(self.tab1, text="Pattern Extension",
                                              command=lambda: self.extension_more(text_holder))
            self.get_input_button.pack(side=tk.TOP, padx=10, pady=5)

            # # show results on canvas
            self.result_canvas2 = tk.Canvas(self.tab1)
            self.result_canvas2.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            # add a frame for showing the more results beside canvas
            table_result_frame = tk.Frame(self.result_canvas2)
            table_result_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

            # add a message on top the table
            self.table_result_text = tk.Label(table_result_frame, text="Right click on your desired pattern to"
                                                                       " put in its name on the clipboard")
            self.table_result_text.pack(side=tk.TOP, padx=10, pady=5)

            # add tree widget for showing the results
            tree = self.creat_table(table_result_frame)

            # create a new thread for running the detection
            self.progress_bar_2.start(10)
            # self.run_extension()
            self.thread_2 = threading.Thread(target=self.run_extension(tree, Core_activity))
            self.thread_2.start()

    def run_extension(self, tree, Core_activity):
        self.all_pattern_dictionary = dict()
        self.All_Pareto_front = dict()
        self.EventLog_graphs = dict()
        self.Patterns_Dictionary = dict()
        self.all_variants = dict()
        d_time = self.delta_time_entry.get()
        d_time = float(d_time)

        filtered_cases = self.df.loc[self.df[self.activity] == Core_activity, self.case_id]
        filtered_main_data = self.df[self.df[self.case_id].isin(filtered_cases)]
        # # Keep only variants and its frequency
        # timestamp = self.timestamp
        # filtered_main_data = pm4py.format_dataframe(filtered_main_data, case_id=self.case_id,
        #                                             activity_key=self.activity,
        #                                             timestamp_key=timestamp)
        # filtered_main_log = pm4py.convert_to_event_log(filtered_main_data)
        # variants = variants_filter.get_variants(filtered_main_log)
        # pp_log = EventLog()
        # pp_log._attributes = filtered_main_log.attributes
        # for i, k in enumerate(variants):
        #     variants[k][0].attributes['VariantFrequency'] = len(variants[k])
        #     Case_ids = []
        #
        #     for trace in variants[k]:
        #         Case_ids.append(trace.attributes['concept:name'])
        #
        #     variants[k][0].attributes['CaseIDs'] = Case_ids
        #     pp_log.append(variants[k][0])
        #
        # selected_variants = pm4py.convert_to_dataframe(pp_log)
        # self.all_variants[Core_activity] = selected_variants
        # timestamp = 'time:timestamp'
        for case in filtered_main_data[self.case_id].unique():
            case_data = filtered_main_data[filtered_main_data[self.case_id] == case]
            if case not in self.EventLog_graphs.keys():
                Trace_graph = Trace_graph_generator(filtered_main_data, d_time,
                                                    case, self.color_act_dict,
                                                    self.case_id, self.activity, self.timestamp)

                self.EventLog_graphs[case] = Trace_graph.copy()
            else:
                Trace_graph = self.EventLog_graphs[case].copy()

            self.Patterns_Dictionary, _ = Pattern_extension(case_data, Trace_graph, Core_activity,
                                                         self.case_id, self.Patterns_Dictionary,
                                                         self.Max_gap_between_events)

        self.patient_data[list(self.Patterns_Dictionary.keys())] = 0
        for PID in self.Patterns_Dictionary:
            for CaseID in np.unique(self.Patterns_Dictionary[PID]['Instances']['case']):
                variant_frequency_case = self.Patterns_Dictionary[PID]['Instances']['case'].count(CaseID)
                self.patient_data.loc[self.patient_data[self.case_id] == CaseID, PID] = variant_frequency_case

        pattern_attributes = create_pattern_attributes(self.patient_data, self.outcome,
                                                       list(self.Patterns_Dictionary.keys()),
                                                       self.pairwise_distances_array, self.pair_cases,
                                                       self.start_search_points, self.outcome_type)

        Objectives_attributes = pattern_attributes[self.pareto_features]
        if 'Outcome_Correlation' in self.pareto_features:
            Objectives_attributes['Outcome_Correlation'] = np.abs(Objectives_attributes['Outcome_Correlation'])

        mask = paretoset(Objectives_attributes, sense=self.pareto_sense)
        paretoset_patterns = pattern_attributes[mask]

        self.All_Pareto_front[Core_activity] = dict()
        self.All_Pareto_front[Core_activity]['dict'] = self.Patterns_Dictionary
        # self.All_Pareto_front[Core_activity]['variants'] = selected_variants
        self.all_pattern_dictionary.update(self.Patterns_Dictionary)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(self.visualization[0])
        ax.set_ylabel(self.visualization[1])
        ax.set_zlabel(self.visualization[2])
        for ticker, row in paretoset_patterns.iterrows():
            ax.scatter(row[self.visualization[0]], row[self.visualization[1]], row[self.visualization[2]])
            ax.text(row[self.visualization[0]], row[self.visualization[1]], row[self.visualization[2]],
                    row['patterns'])

        self.progress_bar_2.stop()
        self.progress_bar_2.destroy()

        canvas = FigureCanvasTkAgg(fig, master=self.result_canvas2)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        for ticker, row in paretoset_patterns.iterrows():
            tree.insert("", tk.END, values=(row['patterns'], row['Frequency_Interest'],
                                            row['Outcome_Interest'], row['Case_Distance_Interest']))

        # create a tab for each pattern in the pareto front
        for ticker, row in paretoset_patterns.iterrows():
            tab_name = row['patterns']
            tab = ttk.Frame(self.tab_control)
            self.tab_control.add(tab, text=tab_name)
            self.tab_control.pack(expand=1, fill="both")
            ploting_features = 2 + len(self.numerical_attributes) + len(self.categorical_attributes)

            col_numbers = int(self.visualization_col_entry.get()) + 1
            row_numbers = int(self.visualization_row_entry.get())

            fig, ax = plot_patterns(self.Patterns_Dictionary, row['patterns'], self.color_act_dict
                                    , pattern_attributes, dim=(row_numbers, col_numbers))
            print(self.categorical_attributes)
            fig, ax = plot_dashboard(fig, ax, self.patient_data, self.numerical_attributes,
                                     self.categorical_attributes, tab_name)


            # Create vertical scrollbar
            scrollbary = tk.Scrollbar(tab, orient=tk.VERTICAL)
            scrollbary.pack(side=tk.RIGHT, fill=tk.Y)

            # Create horizontal scrollbar
            scrollbarx = tk.Scrollbar(tab, orient=tk.HORIZONTAL)
            scrollbarx.pack(side=tk.BOTTOM, fill=tk.X)

            # Create a canvas with a scroll region
            canvas = tk.Canvas(tab)
            canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            # Attach the canvas to the scrollbar
            canvas.config(yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)
            scrollbary.config(command=canvas.yview)
            scrollbarx.config(command=canvas.xview)

            # Create a frame for the figure inside the canvas
            frame = tk.Frame(canvas)
            canvas.create_window((0, 0), window=frame, anchor="nw")

            # Add the figure to the frame
            fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            # plt.savefig('./dashboard_%s_%s.png' % (Core_activity, ticker))
            canvas.draw_figure = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw_figure.get_tk_widget().pack(fill=tk.BOTH, expand=1)

            # Bind the canvas scrolling to the canvas size
            canvas.bind("<Configure>", lambda event, canvas=canvas: self.on_configure(event, canvas))

            # Bind mousewheel scroll event to the canvas
            canvas.bind("<MouseWheel>", lambda event, canvas=canvas: self.scroll(event, canvas))

            # Make the frame expand when the window is resized
            tab.grid_rowconfigure(0, weight=1)
            tab.grid_columnconfigure(0, weight=1)

    def on_configure(self, event, canvas):
        # Update the scroll region to match the canvas size
        canvas.configure(scrollregion=canvas.bbox("all"))

    def scroll(self, event, canvas):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def copy_to_clipboard(self, event, tree):
        # get the selected row and column
        region = tree.identify_region(event.x, event.y)
        if region == "cell":
            row, column = tree.identify_row(event.y), tree.identify_column(event.x)

            # get the value of the selected cell
            value = tree.item(row)['values'][0]

            # put the value on the clipboard
            pyperclip.copy(value)
            # show a message to the user
            messagebox.showinfo("Copied", "%s been copied to the clipboard" % value)

    def extension_more(self, text_holder):
        # get the input from the user
        Core_pattern = text_holder.get("1.0", tk.END)
        Core_pattern = Core_pattern[:-1]
        self.all_extended_patterns.update(self.Patterns_Dictionary)

        # check if the input is not empty
        if Core_pattern == "":
            messagebox.showerror("Error", "Please enter a foundational pattern for extension!")

        # check if the input is not empty
        elif Core_pattern not in self.all_extended_patterns.keys():
            messagebox.showerror("Error", "Selected pattern is not valid!")

        elif any(nx.get_edge_attributes(self.all_extended_patterns[Core_pattern]['pattern'], 'eventually').values()):
            messagebox.showerror("Error", "Patterns including eventually relations are not supported yet for extension")
        # if the input is valid
        else:
            # create a new windows for the results of extension
            extension_window = tk.Toplevel(self.master)
            extension_window.title("Extension Results: %s" % Core_pattern)
            extension_window.geometry("1000x1000")
            # extension_window.resizable(False, False)
            extension_window.grab_set()
            extension_window.focus_set()
            # add multiple tab for showing the results
            self.tab_control = ttk.Notebook(extension_window)
            self.tab_control.pack(expand=1, fill="both")
            # add a tab for showing the results of extension
            self.tab1 = ttk.Frame(self.tab_control)
            self.tab_control.add(self.tab1, text="Pareto Front")

            # add a progress bar for showing the progress of the extension
            self.progress_bar_2 = ttk.Progressbar(self.tab1, orient="horizontal", length=200, mode="indeterminate")
            self.progress_bar_2.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            # add text holder for recieving input from user
            text_holder = tk.Text(self.tab1, height=1, width=50)
            text_holder.pack(side=tk.TOP, padx=10, pady=5)

            # add button for getting input from user
            self.get_input_button = tk.Button(self.tab1, text="Pattern Extension",
                                              command=lambda: self.extension_more(text_holder))
            self.get_input_button.pack(side=tk.TOP, padx=10, pady=5)

            # # show results on canvas
            self.result_canvas2 = tk.Canvas(self.tab1)
            self.result_canvas2.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            # add a frame for showing the more results beside canvas
            table_result_frame = tk.Frame(self.result_canvas2)
            table_result_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

            # add a message on top the table
            self.table_result_text = tk.Label(table_result_frame, text="Right click on your desired pattern to"
                                                                       " put in its name on the clipboard")
            self.table_result_text.pack(side=tk.TOP, padx=10, pady=5)

            # add tree widget for showing the results
            tree = self.creat_table(table_result_frame)

            # create a new thread for running the detection
            self.progress_bar_2.start(10)
            thread = threading.Thread(target=self.run_pattern_extension(tree, Core_pattern))
            thread.start()

    def run_pattern_extension(self, tree, Core_pattern):
        self.all_extended_patterns, self.Patterns_Dictionary, self.patient_data = Single_Pattern_Extender(
            self.all_extended_patterns,
            Core_pattern,
            self.patient_data, self.EventLog_graphs,
            self.df, self.Max_gap_between_events, self.activity, self.case_id)

        result_dict = {'K': [], 'N': [], 'Pareto': [], 'All': []}
        for obj in self.pareto_features:
            result_dict[obj] = []
        pattern_attributes = create_pattern_attributes(self.patient_data, self.outcome,
                                                       list(self.Patterns_Dictionary.keys()),
                                                       self.pairwise_distances_array, self.pair_cases,
                                                       self.start_search_points, self.outcome_type)
        Objectives_attributes = pattern_attributes[self.pareto_features]
        if 'OutcomeCorrelation' in self.pareto_features:
            Objectives_attributes['OutcomeCorrelation'] = np.abs(Objectives_attributes['OutcomeCorrelation'])
        mask = paretoset(Objectives_attributes, sense=self.pareto_sense)
        paretoset_patterns = pattern_attributes[mask]

        # plot the results on the canvas
        col_numbers = int(self.visualization_col_entry.get()) + 1
        row_numbers = int(self.visualization_row_entry.get())

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(self.visualization[0])
        ax.set_ylabel(self.visualization[1])
        ax.set_zlabel(self.visualization[2])
        for ticker, row in paretoset_patterns.iterrows():
            ax.scatter(row[self.visualization[0]], row[self.visualization[1]], row[self.visualization[2]])
            ax.text(row[self.visualization[0]], row[self.visualization[1]], row[self.visualization[2]],
                    row['patterns'])

        self.progress_bar_2.stop()
        self.progress_bar_2.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.result_canvas2)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        for ticker, row in paretoset_patterns.iterrows():
            tree.insert("", tk.END, values=(row['patterns'], row['Frequency_Interest'],
                                            row['Outcome_Interest'], row['Case_Distance_Interest']))

        # create a tab for each pattern in the pareto front
        for ticker, row in paretoset_patterns.iterrows():
            tab_name = row['patterns']
            # number_of_pattern = tab_name.split('_')[-1]
            tab = ttk.Frame(self.tab_control)
            self.tab_control.add(tab, text=tab_name)
            self.tab_control.pack(expand=1, fill="both")
            ploting_features = 2 + len(self.numerical_attributes) + len(self.categorical_attributes)

            fig, ax = plot_patterns(self.Patterns_Dictionary, row['patterns'], self.color_act_dict
                                    , pattern_attributes, dim=(row_numbers, col_numbers))

            fig, ax = plot_dashboard(fig, ax, self.patient_data, self.numerical_attributes,
                                     self.categorical_attributes, tab_name)

            # Create vertical scrollbar
            scrollbary = tk.Scrollbar(tab, orient=tk.VERTICAL)
            scrollbary.pack(side=tk.RIGHT, fill=tk.Y)

            # Create horizontal scrollbar
            scrollbarx = tk.Scrollbar(tab, orient=tk.HORIZONTAL)
            scrollbarx.pack(side=tk.BOTTOM, fill=tk.X)

            # Create a canvas with a scroll region
            canvas = tk.Canvas(tab)
            canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            # Attach the canvas to the scrollbar
            canvas.config(yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)
            scrollbary.config(command=canvas.yview)
            scrollbarx.config(command=canvas.xview)

            # Create a frame for the figure inside the canvas
            frame = tk.Frame(canvas)
            canvas.create_window((0, 0), window=frame, anchor="nw")

            # Add the figure to the frame
            fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            # plt.savefig('./dashboard_%s_%s.png' % (Core_pattern, ticker))
            canvas.draw_figure = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw_figure.get_tk_widget().pack(fill=tk.BOTH, expand=1)

            # Bind the canvas scrolling to the canvas size
            canvas.bind("<Configure>", lambda event, canvas=canvas: self.on_configure(event, canvas))

            # Bind mousewheel scroll event to the canvas
            canvas.bind("<MouseWheel>", lambda event, canvas=canvas: self.scroll(event, canvas))

            # Make the frame expand when the window is resized
            tab.grid_rowconfigure(0, weight=1)
            tab.grid_columnconfigure(0, weight=1)

    def Automatic_detection(self):
        if self.correlation_function.get() == 0 and \
                self.frequency_function.get() == 0 and \
                self.distance_function.get() == 0:
            messagebox.showerror("Error", "You need to select at least one interest function!")
        else:
            # check if the checkbox for interest function is checked
            self.pareto_features = []
            self.pareto_sense = []
            if self.correlation_function.get():
                self.pareto_features.append('Outcome_Interest')
                self.pareto_sense.append(self.direction_correlation_function.get())
            if self.frequency_function.get():
                self.pareto_features.append('Frequency_Interest')
                self.pareto_sense.append(self.direction_frequency_function.get())
            if self.distance_function.get():
                self.pareto_features.append('Case_Distance_Interest')
                self.pareto_sense.append(self.direction_distance_function.get())

            # open new windows to get the parameters for the automatic detection
            self.Automatic_detection_window = tk.Toplevel(self.master)
            self.Automatic_detection_window.title("Automatic detection")
            self.Automatic_detection_window.grab_set()
            self.Automatic_detection_window.focus_set()
            # self.Automatic_detection_window.geometry("500x500")

            # add text holder for receiving input from user below
            setting_frame = tk.Frame(self.Automatic_detection_window)
            setting_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
            text_holder_label = tk.Label(setting_frame, text="Max extension step: ")
            text_holder_label.pack(side=tk.LEFT, padx=10, pady=5)
            text_holder = tk.Entry(setting_frame, width=5)
            text_holder.pack(side=tk.LEFT, padx=10, pady=5)

            eventual_label = tk.Label(setting_frame, text="Max gap between events: ")
            eventual_label.pack(side=tk.LEFT, padx=10, pady=5)
            eventual_holder = tk.Entry(setting_frame, width=5)
            eventual_holder.pack(side=tk.LEFT, padx=10, pady=5)

            test_label = tk.Label(setting_frame, text="Testing percentage: ")
            test_label.pack(side=tk.LEFT, padx=10, pady=5)
            test_holder = tk.Entry(setting_frame, width=5)
            test_holder.pack(side=tk.LEFT, padx=10, pady=5)

            # get the location to save the results of the automatic detection
            folder_frame = tk.Frame(self.Automatic_detection_window)
            folder_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
            folder_label = tk.Label(folder_frame, text="No folder selected", borderwidth=1,
                                    relief="solid",
                                    width=50, background="white", anchor="w", padx=5, pady=5, justify="left")
            folder_label.pack(side=tk.LEFT, padx=10, pady=5)

            select_file_button = tk.Button(folder_frame, text="Select folder",
                                           command=lambda: self.select_folder(folder_label))
            select_file_button.pack(side=tk.LEFT, padx=10, pady=5)

            # add button to start the automatic detection
            button_frame = tk.Frame(self.Automatic_detection_window)
            button_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
            button = tk.Button(button_frame, text="Start Automatic detection",
                               command=lambda: self.start_automatic_detection(text_holder, eventual_holder,
                                                                              test_holder))
            button.pack(side=tk.BOTTOM, padx=10, pady=5)

    def select_folder(self, folder_label):
        # get file path for only csv files
        self.saving_directory = filedialog.askdirectory(title="Select an event log (.cvs format)")

        # check if the directory exists
        if self.saving_directory != "":
            folder_label.config(text=self.saving_directory)
        else:
            messagebox.showerror("Error", "Please select a folder")

    def start_automatic_detection(self, text_holder, eventual_holder, test_holder):
        # creat a frame for loading message
        self.loading_frame = tk.Frame(self.Automatic_detection_window)
        self.loading_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        self.loading_text = "Loading... \n Please wait..."
        self.loading_label = tk.Label(self.loading_frame, text=self.loading_text, justify="left")
        self.loading_label.pack(side=tk.BOTTOM, padx=10, pady=5)

        Max_extension_step = text_holder.get()
        Max_extension_step = int(Max_extension_step)

        self.Max_gap_between_events = eventual_holder.get()
        self.Max_gap_between_events = int(self.Max_gap_between_events)

        test_data_percentage = test_holder.get()
        test_data_percentage = float(test_data_percentage)

        self.patient_data = self.create_patient_data()
        selected_variants = VariantSelection(self.df, self.case_id, self.activity, self.timestamp)
        counter = 1
        for case in selected_variants["case:concept:name"].unique():
            counter += 1
            Other_cases = \
                selected_variants.loc[selected_variants["case:concept:name"] == case, 'case:CaseIDs'].tolist()[0]
            trace = self.df.loc[self.df[self.case_id] == case, self.activity].tolist()
            for act in np.unique(trace):
                Number_of_act = trace.count(act)
                for Ocase in Other_cases:
                    self.patient_data.loc[self.patient_data[self.case_id] == Ocase, act] = Number_of_act

        d_time = self.delta_time_entry.get()
        d_time = float(d_time)
        pairwise_distances_array, pair_cases, start_search_points = self.creat_pairwise_distance()

        train_X, test_X = AutoStepWise_PPD(Max_extension_step, self.Max_gap_between_events,
                                           test_data_percentage, self.df, self.patient_data,
                                           pairwise_distances_array, pair_cases,
                                           start_search_points, self.case_id,
                                           self.activity, self.outcome, self.outcome_type,
                                           self.timestamp,
                                           self.pareto_features, self.pareto_sense, d_time,
                                           self.color_act_dict, self.saving_directory)

        # save the results
        train_X.to_csv(self.saving_directory + "/training_encoded_log.csv", index=False)
        test_X.to_csv(self.saving_directory + "/testing_encoded_log.csv", index=False)

        self.loading_label.config(text="discovery is done! \n Please check the folder for the results")


root = tk.Tk()
# add title to the main window
root.title("IMPresseD: Interactive Multi-Interest Process Pattern Discovery")
app = GUI_IMOPD_IKNL_tool(master=root)

app.master.mainloop()
