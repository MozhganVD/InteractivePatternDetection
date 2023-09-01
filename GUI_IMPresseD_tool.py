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
import pm4py
from matplotlib import pyplot as plt
import seaborn as sb
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from paretoset import paretoset
import pyperclip
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.obj import EventLog
from IMIPD import VariantSelection, create_pattern_attributes, calculate_pairwise_case_distance, Trace_graph_generator, \
    Pattern_extension, plot_patterns, Single_Pattern_Extender


class GUI_IMOPD_IKNL_tool:
    def __init__(self, master=None):
        super().__init__()
        self.master = master
        self.with_categorical = False
        self.with_numerical = False
        # add welcome message
        self.welcome_label = tk.Label(self.master, text="Welcome to IMPresseD")
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

        # create a button for starting the detection
        self.start_detection_button = tk.Button(self.master, text="Start Pattern Discovery",
                                                command=self.start_detection)
        self.interest_function_frame = tk.Frame(self.master)
        self.interest_function_frame_2 = tk.Frame(self.interest_function_frame)
        self.interest_function_frame_3 = tk.Frame(self.interest_function_frame)

    def select_file(self):
        # get file path for only csv files
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")],
                                                    title="Select an event log (.cvs format)")

        # check if filepath exists and in csv format
        if self.file_path and self.file_path.endswith('.csv'):
            # update the file label
            self.file_label.config(text=self.file_path)
            self.df = pd.read_csv(self.file_path)
            self.show_setting(self.df)
        else:
            messagebox.showerror("Error", "Please select a csv file")

    def show_setting(self, df):
        # clear combo boxes
        self.case_id_combobox.set('')
        self.activity_combobox.set('')
        self.timestamp_combobox.set('')
        self.outcome_combobox.set('')

        # Clear all buttons and text widgets
        self.start_detection_button.destroy()
        self.interest_function_frame.destroy()
        self.interest_function_frame_2.destroy()
        self.interest_function_frame_3.destroy()

        # add column names to the comboboxes
        self.case_id_combobox.config(values=list(df.columns))
        self.activity_combobox.config(values=list(df.columns))
        self.timestamp_combobox.config(values=list(df.columns))
        self.outcome_combobox.config(values=list(df.columns))
        # clear menu for numerical attributes
        self.menu_num.delete(0, tk.END)
        self.menu.delete(0, tk.END)

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
        # create a dictionary for saving all the extended patterns
        self.all_extended_patterns = dict()
        # get the selected values from the comboboxes
        self.case_id = self.case_id_combobox.get()
        self.activity = self.activity_combobox.get()
        self.timestamp = self.timestamp_combobox.get()
        self.outcome = self.outcome_combobox.get()

        # check if all comboboxes are filled
        if self.case_id and self.activity and self.timestamp and self.outcome:
            # set the right format for the dataframe columns
            self.df[self.activity] = self.df[self.activity].str.replace("_", "-")
            self.df[self.timestamp] = pd.to_datetime(self.df[self.timestamp])
            self.df[self.case_id] = self.df[self.case_id].astype(str)
            self.df[self.case_id] = self.df[self.case_id].astype(str)

            color_codes = ["#" + ''.join([random.choice('000123456789ABCDEF') for i in range(6)])
                           for j in range(len(self.df[self.activity].unique()))]

            self.color_act_dict = dict()
            counter = 0
            for act in self.df[self.activity].unique():
                self.color_act_dict[act] = color_codes[counter]
                counter += 1
            self.color_act_dict['start'] = 'k'
            self.color_act_dict['end'] = 'k'

            self.categorical_attributes = []
            self.numerical_attributes = []
            for name, var in self.choices.items():
                if var.get() == 1:
                    self.categorical_attributes.append(name)
            for name, var in self.choices_num.items():
                if var.get() == 1:
                    self.numerical_attributes.append(name)

            # create a button for starting the detection
            self.start_detection_button = tk.Button(self.master, text="Start Pattern Discovery",
                                                    command=self.start_detection)
            self.start_detection_button.pack(side=tk.BOTTOM, padx=10, pady=10)

            # create a checkbox and combo box for three interest functions
            self.interest_function_frame = tk.Frame(self.master)
            self.interest_function_frame.pack(side=tk.BOTTOM, padx=10, pady=10)
            # self.correlation_function = tk.IntVar()
            # set a text box for the correlation function
            self.interest_function_label_1 = tk.Label(self.interest_function_frame,
                                                        text="Correlation interest function")
            self.interest_function_label_1.pack(side=tk.LEFT, padx=10, pady=10)
            # self.interest_function_checkbox_1 = tk.Checkbutton(self.interest_function_frame,
            #                                                    text="Correlation interest function",
            #                                                    variable=self.correlation_function)
            # self.interest_function_checkbox_1.pack(side=tk.LEFT, padx=10, pady=10)
            self.direction_correlation_function = tk.StringVar()
            self.direction_combobox_1 = ttk.Combobox(self.interest_function_frame,
                                                     textvariable=self.direction_correlation_function, state="readonly")
            self.direction_combobox_1['values'] = ("Max", "Min")
            self.direction_combobox_1.pack(side=tk.RIGHT, padx=10, pady=10)
            self.direction_combobox_1.current(0)

            self.interest_function_frame_2 = tk.Frame(self.master)
            self.interest_function_frame_2.pack(side=tk.BOTTOM, padx=10, pady=10)
            # self.frequency_function = tk.IntVar()
            self.interest_function_label_2 = tk.Label(self.interest_function_frame_2,
                                                        text="Frequency interest function")
            self.interest_function_label_2.pack(side=tk.LEFT, padx=10, pady=10)
            # self.interest_function_checkbox_2 = tk.Checkbutton(self.interest_function_frame_2,
            #                                                    text="Frequency interest function",
            #                                                    variable=self.frequency_function)
            # self.interest_function_checkbox_2.pack(side=tk.LEFT, padx=10, pady=10)
            self.direction_frequency_function = tk.StringVar()
            self.direction_combobox_2 = ttk.Combobox(self.interest_function_frame_2,
                                                     textvariable=self.direction_frequency_function, state="readonly")
            self.direction_combobox_2['values'] = ("Max", "Min")
            self.direction_combobox_2.pack(side=tk.RIGHT, padx=10, pady=10)
            self.direction_combobox_2.current(0)

            self.interest_function_frame_3 = tk.Frame(self.master)
            self.interest_function_frame_3.pack(side=tk.BOTTOM, padx=10, pady=10)
            # self.distance_function = tk.IntVar()
            self.interest_function_label_3 = tk.Label(self.interest_function_frame_3,
                                                        text="Case Distance interest function")
            self.interest_function_label_3.pack(side=tk.LEFT, padx=10, pady=10)
            # self.interest_function_checkbox_3 = tk.Checkbutton(self.interest_function_frame_3,
            #                                                    text="Case Distance interest function",
            #                                                    variable=self.distance_function)
            # self.interest_function_checkbox_3.pack(side=tk.LEFT, padx=10, pady=10)
            self.direction_distance_function = tk.StringVar()
            self.direction_combobox_3 = ttk.Combobox(self.interest_function_frame_3,
                                                     textvariable=self.direction_distance_function, state="readonly")
            self.direction_combobox_3['values'] = ("Max", "Min")
            self.direction_combobox_3.pack(side=tk.RIGHT, padx=10, pady=10)
            self.direction_combobox_3.current(1)

        else:
            messagebox.showerror("Error", "You need to select case id, activity, timestamp and outcome!")

    def start_detection(self):
        # create a new window for showing the results
        self.result_window = tk.Toplevel(self.master)
        self.result_window.title("Results")
        self.result_window.geometry("1000x900")
        self.result_window.resizable(False, False)
        self.result_window.grab_set()
        self.result_window.focus_set()
        self.result_window.transient(self.master)

        self.progress_bar = ttk.Progressbar(self.result_window, orient=tk.HORIZONTAL, length=900, mode='indeterminate')
        self.progress_bar.pack(side=tk.TOP, padx=10, pady=10)

        # add text holder for recieving input from user
        self.text_holder = tk.Text(self.result_window, height=1, width=50)
        self.text_holder.pack(side=tk.TOP, padx=10, pady=10)

        # add button for getting input from user
        self.get_input_button = tk.Button(self.result_window, text="Pattern Extension", command=self.extension)
        self.get_input_button.pack(side=tk.TOP, padx=10, pady=10)

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
        self.table_result_text.pack(side=tk.TOP, padx=10, pady=10)
        # add tree widget for showing the results
        tree = self.creat_table(self.table_result_frame)
        # create a new thread for running the detection
        self.thread = threading.Thread(target=self.run_detection(tree))
        self.thread.start()

    def run_detection(self, tree):
        # create a progress bar at the bottom of the result window
        self.progress_bar.start(10)

        self.patient_data = pd.DataFrame()
        if self.with_numerical:
            self.patient_data[self.numerical_attributes] = self.df[self.numerical_attributes]
        if self.with_categorical:
            self.patient_data[self.categorical_attributes] = self.df[self.categorical_attributes]

        self.patient_data[self.case_id] = self.df[self.case_id]
        self.patient_data[self.outcome] = self.df[self.outcome]
        self.patient_data = self.patient_data.drop_duplicates(subset=[self.case_id], keep='first')

        self.patient_data.sort_values(by=self.case_id, inplace=True)
        self.patient_data.reset_index(inplace=True, drop=True)

        # variant selection
        selected_variants = VariantSelection(self.df, self.case_id, self.activity, self.timestamp)

        # create feature for each activity
        self.patient_data[list(self.df[self.activity].unique())] = 0
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

        # calculate the pairwise case distance or load it from the file if it is already calculated
        folder_address = os.path.dirname(self.file_path) + "/"
        distance_files = folder_address + "dist"
        if not os.path.exists(distance_files):
            X_features = self.patient_data.drop([self.case_id, self.outcome], axis=1)
            self.pairwise_distances_array = calculate_pairwise_case_distance(X_features, self.numerical_attributes)
            os.makedirs(distance_files)
            with open(distance_files + "/pairwise_case_distances.pkl", 'wb') as f:
                pickle.dump(self.pairwise_distances_array, f)

        else:
            with open(distance_files + "/pairwise_case_distances.pkl", 'rb') as f:
                self.pairwise_distances_array = pickle.load(f)

        self.pair_cases = [(a, b) for idx, a in enumerate(self.patient_data.index) for b in
                           self.patient_data.index[idx + 1:]]
        case_size = len(self.patient_data)
        i = 0
        self.start_search_points = []
        for k in range(case_size):
            self.start_search_points.append(k * case_size - (i + k))
            i += k

        activity_attributes = create_pattern_attributes(self.patient_data, self.outcome, None,
                                                        list(self.df[self.activity].unique()),
                                                        self.pairwise_distances_array, self.pair_cases,
                                                        self.start_search_points)

        self.pareto_features = ['Outcome_Interest', 'Frequency_Interest', 'Case_Distance_Interest']
        self.pareto_sense = [self.direction_correlation_function.get(),
                             self.direction_frequency_function.get(),
                             self.direction_distance_function.get()]
        # check if the checkbox for interest function is checked
        # if self.correlation_function.get():
        #     self.pareto_features.append('Outcome_Interest')
        #     self.pareto_sense.append(self.direction_correlation_function.get())
        # if self.frequency_function.get():
        #     self.pareto_features.append('Frequency_Interest')
        #     self.pareto_sense.append(self.direction_frequency_function.get())
        # if self.distance_function.get():
        #     self.pareto_features.append('Case_Distance_Interest')
        #     self.pareto_sense.append(self.direction_distance_function.get())

        Objectives_attributes = activity_attributes[self.pareto_features]
        if 'Outcome_Correlation' in self.pareto_features:
            Objectives_attributes['Outcome_Correlation'] = np.abs(Objectives_attributes['Outcome_Correlation'])

        mask = paretoset(Objectives_attributes, sense=self.pareto_sense)
        paretoset_activities = activity_attributes[mask]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(self.pareto_features[0])
        ax.set_ylabel(self.pareto_features[1])
        ax.set_zlabel(self.pareto_features[2])
        for ticker, row in paretoset_activities.iterrows():
            ax.scatter(row[self.pareto_features[0]], row[self.pareto_features[1]], row[self.pareto_features[2]],
                       c=self.color_act_dict[row['patterns']])
            ax.text(row[self.pareto_features[0]], row[self.pareto_features[1]], row[self.pareto_features[2]],
                    row['patterns'])

        # make progress bar invisible
        self.progress_bar.stop()
        self.progress_bar.pack_forget()

        # create a new thread for running the detection
        canvas = FigureCanvasTkAgg(fig, master=self.result_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # insert the unique activities in dataframe to the text box for showing the results i none column
        # for activity in paretoset_activities['patterns'].unique():
        #     self.result_text.insert(tk.END, activity + "\n")

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

        # add vertical scrollbar for the tree widget
        tree_scrollbar = tk.Scrollbar(table_result_frame, orient=tk.VERTICAL)
        tree_scrollbar.config(command=tree.yview)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.config(yscrollcommand=tree_scrollbar.set)

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
            extension_window.resizable(False, False)
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
            text_holder.pack(side=tk.TOP, padx=10, pady=10)

            # add button for getting input from user
            self.get_input_button = tk.Button(self.tab1, text="Pattern Extension",
                                              command=lambda: self.extension_more(text_holder))
            self.get_input_button.pack(side=tk.TOP, padx=10, pady=10)

            # # show results on canvas
            self.result_canvas2 = tk.Canvas(self.tab1)
            self.result_canvas2.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            # add a frame for showing the more results beside canvas
            table_result_frame = tk.Frame(self.result_canvas2)
            table_result_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

            # add a message on top the table
            self.table_result_text = tk.Label(table_result_frame, text="Right click on your desired pattern to"
                                                                       " put in its name on the clipboard")
            self.table_result_text.pack(side=tk.TOP, padx=10, pady=10)

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
        filtered_cases = self.df.loc[self.df[self.activity] == Core_activity, self.case_id]
        filtered_main_data = self.df[self.df[self.case_id].isin(filtered_cases)]
        # Keep only variants and its frequency
        timestamp = self.timestamp
        filtered_main_data = pm4py.format_dataframe(filtered_main_data, case_id=self.case_id,
                                                    activity_key=self.activity,
                                                    timestamp_key=timestamp)
        filtered_main_log = pm4py.convert_to_event_log(filtered_main_data)
        variants = variants_filter.get_variants(filtered_main_log)
        pp_log = EventLog()
        pp_log._attributes = filtered_main_log.attributes
        for i, k in enumerate(variants):
            variants[k][0].attributes['VariantFrequency'] = len(variants[k])
            Case_ids = []

            for trace in variants[k]:
                Case_ids.append(trace.attributes['concept:name'])

            variants[k][0].attributes['CaseIDs'] = Case_ids
            pp_log.append(variants[k][0])

        selected_variants = pm4py.convert_to_dataframe(pp_log)
        self.all_variants[Core_activity] = selected_variants
        timestamp = 'time:timestamp'
        for case in selected_variants[self.case_id].unique():
            case_data = selected_variants[selected_variants[self.case_id] == case]
            if case not in self.EventLog_graphs.keys():
                Trace_graph = Trace_graph_generator(selected_variants, self.patient_data, Core_activity, 1,
                                                    case, self.color_act_dict,
                                                    self.case_id, self.activity, timestamp)

                self.EventLog_graphs[case] = Trace_graph.copy()
            else:
                Trace_graph = self.EventLog_graphs[case].copy()

            self.Patterns_Dictionary = Pattern_extension(case_data, Trace_graph, Core_activity,
                                                         self.case_id, self.Patterns_Dictionary)

        self.patient_data[list(self.Patterns_Dictionary.keys())] = 0
        for PID in self.Patterns_Dictionary:
            for CaseID in np.unique(self.Patterns_Dictionary[PID]['Instances']['case']):
                variant_frequency_case = self.Patterns_Dictionary[PID]['Instances']['case'].count(CaseID)
                Other_cases = \
                    selected_variants.loc[selected_variants[self.case_id] == CaseID, 'case:CaseIDs'].tolist()[
                        0]
                for Ocase in Other_cases:
                    self.patient_data.loc[self.patient_data[self.case_id] == Ocase, PID] = variant_frequency_case

        pattern_attributes = create_pattern_attributes(self.patient_data, self.outcome,
                                                       Core_activity, list(self.Patterns_Dictionary.keys()),
                                                       self.pairwise_distances_array, self.pair_cases,
                                                       self.start_search_points)

        Objectives_attributes = pattern_attributes[self.pareto_features]
        if 'Outcome_Correlation' in self.pareto_features:
            Objectives_attributes['Outcome_Correlation'] = np.abs(Objectives_attributes['Outcome_Correlation'])

        mask = paretoset(Objectives_attributes, sense=self.pareto_sense)
        paretoset_patterns = pattern_attributes[mask]

        self.All_Pareto_front[Core_activity] = dict()
        self.All_Pareto_front[Core_activity]['dict'] = self.Patterns_Dictionary
        self.All_Pareto_front[Core_activity]['variants'] = selected_variants
        self.all_pattern_dictionary.update(self.Patterns_Dictionary)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(self.pareto_features[0])
        ax.set_ylabel(self.pareto_features[1])
        ax.set_zlabel(self.pareto_features[2])
        for ticker, row in paretoset_patterns.iterrows():
            ax.scatter(row[self.pareto_features[0]], row[self.pareto_features[1]], row[self.pareto_features[2]])
            ax.text(row[self.pareto_features[0]], row[self.pareto_features[1]], row[self.pareto_features[2]],
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

            col_numbers = int(np.ceil(ploting_features / 2))
            row_numbers = 2

            fig, ax = plot_patterns(self.Patterns_Dictionary, row['patterns'], self.color_act_dict
                                    , pattern_attributes, dim=(row_numbers, col_numbers))

            # plot the distribution of numerical attributes for the pattern
            for ii, num in enumerate(self.numerical_attributes):
                # patient_data_core = self.patient_data[self.patient_data[Core_activity] != 0]
                sb.distplot(self.patient_data.loc[self.patient_data[tab_name] == 0, num], ax=ax[0, ii + 1], color="g")
                sb.distplot(self.patient_data.loc[self.patient_data[tab_name] > 0, num], ax=ax[0, ii + 1], color="r")
                ax[0, ii + 1].set_title(num)
                # ax[0, ii + 1].title.set_size(40)
                ax[0, ii + 1].set_xlabel('')
                # ax[0, ii + 1].set_ylabel('density')
                # set font size for x and y axis
                # ax[0, ii + 1].tick_params(axis='both', which='major', labelsize=16)

            # plot pie chart for categorical attributes
            r = 1
            jj = 1
            cmap = plt.get_cmap("tab20c")
            for cat in self.categorical_attributes:
                all_cat_features = self.patient_data[cat].unique().tolist()
                all_cat_features.sort()

                cat_features_outpattern = self.patient_data.loc[
                    self.patient_data[tab_name] == 0, cat].unique().tolist()
                cat_features_outpattern.sort()

                cat_features_inpattern = self.patient_data.loc[self.patient_data[tab_name] > 0, cat].unique().tolist()
                cat_features_inpattern.sort()

                indexes = [all_cat_features.index(l) for l in cat_features_inpattern]
                outdexes = [all_cat_features.index(l) for l in cat_features_outpattern]

                all_feature_colors = cmap(np.arange(len(all_cat_features)) * 1)

                outer_colors = all_feature_colors[outdexes]
                inner_colors = all_feature_colors[indexes]

                textprops = {"fontsize": 8}
                ax[r, jj].pie(
                    pd.DataFrame(
                        self.patient_data.loc[self.patient_data[tab_name] == 0, cat].value_counts()).sort_index()[
                        cat],
                    radius=1,
                    labels=cat_features_outpattern, colors=outer_colors, wedgeprops=dict(width=0.4, edgecolor='w'),
                    textprops=textprops)

                ax[r, jj].pie(
                    pd.DataFrame(
                        self.patient_data.loc[self.patient_data[tab_name] > 0, cat].value_counts()).sort_index()[cat],
                    radius=1 - 0.4,
                    labels=cat_features_inpattern, colors=inner_colors, wedgeprops=dict(width=0.4, edgecolor='w'),
                    textprops=textprops)

                ax[r, jj].set_title(cat)
                # ax[r, jj].title.set_size(40)

                jj += 1
                if jj > 5:
                    r += 1
                    jj = 1

            result_canvas = tk.Canvas(tab)
            result_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            canvas = FigureCanvasTkAgg(fig, master=result_canvas)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

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
            extension_window.geometry("1000x900")
            extension_window.resizable(False, False)
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
            text_holder.pack(side=tk.TOP, padx=10, pady=10)

            # add button for getting input from user
            self.get_input_button = tk.Button(self.tab1, text="Pattern Extension",
                                              command=lambda: self.extension_more(text_holder))
            self.get_input_button.pack(side=tk.TOP, padx=10, pady=10)

            # # show results on canvas
            self.result_canvas2 = tk.Canvas(self.tab1)
            self.result_canvas2.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            # add a frame for showing the more results beside canvas
            table_result_frame = tk.Frame(self.result_canvas2)
            table_result_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

            # add a message on top the table
            self.table_result_text = tk.Label(table_result_frame, text="Right click on your desired pattern to"
                                                                       " put in its name on the clipboard")
            self.table_result_text.pack(side=tk.TOP, padx=10, pady=10)

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
            self.all_variants)

        result_dict = {'K': [], 'N': [], 'Pareto': [], 'All': []}
        for obj in self.pareto_features:
            result_dict[obj] = []
        pattern_attributes = create_pattern_attributes(self.patient_data, self.outcome,
                                                       Core_pattern, list(self.Patterns_Dictionary.keys()),
                                                       self.pairwise_distances_array, self.pair_cases,
                                                       self.start_search_points)
        Objectives_attributes = pattern_attributes[self.pareto_features]
        if 'OutcomeCorrelation' in self.pareto_features:
            Objectives_attributes['OutcomeCorrelation'] = np.abs(Objectives_attributes['OutcomeCorrelation'])
        mask = paretoset(Objectives_attributes, sense=self.pareto_sense)
        paretoset_patterns = pattern_attributes[mask]

        # plot the results on the canvas
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(self.pareto_features[0])
        ax.set_ylabel(self.pareto_features[1])
        ax.set_zlabel(self.pareto_features[2])
        for ticker, row in paretoset_patterns.iterrows():
            ax.scatter(row[self.pareto_features[0]], row[self.pareto_features[1]], row[self.pareto_features[2]])
            ax.text(row[self.pareto_features[0]], row[self.pareto_features[1]], row[self.pareto_features[2]],
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

            col_numbers = int(np.ceil(ploting_features / 2))
            row_numbers = 2

            fig, ax = plot_patterns(self.Patterns_Dictionary, row['patterns'], self.color_act_dict
                                    , pattern_attributes, dim=(row_numbers, col_numbers))

            # plot the distribution of numerical attributes for the pattern
            for ii, num in enumerate(self.numerical_attributes):
                # patient_data_core = self.patient_data[self.patient_data[Core_activity] != 0]
                sb.distplot(self.patient_data.loc[self.patient_data[tab_name] == 0, num], ax=ax[0, ii + 1], color="g")
                sb.distplot(self.patient_data.loc[self.patient_data[tab_name] > 0, num], ax=ax[0, ii + 1], color="r")
                ax[0, ii + 1].set_title(num)
                # ax[0, ii + 1].title.set_size(40)
                ax[0, ii + 1].set_xlabel('')
                # ax[0, ii + 1].set_ylabel('density')
                # set font size for x and y axis
                # ax[0, ii + 1].tick_params(axis='both', which='major', labelsize=16)

            # plot pie chart for categorical attributes
            r = 1
            jj = 1
            cmap = plt.get_cmap("tab20c")
            for cat in self.categorical_attributes:
                all_cat_features = self.patient_data[cat].unique().tolist()
                all_cat_features.sort()

                cat_features_outpattern = self.patient_data.loc[
                    self.patient_data[tab_name] == 0, cat].unique().tolist()
                cat_features_outpattern.sort()

                cat_features_inpattern = self.patient_data.loc[self.patient_data[tab_name] > 0, cat].unique().tolist()
                cat_features_inpattern.sort()

                indexes = [all_cat_features.index(l) for l in cat_features_inpattern]
                outdexes = [all_cat_features.index(l) for l in cat_features_outpattern]

                all_feature_colors = cmap(np.arange(len(all_cat_features)) * 1)

                outer_colors = all_feature_colors[outdexes]
                inner_colors = all_feature_colors[indexes]

                textprops = {"fontsize": 8}
                ax[r, jj].pie(
                    pd.DataFrame(
                        self.patient_data.loc[self.patient_data[tab_name] == 0, cat].value_counts()).sort_index()[
                        cat],
                    radius=1,
                    labels=cat_features_outpattern, colors=outer_colors, wedgeprops=dict(width=0.4, edgecolor='w'),
                    textprops=textprops)

                ax[r, jj].pie(
                    pd.DataFrame(
                        self.patient_data.loc[self.patient_data[tab_name] > 0, cat].value_counts()).sort_index()[cat],
                    radius=1 - 0.4,
                    labels=cat_features_inpattern, colors=inner_colors, wedgeprops=dict(width=0.4, edgecolor='w'),
                    textprops=textprops)

                ax[r, jj].set_title(cat)
                # ax[r, jj].title.set_size(40)

                jj += 1
                if jj > 5:
                    r += 1
                    jj = 1

            result_canvas = tk.Canvas(tab)
            result_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            canvas = FigureCanvasTkAgg(fig, master=result_canvas)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


root = tk.Tk()
# add title to the main window
root.title("IMPresseD: Interactive Multi-interest Process Pattern Discovery")
app = GUI_IMOPD_IKNL_tool(master=root)
app.master.mainloop()
