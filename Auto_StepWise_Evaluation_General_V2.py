from paretoset import paretoset
import pandas as pd
import networkx as nx
import pm4py
import pickle
import numpy as np
import networkx.algorithms.isomorphism as iso
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.obj import EventLog
import random
import os
import argparse
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from tools import create_pattern_attributes_IKNL_auto, create_embedded_pattern_in_trace, \
    Trace_graph_generator, update_pattern_dict, calculate_pairwise_case_distance, Pattern_Extender

random.seed(123)

parser = argparse.ArgumentParser(description="Pattern Detection Evaluation")

parser.add_argument("--folder_address",
                    type=str,
                    default="./datasets/BPIC2012_1/",
                    help="path to csv format log file")

parser.add_argument("--dataset_name",
                    type=str,
                    default="bpic_acc_trun40_allActs.csv",
                    help="name of the csv format log file")

parser.add_argument('--extension_numbers', default=6, type=float)
parser.add_argument('--Eventual_window', default=3, type=float)
parser.add_argument('--K_folds', default=5, type=float)

args = parser.parse_args()

folder_address = args.folder_address
dataset_name = args.dataset_name
extension_numbers = args.extension_numbers
Eventual_window = args.Eventual_window
Folds = args.K_folds

case_id = "case:concept:name"
activities = "concept:name"
label_col = None
label_class = 'label'
timestamp = 'Complete Timestamp'
Only_Complete = True

# select detectors:
Direct_context = True
Concurrence = True
Direct_predecessor = True
Direct_successor = True

# dataset settings
num_col = ['Age']
binary_col = []
cat_col = ['Diagnosis', 'Treatment code', 'Specialism code']
# num_col = ['open_cases', 'Work_Order_Qty']
# binary_col = []
# cat_col = ['Part_Desc_']

# defining result dataframe
results_dataframe = pd.DataFrame(columns=['Extension_step', 'K', 'Pareto', 'Pareto_std',
                                          'OutcomeCorrelation', 'OutcomeCorrelation_std',
                                          'Case_Coverage', 'Case_Coverage_std',
                                          'Case_Distance', 'Case_Distance_std',
                                          'All', 'All_std'])

d_time = 1  # in seconds
visualization_features = ['OutcomeCorrelation', 'Case_Coverage', 'Case_Distance']
pareto_features = ['OutcomeCorrelation', 'Case_Coverage', 'Case_Distance']  # , 'Case_Similarity'
pareto_sense = ["max", "max", "min"]  # , "max"

# read dataset
main_data = pd.read_csv(folder_address + dataset_name, sep=',')

if case_id not in main_data.columns:
    main_data.rename(columns={'Case ID': case_id, 'Activity': activities}, inplace=True)

main_data.loc[main_data[label_class] == 'deviant', label_class] = 1
main_data.loc[main_data[label_class] == 'regular', label_class] = 0

# main_data[activities] = main_data[activities].map(lambda x: x.split('-')[0])
# if Only_Complete:
#     main_data = main_data[main_data['lifecycle:transition'] == 'COMPLETE']

main_data[activities] = main_data[activities].str.replace("_", "-")
main_data[timestamp] = pd.to_datetime(main_data[timestamp])
main_data[case_id] = main_data[case_id].astype(str)

activities_freq = main_data.groupby(by=[activities])[case_id].count()
activities_freq = set(activities_freq[activities_freq < 3].index.tolist())
to_remove = []
for case in main_data[case_id].unique():
    trace = main_data.loc[main_data[case_id] == case, activities].tolist()
    if len(set(trace).intersection(activities_freq)) != 0:
        to_remove.append(case)

main_data = main_data[~main_data[case_id].isin(to_remove)]

color_codes = ["#" + ''.join([random.choice('000123456789ABCDEF') for i in range(6)])
               for j in range(len(main_data[activities].unique()))]

color_act_dict = dict()
counter = 0
for act in main_data[activities].unique():
    color_act_dict[act] = color_codes[counter]
    counter += 1
color_act_dict['start'] = 'k'
color_act_dict['end'] = 'k'

patient_data = main_data[num_col]
patient_data[cat_col] = main_data[cat_col]
patient_data[[case_id, label_class]] = main_data[[case_id, label_class]]
patient_data = patient_data.drop_duplicates(subset=[case_id], keep='first')

patient_data.sort_values(by=case_id, inplace=True)
patient_data.reset_index(inplace=True, drop=True)

distance_files = folder_address + "dist"
if not os.path.exists(distance_files):
    X_features = patient_data.drop([case_id, label_class], axis=1)
    pairwise_distances_array = calculate_pairwise_case_distance(X_features, num_col, Imput=False)
    os.makedirs(distance_files)
    with open(distance_files + "/pairwise_case_distances.pkl", 'wb') as f:
        pickle.dump(pairwise_distances_array, f)

else:
    with open(distance_files + "/pairwise_case_distances.pkl", 'rb') as f:
        pairwise_distances_array = pickle.load(f)

pair_cases = [(a, b) for idx, a in enumerate(patient_data.index) for b in patient_data.index[idx + 1:]]
case_size = len(patient_data)
i = 0
start_search_points = []
for k in range(case_size):
    start_search_points.append(k * case_size - (i + k))
    i += k
########################################################################################################################
#########################################Test and Train seperation for pattern detection################################
Unique_cases = patient_data[case_id].unique().tolist()
random.shuffle(Unique_cases)
split_step = len(Unique_cases) / Folds
train_list, test_list = dict(), dict()
for k in range(Folds):
    start_range = int(0 + (k * split_step))
    end_range = int(start_range + split_step)
    test_list[k + 1] = Unique_cases[start_range:end_range]
    train_list[k + 1] = list(set(Unique_cases).difference(set(test_list[k + 1])))

# Keep only variants and its frequency
filtered_main_data = pm4py.format_dataframe(main_data, case_id=case_id, activity_key=activities,
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

print("start single activity level search .... ")
patient_data[list(main_data[activities].unique())] = 0
counter = 1
for case in selected_variants[case_id].unique():
    # print(counter)
    counter += 1
    # case_data = main_data[main_data[case_id] == case]
    Other_cases = selected_variants.loc[selected_variants[case_id] == case, 'case:CaseIDs'].tolist()[0]
    trace = main_data.loc[main_data[case_id] == case, activities].tolist()
    for act in np.unique(trace):
        Number_of_act = trace.count(act)
        for Ocase in Other_cases:
            patient_data.loc[patient_data[case_id] == Ocase, act] = Number_of_act

ML_model = DecisionTreeClassifier()
result_dict = {'K': [], 'N': [], 'Pareto': [], 'All': []}
for obj in pareto_features:
    result_dict[obj] = []
All_extended_patterns = dict()
for k in range(len(train_list)):
    All_extended_patterns[k] = dict()
for fold in range(len(train_list)):
    train_patient_data = patient_data[patient_data[case_id].isin(train_list[fold + 1])]
    test_patient_data = patient_data[patient_data[case_id].isin(test_list[fold + 1])]

    activity_attributes = create_pattern_attributes_IKNL_auto(train_patient_data, label_col, label_class,
                                                              False, list(main_data[activities].unique()),
                                                              pairwise_distances_array, pair_cases, start_search_points)

    Objectives_attributes = activity_attributes[pareto_features]
    if 'OutcomeCorrelation' in pareto_features:
        Objectives_attributes['OutcomeCorrelation'] = np.abs(Objectives_attributes['OutcomeCorrelation'])

    mask = paretoset(Objectives_attributes, sense=pareto_sense)
    paretoset_activities = activity_attributes[mask]

    Top_K = len(paretoset_activities)
    result_dict['K'].append(Top_K)
    result_dict['N'].append(len(activity_attributes))

    All_extended_patterns[fold] = {'All': list(activity_attributes['patterns'])}
    for jj in range(len(pareto_features) + 1):
        if jj == len(pareto_features):
            selected_P = paretoset_activities
            Obj = 'Pareto'
        else:
            if pareto_sense[jj] == 'max':
                selected_P = activity_attributes.sort_values(by=[pareto_features[jj]], ascending=False).head(Top_K)
            else:
                selected_P = activity_attributes.sort_values(by=[pareto_features[jj]], ascending=True).head(Top_K)

            Obj = pareto_features[jj]

        All_extended_patterns[fold][Obj] = list(selected_P['patterns'])

    train_y = train_patient_data[label_class]
    test_y = test_patient_data[label_class]
    train_y = train_y.astype(str)
    test_y = test_y.astype(str)
    for Obj in All_extended_patterns[fold]:
        train_X = train_patient_data[All_extended_patterns[fold][Obj]]
        test_X = test_patient_data[All_extended_patterns[fold][Obj]]

        ML_model.fit(train_X, train_y)
        predicted = ML_model.predict(test_X)

        result_dict[Obj].append(f1_score(test_y, predicted, average='weighted'))

kFold_results = dict()
for obj in result_dict:
    kFold_results[obj] = np.mean(result_dict[obj])
    kFold_results[obj + "_std"] = np.std(result_dict[obj])

results_dataframe = results_dataframe.append(kFold_results, ignore_index=True)
results_dataframe.at[0, 'Extension_step'] = 0
results_dataframe.to_csv(folder_address + 'Auto_Results.csv', index=False)

Extended_patterns_at_stage = dict()
All_extended_patterns_1_list = []
EventLog_graphs = dict()
all_variants = dict()
for Core_activity in main_data[activities].unique():
    print('Core:  ' + Core_activity)
    timestamp = 'Complete Timestamp'
    filtered_cases = main_data.loc[main_data[activities] == Core_activity, case_id]
    filtered_main_data = main_data[main_data[case_id].isin(filtered_cases)]

    # Keep only variants and its frequency
    filtered_main_data = pm4py.format_dataframe(filtered_main_data, case_id=case_id, activity_key=activities,
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
    all_variants[Core_activity] = selected_variants
    timestamp = 'time:timestamp'
    new_patterns_for_core = []
    # Patterns_Dictionary = dict()

    for case in selected_variants[case_id].unique():
        case_data = selected_variants[selected_variants[case_id] == case]

        if case not in EventLog_graphs.keys():
            Trace_graph = Trace_graph_generator(selected_variants, patient_data, Core_activity, d_time,
                                                case, color_act_dict, case_id, activities, timestamp)

            EventLog_graphs[case] = Trace_graph.copy()
        else:
            Trace_graph = EventLog_graphs[case].copy()

        all_nodes = set(Trace_graph.nodes)
        nodes_values = [Trace_graph._node[n]['value'] for n in Trace_graph.nodes]
        nm = iso.categorical_node_match("value", nodes_values)
        em = iso.categorical_node_match("eventually", [True, False])
        # cases_in_variant = case_data['case:CaseIDs'].tolist()[0]
        for n in Trace_graph.nodes:
            if Trace_graph._node[n]['value'] == Core_activity:
                # directly preceding patterns
                preceding_pattern = nx.DiGraph()
                in_pattern_nodes = set(Trace_graph.pred[n].keys())
                if len(in_pattern_nodes) > 0:
                    preceding_pattern = Trace_graph.copy()
                    in_pattern_nodes.add(n)
                    to_remove = all_nodes.difference(in_pattern_nodes)
                    preceding_pattern.remove_nodes_from(to_remove)
                    if Direct_predecessor:
                        embedded_trace_graph = create_embedded_pattern_in_trace(in_pattern_nodes, Trace_graph)
                        Extended_patterns_at_stage, new_Pattern_IDs = update_pattern_dict(Extended_patterns_at_stage,
                                                                                          preceding_pattern,
                                                                                          embedded_trace_graph,
                                                                                          case_data, case_id, nm, em,
                                                                                          Core_activity,
                                                                                          new_patterns_for_core)

                        if new_Pattern_IDs != "":
                            new_patterns_for_core.append(new_Pattern_IDs)
                    in_pattern_nodes.remove(n)

                # directly following patterns
                following_pattern = nx.DiGraph()
                out_pattern_nodes = set(Trace_graph.succ[n].keys())
                if len(out_pattern_nodes) > 0:
                    following_pattern = Trace_graph.copy()
                    out_pattern_nodes.add(n)
                    to_remove = all_nodes.difference(out_pattern_nodes)
                    following_pattern.remove_nodes_from(to_remove)
                    if Direct_successor:
                        embedded_trace_graph = create_embedded_pattern_in_trace(out_pattern_nodes, Trace_graph)
                        Extended_patterns_at_stage, new_Pattern_IDs = update_pattern_dict(Extended_patterns_at_stage,
                                                                                          following_pattern,
                                                                                          embedded_trace_graph,
                                                                                          case_data, case_id, nm, em,
                                                                                          Core_activity,
                                                                                          new_patterns_for_core)
                        if new_Pattern_IDs != "":
                            new_patterns_for_core.append(new_Pattern_IDs)
                    out_pattern_nodes.remove(n)

                # parallel patterns (partial order)
                parallel_pattern_nodes = set()
                parallel_pattern = nx.DiGraph()
                if Trace_graph._node[n]['parallel']:
                    parallel_pattern_nodes.add(n)
                    for ND in Trace_graph.nodes:
                        if not Trace_graph._node[ND]['parallel']:
                            continue
                        in_pattern_ND = set(Trace_graph.in_edges._adjdict[ND].keys())
                        out_pattern_ND = set(Trace_graph.out_edges._adjdict[ND].keys())
                        if in_pattern_nodes == in_pattern_ND and out_pattern_nodes == out_pattern_ND:
                            parallel_pattern_nodes.add(ND)

                    parallel_pattern = Trace_graph.copy()
                    to_remove = all_nodes.difference(parallel_pattern_nodes)
                    parallel_pattern.remove_nodes_from(to_remove)
                    if Concurrence:
                        embedded_trace_graph = create_embedded_pattern_in_trace(parallel_pattern_nodes, Trace_graph)
                        Extended_patterns_at_stage, new_Pattern_IDs = update_pattern_dict(Extended_patterns_at_stage,
                                                                                          parallel_pattern,
                                                                                          embedded_trace_graph,
                                                                                          case_data, case_id, nm, em,
                                                                                          Core_activity,
                                                                                          new_patterns_for_core)
                        if new_Pattern_IDs != "":
                            new_patterns_for_core.append(new_Pattern_IDs)
                if Direct_context:
                    # combining preceding, following, and parallel in one pattern
                    context_direct_pattern = nx.compose(preceding_pattern, following_pattern)
                    context_direct_pattern = nx.compose(context_direct_pattern, parallel_pattern)

                    if len(parallel_pattern.nodes) > 0:
                        for node in parallel_pattern_nodes:
                            for out_node in out_pattern_nodes:
                                context_direct_pattern.add_edge(node, out_node, eventually=False)
                            for in_node in in_pattern_nodes:
                                context_direct_pattern.add_edge(in_node, node, eventually=False)

                    if Direct_successor or Direct_predecessor or Concurrence:
                        if (len(parallel_pattern.nodes) > 0 and (
                                len(preceding_pattern.nodes) > 0 or len(following_pattern.nodes) > 0)) \
                                or (len(preceding_pattern.nodes) > 0 and len(following_pattern.nodes) > 0):
                            embedded_trace_graph = create_embedded_pattern_in_trace(
                                set(context_direct_pattern.nodes),
                                Trace_graph)
                            Extended_patterns_at_stage, new_Pattern_IDs = update_pattern_dict(
                                Extended_patterns_at_stage,
                                context_direct_pattern,
                                embedded_trace_graph,
                                case_data, case_id, nm, em, Core_activity,
                                new_patterns_for_core)
                            if new_Pattern_IDs != "":
                                new_patterns_for_core.append(new_Pattern_IDs)
                    else:
                        embedded_trace_graph = create_embedded_pattern_in_trace(set(context_direct_pattern.nodes),
                                                                                Trace_graph)
                        Extended_patterns_at_stage, new_Pattern_IDs = update_pattern_dict(Extended_patterns_at_stage,
                                                                                          context_direct_pattern,
                                                                                          embedded_trace_graph,
                                                                                          case_data, case_id, nm, em,
                                                                                          Core_activity,
                                                                                          new_patterns_for_core)
                        if new_Pattern_IDs != "":
                            new_patterns_for_core.append(new_Pattern_IDs)

    patient_data[new_patterns_for_core] = 0
    All_extended_patterns_1_list.extend(new_patterns_for_core)
    for PID in new_patterns_for_core:
        for CaseID in np.unique(Extended_patterns_at_stage[PID]['Instances']['case']):
            variant_frequency_case = Extended_patterns_at_stage[PID]['Instances']['case'].count(CaseID)
            Other_cases = selected_variants.loc[selected_variants[case_id] == CaseID, 'case:CaseIDs'].tolist()[0]
            for Ocase in Other_cases:
                patient_data.loc[patient_data[case_id] == Ocase, PID] = variant_frequency_case

result_dict = {'K': [], 'N': [], 'Pareto': [], 'All': []}
for obj in pareto_features:
    result_dict[obj] = []
for fold in range(len(train_list)):
    train_patient_data = patient_data[patient_data[case_id].isin(train_list[fold + 1])]
    test_patient_data = patient_data[patient_data[case_id].isin(test_list[fold + 1])]

    pattern_attributes = create_pattern_attributes_IKNL_auto(train_patient_data, label_col, label_class,
                                                             False, All_extended_patterns_1_list,
                                                             pairwise_distances_array, pair_cases, start_search_points)
    Objectives_attributes = pattern_attributes[pareto_features]
    if 'OutcomeCorrelation' in pareto_features:
        Objectives_attributes['OutcomeCorrelation'] = np.abs(Objectives_attributes['OutcomeCorrelation'])

    mask = paretoset(Objectives_attributes, sense=pareto_sense)
    paretoset_patterns = pattern_attributes[mask]

    Top_K = len(paretoset_patterns)

    result_dict['K'].append(Top_K)
    result_dict['N'].append(len(pattern_attributes))

    All_extended_patterns[fold]['All'].extend(list(pattern_attributes['patterns']))
    for jj in range(len(pareto_features) + 1):
        if jj == len(pareto_features):
            selected_P = paretoset_patterns
            Obj = 'Pareto'
        else:
            if pareto_sense[jj] == 'max':
                selected_P = pattern_attributes.sort_values(by=[pareto_features[jj]], ascending=False).head(Top_K)
            else:
                selected_P = pattern_attributes.sort_values(by=[pareto_features[jj]], ascending=True).head(Top_K)

            Obj = pareto_features[jj]

        All_extended_patterns[fold][Obj].extend(list(selected_P['patterns']))

    train_y = train_patient_data[label_class]
    test_y = test_patient_data[label_class]
    train_y = train_y.astype(str)
    test_y = test_y.astype(str)
    for Obj in All_extended_patterns[fold]:
        train_X = train_patient_data[All_extended_patterns[fold][Obj]]
        test_X = test_patient_data[All_extended_patterns[fold][Obj]]

        ML_model.fit(train_X, train_y)
        predicted = ML_model.predict(test_X)

        result_dict[Obj].append(f1_score(test_y, predicted, average='weighted'))

kFold_results = dict()
for obj in result_dict:
    kFold_results[obj] = np.mean(result_dict[obj])
    kFold_results[obj + "_std"] = np.std(result_dict[obj])

results_dataframe = results_dataframe.append(kFold_results, ignore_index=True)
results_dataframe.at[1, 'Extension_step'] = 1
results_dataframe.to_csv(folder_address + 'Auto_Results.csv', index=False)
print('1st extension is done!')

for ext in range(1, extension_numbers):
    print("extension number %s " % (ext + 1))
    Extension_2_patterns_list, Extended_patterns_at_stage, patient_data = \
        Pattern_Extender(Extended_patterns_at_stage, patient_data, EventLog_graphs, all_variants, Eventual_window)

    result_dict = {'K': [], 'N': [], 'Pareto': [], 'All': []}
    for obj in pareto_features:
        result_dict[obj] = []

    for fold in range(len(train_list)):
        train_patient_data = patient_data[patient_data[case_id].isin(train_list[fold + 1])]
        test_patient_data = patient_data[patient_data[case_id].isin(test_list[fold + 1])]

        pattern_attributes = create_pattern_attributes_IKNL_auto(train_patient_data, label_col, label_class,
                                                                 False, Extension_2_patterns_list,
                                                                 pairwise_distances_array, pair_cases, start_search_points)
        Objectives_attributes = pattern_attributes[pareto_features]
        if 'OutcomeCorrelation' in pareto_features:
            Objectives_attributes['OutcomeCorrelation'] = np.abs(Objectives_attributes['OutcomeCorrelation'])

        mask = paretoset(Objectives_attributes, sense=pareto_sense)
        paretoset_patterns = pattern_attributes[mask]

        Top_K = len(paretoset_patterns)

        result_dict['K'].append(Top_K)
        result_dict['N'].append(len(pattern_attributes))

        All_extended_patterns[fold]['All'].extend(list(pattern_attributes['patterns']))
        for jj in range(len(pareto_features) + 1):
            if jj == len(pareto_features):
                selected_P = paretoset_patterns
                Obj = 'Pareto'
            else:
                if pareto_sense[jj] == 'max':
                    selected_P = pattern_attributes.sort_values(by=[pareto_features[jj]], ascending=False).head(Top_K)
                else:
                    selected_P = pattern_attributes.sort_values(by=[pareto_features[jj]], ascending=True).head(Top_K)

                Obj = pareto_features[jj]

            All_extended_patterns[fold][Obj].extend(list(selected_P['patterns']))

        train_y = train_patient_data[label_class]
        test_y = test_patient_data[label_class]
        train_y = train_y.astype(str)
        test_y = test_y.astype(str)
        for Obj in All_extended_patterns[fold]:
            train_X = train_patient_data[All_extended_patterns[fold][Obj]]
            test_X = test_patient_data[All_extended_patterns[fold][Obj]]

            ML_model.fit(train_X, train_y)
            predicted = ML_model.predict(test_X)

            result_dict[Obj].append(f1_score(test_y, predicted, average='weighted'))

    kFold_results = dict()
    for obj in result_dict:
        kFold_results[obj] = np.mean(result_dict[obj])
        kFold_results[obj + "_std"] = np.std(result_dict[obj])

    results_dataframe = results_dataframe.append(kFold_results, ignore_index=True)
    results_dataframe.at[ext + 1, 'Extension_step'] = ext + 1
    results_dataframe.to_csv(folder_address + 'Auto_Results.csv', index=False)

results_dataframe.to_csv(folder_address + 'Auto_Results.csv', index=False)

print('done')
