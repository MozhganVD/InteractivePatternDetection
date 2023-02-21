from paretoset import paretoset
import pandas as pd
import networkx as nx
import pm4py
import numpy as np
import seaborn as sb
import networkx.algorithms.isomorphism as iso
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.obj import EventLog
import matplotlib.pyplot as plt
from pylab import figure
import random
import os
from tools import create_pattern_attributes_BPIC12, threeD_ploting, create_embedded_pattern_in_trace, \
    plot_patterns, Trace_graph_generator, update_pattern_dict
import matplotlib

# matplotlib.use('Qt5Agg')

random.seed(123)

folder_address = "./datasets/BPIC2012/"
dataset_name = "bpic_acc_trun40_allActs.csv"
extension_numbers = 6
case_id = "case:concept:name"
activities = "concept:name"
label_col = None
label_class = 'label'
timestamp = 'Complete Timestamp'
Only_Complete = True

Folds = 5
# select detectors:
Direct_context = True
Concurrence = True
Direct_predecessor = True
Direct_successor = True

# dataset settings
num_col = ['AMOUNT_REQ']
binary_col = []
cat_col = []

d_time = 1  # in seconds
visualization_features = ['OutcomeCorrelation', 'Case_Coverage', 'Case_Distance']
pareto_features = ['OutcomeCorrelation', 'Case_Coverage', 'Case_Distance']  # , 'Case_Distance'
pareto_sense = ["max", "max", "min"]  # , "min"

# read dataset
main_data = pd.read_csv(folder_address + dataset_name, sep=',')

main_data.rename(columns={'Case ID': case_id, 'Activity': activities}, inplace=True)

main_data.loc[main_data[label_class] == 'deviant', label_class] = 1
main_data.loc[main_data[label_class] == 'regular', label_class] = 0

main_data[activities] = main_data[activities].map(lambda x: x.split('-')[0])
if Only_Complete:
    main_data = main_data[main_data['lifecycle:transition'] == 'COMPLETE']

main_data[activities] = main_data[activities].str.replace("_", "-")
main_data[timestamp] = pd.to_datetime(main_data[timestamp])

main_data[case_id] = main_data[case_id].astype(str)

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
patient_data[[case_id, label_class]] = main_data[[case_id, label_class]]
patient_data = patient_data.drop_duplicates(subset=[case_id], keep='first')

patient_data.sort_values(by=case_id, inplace=True)
patient_data.reset_index(inplace=True, drop=True)

# Keep only variants and its frequency
filtered_main_data = pm4py.format_dataframe(main_data, case_id=case_id, activity_key=activities,
                                            timestamp_key=timestamp)
timestamp = 'Complete Timestamp'
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
    Other_cases = selected_variants.loc[selected_variants[case_id] == case, 'case:CaseIDs'].tolist()[0]
    trace = main_data.loc[main_data[case_id] == case, activities].tolist()
    for act in np.unique(trace):
        Number_of_act = trace.count(act)
        for Ocase in Other_cases:
            patient_data.loc[patient_data[case_id] == Ocase, act] = Number_of_act

activity_attributes = create_pattern_attributes_BPIC12(patient_data, label_col, label_class,
                                                       None, list(main_data[activities].unique()))

Objectives_attributes = activity_attributes[pareto_features]
if 'OutcomeCorrelation' in pareto_features:
    Objectives_attributes['OutcomeCorrelation'] = np.abs(Objectives_attributes['OutcomeCorrelation'])

mask = paretoset(Objectives_attributes, sense=pareto_sense)
paretoset_activities = activity_attributes[mask]
threeD_ploting(paretoset_activities,
               visualization_features[0],
               visualization_features[1],
               visualization_features[2], folder_address,
               color_act_dict=color_act_dict, activity_level=True)

Extending_Core = True
all_pattern_dictionary = dict()
All_Pareto_front = dict()
EventLog_graphs = dict()
while Extending_Core:
    for pattern in activity_attributes['patterns']:
        print(pattern)
    Core_activity = input('insert the name of your desired treatment as a Core pattern: ')
    Patterns_Dictionary = dict()
    filtered_cases = main_data.loc[main_data[activities] == Core_activity, case_id]
    filtered_main_data = main_data[main_data[case_id].isin(filtered_cases)]

    # Keep only variants and its frequency
    timestamp = 'Complete Timestamp'
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
    timestamp = 'time:timestamp'

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
                        Patterns_Dictionary, _ = update_pattern_dict(Patterns_Dictionary, preceding_pattern,
                                                                     embedded_trace_graph,
                                                                     case_data, case_id, nm, em, Core_activity)
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
                        Patterns_Dictionary, _ = update_pattern_dict(Patterns_Dictionary, following_pattern,
                                                                     embedded_trace_graph,
                                                                     case_data, case_id, nm, em, Core_activity)
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
                        Patterns_Dictionary, _ = update_pattern_dict(Patterns_Dictionary, parallel_pattern,
                                                                     embedded_trace_graph,
                                                                     case_data, case_id, nm, em, Core_activity)
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
                            embedded_trace_graph = create_embedded_pattern_in_trace(set(context_direct_pattern.nodes),
                                                                                    Trace_graph)
                            Patterns_Dictionary, _ = update_pattern_dict(Patterns_Dictionary, context_direct_pattern,
                                                                         embedded_trace_graph,
                                                                         case_data, case_id, nm, em, Core_activity)
                    else:
                        embedded_trace_graph = create_embedded_pattern_in_trace(set(context_direct_pattern.nodes),
                                                                                Trace_graph)
                        Patterns_Dictionary, _ = update_pattern_dict(Patterns_Dictionary, context_direct_pattern,
                                                                     embedded_trace_graph,
                                                                     case_data, case_id, nm, em, Core_activity)

    patient_data[list(Patterns_Dictionary.keys())] = 0
    for PID in Patterns_Dictionary:
        for CaseID in np.unique(Patterns_Dictionary[PID]['Instances']['case']):
            variant_frequency_case = Patterns_Dictionary[PID]['Instances']['case'].count(CaseID)
            Other_cases = selected_variants.loc[selected_variants[case_id] == CaseID, 'case:CaseIDs'].tolist()[0]
            for Ocase in Other_cases:
                patient_data.loc[patient_data[case_id] == Ocase, PID] = variant_frequency_case

    pattern_attributes = create_pattern_attributes_BPIC12(patient_data, label_col, label_class,
                                                          Core_activity, list(Patterns_Dictionary.keys()))

    Objectives_attributes = pattern_attributes[pareto_features]
    if 'OutcomeCorrelation' in pareto_features:
        Objectives_attributes['OutcomeCorrelation'] = np.abs(Objectives_attributes['OutcomeCorrelation'])

    mask = paretoset(Objectives_attributes, sense=pareto_sense)
    paretoset_patterns = pattern_attributes[mask]

    All_Pareto_front[Core_activity] = dict()
    All_Pareto_front[Core_activity]['dict'] = Patterns_Dictionary
    All_Pareto_front[Core_activity]['variants'] = selected_variants

    threeD_ploting(paretoset_patterns, visualization_features[0],
                   visualization_features[1],
                   visualization_features[2], folder_address)

    for pattern_id in paretoset_patterns['patterns']:
        plot_patterns(Patterns_Dictionary, pattern_id, color_act_dict, pattern_attributes, folder_address)

    all_pattern_dictionary.update(Patterns_Dictionary)
    repeat = input('do you want to select another core activity? (yes/no)').lower()
    if repeat == 'no':
        Extending_Core = False

## extending initial discovered patterns
Extension = input('do you want to extend one pattern? (yes/no)').lower()
while Extension == 'yes':
    for Pat in all_pattern_dictionary.keys():
        print(Pat)
    chosen_pattern_ID = input('enter the ID of one pattern to extend: ')
    core = chosen_pattern_ID.split("_")[0]
    selected_variants = All_Pareto_front[core]['variants']

    while any(nx.get_edge_attributes(all_pattern_dictionary[chosen_pattern_ID]['pattern'], 'eventually').values()):
        print('WARNING: patterns containing eventually relations cannot be selected for extension')
        chosen_pattern_ID = input('enter the ID of one pattern to extend: ')

    chosen_case_list = all_pattern_dictionary[chosen_pattern_ID]['Instances']['case']

    Extended_pattern_dictionary = dict()
    extended_ID = 0
    for idx, case in enumerate(all_pattern_dictionary[chosen_pattern_ID]['Instances']['case']):
        Trace_graph = EventLog_graphs[case].copy()
        nodes_values = [Trace_graph._node[n]['value'] for n in Trace_graph.nodes]

        embedded_trace_graph = all_pattern_dictionary[chosen_pattern_ID]['Instances']['emb_trace'][idx]
        inside_pattern_nodes = set(Trace_graph.nodes).difference(set(embedded_trace_graph.nodes))
        to_remove = set(Trace_graph.nodes).difference(inside_pattern_nodes)
        chosen_pattern = Trace_graph.copy()
        chosen_pattern.remove_nodes_from(to_remove)

        ending_nodes = {n[0] for n in chosen_pattern.out_degree if n[1] == 0}
        starting_nodes = {n[0] for n in chosen_pattern.in_degree if n[1] == 0}

        case_data = selected_variants[selected_variants[case_id] == case]
        values = nx.get_node_attributes(Trace_graph, 'value')
        parallel = nx.get_node_attributes(Trace_graph, 'parallel')
        color = nx.get_node_attributes(Trace_graph, 'color')

        nm = iso.categorical_node_match("value", nodes_values)
        em = iso.categorical_node_match("eventually", [True, False])

        # preceding extension
        in_pattern_nodes = set(embedded_trace_graph.pred['pattern'].keys())
        if len(in_pattern_nodes) > 0:
            extended_pattern = chosen_pattern.copy()
            in_pattern_values = [values[n] for n in in_pattern_nodes]
            for in_node in in_pattern_nodes:
                extended_pattern.add_node(in_node,
                                          value=values[in_node], parallel=parallel[in_node], color=color[in_node])
                for node in starting_nodes:
                    extended_pattern.add_edge(in_node, node, eventually=False)

            new_embedded_trace_graph = create_embedded_pattern_in_trace(set(extended_pattern.nodes), Trace_graph)
            Extended_pattern_dictionary, _ = update_pattern_dict(Extended_pattern_dictionary, extended_pattern,
                                                                 new_embedded_trace_graph,
                                                                 case_data, case_id, nm, em, chosen_pattern_ID)

        # following extension
        out_pattern_nodes = set(embedded_trace_graph.succ['pattern'].keys())
        if len(out_pattern_nodes) > 0:
            extended_pattern = chosen_pattern.copy()
            out_pattern_values = [values[n] for n in out_pattern_nodes]
            for out_node in out_pattern_nodes:
                extended_pattern.add_node(out_node,
                                          value=values[out_node], parallel=parallel[out_node],
                                          color=color[out_node])
                for node in ending_nodes:
                    extended_pattern.add_edge(node, out_node, eventually=False)

            new_embedded_trace_graph = create_embedded_pattern_in_trace(set(extended_pattern.nodes), Trace_graph)
            Extended_pattern_dictionary, _ = update_pattern_dict(Extended_pattern_dictionary, extended_pattern,
                                                                 new_embedded_trace_graph,
                                                                 case_data, case_id, nm, em, chosen_pattern_ID)

        ## all non-direct nodes
        Eventual_relations_nodes = set(embedded_trace_graph.nodes).difference(
            in_pattern_nodes.union(out_pattern_nodes))
        Eventual_relations_nodes.remove('pattern')
        # Eventually following patterns
        if len(out_pattern_nodes) > 0:
            Eventual_following_nodes = {node for node in Eventual_relations_nodes if node > max(out_pattern_nodes)}
            for Ev_F_nodes in Eventual_following_nodes:
                Eventual_follow_pattern = chosen_pattern.copy()
                Eventual_follow_pattern.add_node(Ev_F_nodes,
                                                 value=values[Ev_F_nodes], parallel=parallel[Ev_F_nodes],
                                                 color=color[Ev_F_nodes])
                for node in ending_nodes:
                    Eventual_follow_pattern.add_edge(node, Ev_F_nodes, eventually=True)

                Extended_pattern_dictionary, _ = update_pattern_dict(Extended_pattern_dictionary,
                                                                     Eventual_follow_pattern, [],
                                                                     case_data, case_id, nm, em, chosen_pattern_ID)

        # Eventually preceding patterns
        if len(in_pattern_nodes) > 0:
            Eventual_preceding_nodes = {node for node in Eventual_relations_nodes if node < min(in_pattern_nodes)}
            for Ev_P_nodes in Eventual_preceding_nodes:
                Eventual_preceding_pattern = chosen_pattern.copy()
                Eventual_preceding_pattern.add_node(Ev_P_nodes,
                                                    value=values[Ev_P_nodes], parallel=parallel[Ev_P_nodes],
                                                    color=color[Ev_P_nodes])
                for node in starting_nodes:
                    Eventual_preceding_pattern.add_edge(Ev_P_nodes, node, eventually=True)

                Extended_pattern_dictionary, _ = update_pattern_dict(Extended_pattern_dictionary,
                                                                     Eventual_preceding_pattern, [],
                                                                     case_data, case_id, nm, em, chosen_pattern_ID)

    if len(Extended_pattern_dictionary) == 0:
        print("no pattern with chosen target activity is found!")
        continue
    else:
        patient_data[list(Extended_pattern_dictionary.keys())] = 0
        for PID in Extended_pattern_dictionary:
            for CaseID in np.unique(Extended_pattern_dictionary[PID]['Instances']['case']):
                variant_frequency_case = Extended_pattern_dictionary[PID]['Instances']['case'].count(CaseID)
                Other_cases = selected_variants.loc[selected_variants[case_id] == CaseID, 'case:CaseIDs'].tolist()[0]
                for Ocase in Other_cases:
                    patient_data.loc[patient_data[case_id] == Ocase, PID] = variant_frequency_case

        pattern_attributes = create_pattern_attributes_BPIC12(patient_data, label_col, label_class,
                                                              core,
                                                              list(Extended_pattern_dictionary.keys()))

        Objectives_attributes = pattern_attributes[pareto_features]
        if 'OutcomeCorrelation' in pareto_features:
            Objectives_attributes['OutcomeCorrelation'] = np.abs(Objectives_attributes['OutcomeCorrelation'])

        mask = paretoset(Objectives_attributes, sense=pareto_sense)
        paretoset_patterns = pattern_attributes[mask]

        threeD_ploting(paretoset_patterns, visualization_features[0],
                       visualization_features[1],
                       visualization_features[2], folder_address)

        for pattern_id in paretoset_patterns['patterns']:
            plot_patterns(Extended_pattern_dictionary, pattern_id, color_act_dict, pattern_attributes, folder_address)

        all_pattern_dictionary.update(Extended_pattern_dictionary)

    Extension = input('do you want to extend one pattern? (yes/no)').lower()

print('done')
