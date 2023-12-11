from paretoset import paretoset
import networkx as nx
from networkx.readwrite import json_graph
import json
import numpy as np
import networkx.algorithms.isomorphism as iso
from sklearn.model_selection import train_test_split
from IMIPD import create_pattern_attributes, Trace_graph_generator
from tools import create_embedded_pattern_in_trace, update_pattern_dict, Pattern_Extender


def AutoStepWise_PPD(Max_extension_step, Max_gap_between_events, test_data_percentage, data, patient_data,
                     pairwise_distances_array, pair_cases, start_search_points, case_id, activity, outcome,
                     outcome_type,
                     timestamp,
                     pareto_features, pareto_sense, d_time, color_act_dict, save_path):
    Direct_predecessor = True
    Direct_successor = True
    Direct_context = True
    Concurrence = True
    # Unique_cases = patient_data.drop_duplicates(subset=[case_id], keep='first')
    # split test and train data
    if outcome_type == 'binary':
        train, test = train_test_split(patient_data, test_size=test_data_percentage, random_state=42,
                                                 stratify=patient_data[outcome])
    else:
        train, test = train_test_split(patient_data, test_size=test_data_percentage, random_state=42)

    # Filter the data based on case_id
    train_ids, test_ids = train[case_id], test[case_id]
    train_data = patient_data[patient_data[case_id].isin(train_ids)]
    test_data = patient_data[patient_data[case_id].isin(test_ids)]

    All_extended_patterns = []
    activity_attributes = create_pattern_attributes(train_data, outcome,
                                                    list(data[activity].unique()),
                                                    pairwise_distances_array, pair_cases,
                                                    start_search_points, outcome_type)

    Objectives_attributes = activity_attributes[pareto_features]
    mask = paretoset(Objectives_attributes, sense=pareto_sense)
    paretoset_activities = activity_attributes[mask]
    All_extended_patterns.extend(list(paretoset_activities['patterns']))

    train_X = train_data[All_extended_patterns]
    test_X = test_data[All_extended_patterns]
    train_X['Case_ID'] = train_data[case_id]
    test_X['Case_ID'] = test_data[case_id]
    train_X['Outcome'] = train_data[outcome]
    test_X['Outcome'] = test_data[outcome]

    Extended_patterns_at_stage = dict()
    All_extended_patterns_1_list = []
    EventLog_graphs = dict()
    for Core_activity in All_extended_patterns:
        filtered_cases = data.loc[data[activity] == Core_activity, case_id]
        filtered_main_data = data[data[case_id].isin(filtered_cases)]
        new_patterns_for_core = []
        for case in filtered_main_data[case_id].unique():
            case_data = filtered_main_data[filtered_main_data[case_id] == case]

            if case not in EventLog_graphs.keys():
                Trace_graph = Trace_graph_generator(filtered_main_data, d_time,
                                                    case, color_act_dict, case_id, activity, timestamp)

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
                            Extended_patterns_at_stage, new_Pattern_IDs = update_pattern_dict(
                                Extended_patterns_at_stage,
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
                            Extended_patterns_at_stage, new_Pattern_IDs = update_pattern_dict(
                                Extended_patterns_at_stage,
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
                            Extended_patterns_at_stage, new_Pattern_IDs = update_pattern_dict(
                                Extended_patterns_at_stage,
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
                            Extended_patterns_at_stage, new_Pattern_IDs = update_pattern_dict(
                                Extended_patterns_at_stage,
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
                patient_data.loc[patient_data[case_id] == CaseID, PID] = variant_frequency_case

    new_train_data = patient_data[patient_data[case_id].isin(train_data[case_id])]
    new_test_data = patient_data[patient_data[case_id].isin(test_data[case_id])]

    pattern_attributes = create_pattern_attributes(new_train_data, outcome,
                                                   All_extended_patterns_1_list,
                                                   pairwise_distances_array, pair_cases, start_search_points,
                                                   outcome_type)
    Objectives_attributes = pattern_attributes[pareto_features]
    mask = paretoset(Objectives_attributes, sense=pareto_sense)
    paretoset_patterns = pattern_attributes[mask]
    All_extended_patterns.extend(list(paretoset_patterns['patterns']))

    # save all patterns in paretofront in json format
    for pattern in paretoset_patterns['patterns']:
        P_graph = Extended_patterns_at_stage[pattern]['pattern']
        P_data = json_graph.node_link_data(P_graph)
        json_string = json.dumps(P_data)
        with open(save_path + '/%s.json' % pattern, "w") as file:
            file.write(json_string)

    train_X = new_train_data[All_extended_patterns]
    test_X = new_test_data[All_extended_patterns]
    train_X['Case_ID'] = new_train_data[case_id]
    test_X['Case_ID'] = new_test_data[case_id]
    train_X['Outcome'] = new_train_data[outcome]
    test_X['Outcome'] = new_test_data[outcome]

    for ext in range(1, Max_extension_step):
        print("extension number %s " % (ext + 1))
        Extension_2_patterns_list, Extended_patterns_at_stage, patient_data = \
            Pattern_Extender(Extended_patterns_at_stage, patient_data, EventLog_graphs, data, case_id, activity)

        train_patient_data = patient_data[patient_data[case_id].isin(train_data[case_id])]
        test_patient_data = patient_data[patient_data[case_id].isin(test_data[case_id])]

        pattern_attributes = create_pattern_attributes(train_patient_data, outcome,
                                                       Extension_2_patterns_list,
                                                       pairwise_distances_array, pair_cases, start_search_points,
                                                       outcome_type)

        Objectives_attributes = pattern_attributes[pareto_features]
        mask = paretoset(Objectives_attributes, sense=pareto_sense)
        paretoset_patterns = pattern_attributes[mask]
        All_extended_patterns.extend(list(paretoset_patterns['patterns']))

        # save all patterns in paretofront in json format
        for pattern in paretoset_patterns['patterns']:
            P_graph = Extended_patterns_at_stage[pattern]['pattern']
            P_data = json_graph.node_link_data(P_graph)
            json_string = json.dumps(P_data)
            with open(save_path + '/%s.json' % pattern, "w") as file:
                file.write(json_string)

        train_X = train_patient_data[All_extended_patterns]
        test_X = test_patient_data[All_extended_patterns]
        train_X['Case_ID'] = train_patient_data[case_id]
        test_X['Case_ID'] = test_patient_data[case_id]
        train_X['Outcome'] = train_patient_data[outcome]
        test_X['Outcome'] = test_patient_data[outcome]

    return train_X, test_X
