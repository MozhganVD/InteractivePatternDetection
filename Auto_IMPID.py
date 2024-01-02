from paretoset import paretoset
import networkx as nx
from networkx.readwrite import json_graph
import json
import numpy as np
from sklearn.model_selection import train_test_split
from IMIPD import create_pattern_attributes, Trace_graph_generator, Pattern_extension, Single_Pattern_Extender

def AutoStepWise_PPD(Max_extension_step, Max_gap_between_events, test_data_percentage, data, patient_data,
                     pairwise_distances_array, pair_cases, start_search_points, case_id, activity, outcome,
                     outcome_type,
                     timestamp,
                     pareto_features, pareto_sense, d_time, color_act_dict, save_path):

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

            Extended_patterns_at_stage, new_patterns_for_core = Pattern_extension(case_data, Trace_graph, Core_activity,
                                                                                  case_id, Extended_patterns_at_stage,
                                                                                  Max_gap_between_events,
                                                                                  new_patterns_for_core)

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

    All_extended_patterns_dict = Extended_patterns_at_stage.copy()
    for ext in range(1, Max_extension_step):
        print("extension number %s " % (ext + 1))
        new_patterns_per_extension = []
        for chosen_pattern_ID in paretoset_patterns['patterns']:
            new_patterns_for_core = []
            if any(nx.get_edge_attributes(All_extended_patterns_dict[chosen_pattern_ID]['pattern'],
                                          'eventually').values()):
                continue

            All_extended_patterns_dict, Extended_patterns_at_stage, patient_data = Single_Pattern_Extender(
                All_extended_patterns_dict,
                chosen_pattern_ID,
                patient_data, EventLog_graphs,
                data, Max_gap_between_events, activity, case_id)

            new_patterns_per_extension.extend(Extended_patterns_at_stage.keys())

        # Extension_2_patterns_list, Extended_patterns_at_stage, patient_data = \
        #     Pattern_Extender(Extended_patterns_at_stage, patient_data, EventLog_graphs, data, case_id, activity)

        train_patient_data = patient_data[patient_data[case_id].isin(train_data[case_id])]
        test_patient_data = patient_data[patient_data[case_id].isin(test_data[case_id])]

        pattern_attributes = create_pattern_attributes(train_patient_data, outcome,
                                                       new_patterns_per_extension,
                                                       pairwise_distances_array, pair_cases, start_search_points,
                                                       outcome_type)

        Objectives_attributes = pattern_attributes[pareto_features]
        mask = paretoset(Objectives_attributes, sense=pareto_sense)
        paretoset_patterns = pattern_attributes[mask]
        All_extended_patterns.extend(list(paretoset_patterns['patterns']))

        # save all patterns in paretofront in json format
        for pattern in paretoset_patterns['patterns']:
            P_graph = All_extended_patterns_dict[pattern]['pattern']
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
