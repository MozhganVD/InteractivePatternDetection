import pandas as pd
import networkx as nx
import pm4py
import numpy as np
import networkx.algorithms.isomorphism as iso
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.obj import EventLog
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from tools import create_embedded_pattern_in_trace, update_pattern_dict


def VariantSelection(main_data, case_id, activities, timestamp):
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

    return selected_variants


def similarity_measuring_patterns(patterns_data, patient_data, pair_cases, start_search_points,
                                  pairwise_distances_array, Core_activity=False):
    # if Core_activity is not None:
    #     patient_data = patient_data[patient_data[Core_activity] != 0]

    for pattern in patterns_data['patterns']:
        in_pattern_cases = patient_data[patient_data[pattern] > 0].index
        out_pattern_cases = patient_data[patient_data[pattern] == 0].index
        # in_pattern_pair_cases = [(a, b) for idx, a in enumerate(in_pattern_cases) for b in in_pattern_cases[idx + 1:]]
        in_out_pattern_pair_cases = []
        for a in in_pattern_cases:
            for b in out_pattern_cases:
                if a < b:
                    in_out_pattern_pair_cases.append((a, b))
                else:
                    in_out_pattern_pair_cases.append((b, a))

        # for item in in_pattern_pair_cases:
        #     selected_pair_index.append(pair_cases.index(item, start_search_points[item[0]]))
        selected_pair_index = []
        for item in in_out_pattern_pair_cases:
            selected_pair_index.append(pair_cases.index(item, start_search_points[item[0]]))

        patterns_data.loc[patterns_data['patterns'] == pattern, 'Case_Distance_Interest'] = \
            np.mean([pairwise_distances_array[ind] for ind in selected_pair_index])

    return patterns_data


def predictive_measuring_patterns(patterns_data, patient_data, label_class, Core_activity=None):
    # if Core_activity is not None:
    #     patient_data = patient_data[patient_data[Core_activity] != 0]

    x = patient_data[patterns_data['patterns']]
    cat_y_all = patient_data[label_class]

    info_gain = mutual_info_classif(x, cat_y_all, discrete_features=True)
    patterns_data['Outcome_Interest'] = info_gain.reshape(-1, 1)

    # for pattern in patterns_data['patterns']:
    #
    #     cat_y = list(patient_data.loc[patient_data[pattern] > 0, label_class])
    #     cat_y_out = list(patient_data.loc[patient_data[pattern] == 0, label_class])
    #
    #     patterns_data.loc[patterns_data['patterns'] == pattern,
    #                       "PositiveOutcome_rate_pattern"] = np.sum(cat_y) / len(cat_y)
    #     patterns_data.loc[patterns_data['patterns'] == pattern,
    #                       "PositiveOutcome_rate_anti-pattern"] = np.sum(cat_y_out) / len(cat_y_out)
    #
    #     cor, _ = spearmanr(cat_y_all, x[pattern])
    #     if np.isnan(cor):
    #         patterns_data.loc[patterns_data['patterns'] == pattern, 'Outcome_Correlation'] = 0
    #         patterns_data.loc[patterns_data['patterns'] == pattern, 'p_values'] = np.nan
    #     else:
    #         patterns_data.loc[patterns_data['patterns'] == pattern, 'Outcome_Correlation'], _ = \
    #             spearmanr(cat_y_all, x[pattern])
    #
    #         _, patterns_data.loc[patterns_data['patterns'] == pattern, 'p_values'] = \
    #             spearmanr(cat_y_all, x[pattern])

    return patterns_data


# def frequency_measuring_patterns(patterns_data, pattern_list, patient_data, Core_activity):
#     for pattern in pattern_list:
#         patterns_data.loc[patterns_data['patterns'] == pattern, 'Pattern_Frequency'] \
#             = np.sum(patient_data[pattern])
#
#         patterns_data.loc[patterns_data['patterns'] == pattern, 'Case_Support'] = \
#             np.count_nonzero(patient_data[pattern])
#
#         patterns_data.loc[patterns_data['patterns'] == pattern, 'Frequency_Interest'] = \
#             np.count_nonzero(patient_data[pattern]) / len(patient_data)
#
#     return patterns_data
def frequency_measuring_patterns(patterns_data, pattern_list, patient_data, Core_activity):
    pattern_indices = patterns_data.set_index('patterns').index
    for pattern in pattern_list:
        pattern_index = pattern_indices.get_loc(pattern)

        pattern_occurrences = patient_data[pattern]
        pattern_support = np.count_nonzero(pattern_occurrences)
        pattern_frequency_interest = pattern_support / len(patient_data)

        patterns_data.at[pattern_index, 'Pattern_Frequency'] = np.sum(pattern_occurrences)
        patterns_data.at[pattern_index, 'Case_Support'] = pattern_support
        patterns_data.at[pattern_index, 'Frequency_Interest'] = pattern_frequency_interest

    return patterns_data


def create_pattern_frame(pattern_list):
    patterns_data = pd.DataFrame(columns=pattern_list)
    patterns_data = patterns_data.transpose()
    patterns_data['patterns'] = patterns_data.index
    patterns_data.reset_index(inplace=True, drop=True)
    return patterns_data


def create_pattern_attributes(patient_data, label_class, Core_activity, pattern_list,
                              pairwise_distances_array, pair_cases, start_search_points):
    patterns_data = create_pattern_frame(pattern_list)
    ## Frequency-based measures
    patterns_data = frequency_measuring_patterns(patterns_data, pattern_list, patient_data, Core_activity)
    print('frequency measures done')

    ## Discriminative measures
    patterns_data = predictive_measuring_patterns(patterns_data, patient_data, label_class, Core_activity)
    print('predictive measures done')

    ## Similarity measures
    patterns_data = similarity_measuring_patterns(patterns_data, patient_data, pair_cases, start_search_points,
                                                  pairwise_distances_array, Core_activity)
    print('similarity measures done')

    return patterns_data


def calculate_pairwise_case_distance(X_features, num_col):
    Cat_exists = False
    Num_exists = False

    cat_col = [c for c in X_features.columns if c not in num_col]
    le = LabelEncoder()
    for col in cat_col:
        X_features[col] = le.fit_transform(X_features[col])

    if len(cat_col) > 0:
        Cat_exists = True
        cat_dist = pdist(X_features[cat_col].values, 'jaccard')

    if len(num_col) > 0:
        Num_exists = True
        numeric_dist = pdist(X_features[num_col].values, 'euclid')
        normalizer = preprocessing.MinMaxScaler()
        x = numeric_dist.reshape((-1, 1))
        numeric_dist = normalizer.fit_transform(x)
        del x
        numeric_dist = numeric_dist.reshape(len(numeric_dist))

    if Cat_exists and Num_exists:
        Combined_dist = ((len(cat_col) * cat_dist) + numeric_dist) / (1 + len(cat_col))
        return Combined_dist

    elif Cat_exists and not Num_exists:
        return cat_dist

    elif Num_exists and not Cat_exists:
        return numeric_dist


def Pattern_extension(case_data, Trace_graph, Core_activity, case_id, Patterns_Dictionary,
                      Direct_predecessor=True, Direct_successor=True,
                      Direct_context=True, Concurrence=True):
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

    return Patterns_Dictionary


def Trace_graph_generator(selected_variants, patient_data, Core_activity, delta_time,
                          Case_ID, color_act_dict,
                          case_column, activity_column, timestamp):
    Trace_graph = nx.DiGraph()
    case_data = selected_variants[selected_variants[case_column] == Case_ID]

    for i, treatments in enumerate(case_data[activity_column]):
        Trace_graph.add_node(i, value=treatments, parallel=False, color=color_act_dict[treatments])

    trace = selected_variants.loc[selected_variants[case_column] == Case_ID, activity_column].tolist()
    # Number_of_core = trace.count(Core_activity)
    # patient_data.loc[patient_data[case_column] == Case_ID, Core_activity] = Number_of_core

    start_times = selected_variants.loc[selected_variants[case_column] == Case_ID, timestamp].tolist()
    parallel_indexes = {0: set()}
    max_key = max(parallel_indexes.keys())
    parallel = False
    for i in range(0, len(trace) - 1):
        if abs(start_times[i] - start_times[i + 1]).total_seconds() <= delta_time:
            parallel = True
            max_key = max(parallel_indexes.keys())
            parallel_indexes[max_key].add(i)
            parallel_indexes[max_key].add(i + 1)

            Trace_graph._node[i]['parallel'] = True
            Trace_graph._node[i + 1]['parallel'] = True
        else:
            if parallel:
                if max(parallel_indexes[max_key]) == i:
                    for ind in parallel_indexes[max_key]:
                        Trace_graph.add_edge(ind, i + 1, eventually=False)

                if min(parallel_indexes[max_key]) > 0:
                    if max_key > 0:
                        keys_set = list(parallel_indexes.keys())
                        keys_set.remove(max_key)
                        max_last_key = max(keys_set)
                        if len(parallel_indexes[max_last_key]) > 0:
                            for pind in parallel_indexes[max_last_key]:
                                for ind in parallel_indexes[max_key]:
                                    Trace_graph.add_edge(pind, ind, eventually=False)
                        else:
                            for ind in parallel_indexes[max_key]:
                                Trace_graph.add_edge(min(parallel_indexes[max_key]) - 1, ind, eventually=False)

                    else:
                        for ind in parallel_indexes[max_key]:
                            Trace_graph.add_edge(min(parallel_indexes[max_key]) - 1, ind, eventually=False)

            else:
                Trace_graph.add_edge(i, i + 1, eventually=False)

            parallel_indexes.update({i + 1: set()})
            parallel = False

    if parallel and min(parallel_indexes[max_key]) > 0:
        if max_key > 0:
            keys_set = list(parallel_indexes.keys())
            keys_set.remove(max_key)
            max_last_key = max(keys_set)
            if len(parallel_indexes[max_last_key]) > 0:
                for pind in parallel_indexes[max_last_key]:
                    for ind in parallel_indexes[max_key]:
                        Trace_graph.add_edge(pind, ind, eventually=False)
            else:
                for ind in parallel_indexes[max_key]:
                    Trace_graph.add_edge(min(parallel_indexes[max_key]) - 1, ind, eventually=False)
        else:
            for ind in parallel_indexes[max_key]:
                Trace_graph.add_edge(min(parallel_indexes[max_key]) - 1, ind, eventually=False)

    return Trace_graph


def plot_patterns(Patterns_Dictionary, pattern_id, color_act_dict, pattern_attributes, dim=(2, 3)):
    fig, ax = plt.subplots(dim[0], dim[1])
    # fig = figure(figsize=[8, 8])
    # ax = fig.add_subplot()

    pattern_features = pattern_attributes[pattern_attributes['patterns'] == pattern_id]
    info_text = "pattern:" + str(pattern_id) + "\n\n"
    for col in pattern_features:
        if col == 'patterns':
            continue
        elif col == 'not accepted/accepted_in' or col == 'not accepted/accepted_out':
            info_text += col + " : %s \n\n" % list(pattern_features[col])[0]

        elif col == 'Pattern_Frequency' or col == 'Case_Support':
            info_text += col + " : %d \n\n" % list(pattern_features[col])[0]
        else:
            info_text += col + " : %.3f \n\n" % list(pattern_features[col])[0]

    info_text = info_text[:-3]
    nodes_values = [Patterns_Dictionary[pattern_id]['pattern']._node[n]['value'] for n in
                    Patterns_Dictionary[pattern_id]['pattern'].nodes]

    if len(Patterns_Dictionary[pattern_id]['pattern'].edges) == 0:
        P_nodes = list(Patterns_Dictionary[pattern_id]['pattern'].nodes)
        Patterns_Dictionary[pattern_id]['pattern'].add_node('start', value='start', parallel=False, color='k')
        Patterns_Dictionary[pattern_id]['pattern'].add_node('end', value='end', parallel=False, color='k')
        for node in P_nodes:
            Patterns_Dictionary[pattern_id]['pattern'].add_edge('start', node, eventually=False)
            Patterns_Dictionary[pattern_id]['pattern'].add_edge(node, 'end', eventually=False)

    values = nx.get_node_attributes(Patterns_Dictionary[pattern_id]['pattern'], 'value')
    colors = list(nx.get_node_attributes(Patterns_Dictionary[pattern_id]['pattern'], 'color').values())
    edge_styles = []
    for v in nx.get_edge_attributes(Patterns_Dictionary[pattern_id]['pattern'], 'eventually').values():
        if v:
            edge_styles.append('b')
        else:
            edge_styles.append('k')

    sizes = []
    for c in colors:
        if c == 'k':
            sizes.append(10)
        else:
            sizes.append(300)

    pos = defining_graph_pos(Patterns_Dictionary[pattern_id]['pattern'])

    nx.draw_networkx_nodes(Patterns_Dictionary[pattern_id]['pattern'], pos,
                           node_color=colors, node_size=sizes, ax=ax[0][0])

    text = nx.draw_networkx_labels(Patterns_Dictionary[pattern_id]['pattern'], pos, values, ax=ax[0][0])
    for _, t in text.items():
        t.set_rotation('vertical')

    nx.draw_networkx_edges(Patterns_Dictionary[pattern_id]['pattern'], pos, arrows=True,
                           width=2, edge_color=edge_styles, ax=ax[0][0])

    plt.title('Pattern ID: ' + str(pattern_id))
    plt.axis('off')

    # # place a text box in upper left in axes coord

    for v in np.unique(nodes_values):
        if v in ['start', 'end']:
            continue
        plt.scatter([], [], c=color_act_dict[v], label=v)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
    plt.text(0.05, 0.9, info_text, fontsize=12,
             verticalalignment='top', bbox=props, transform=ax[1][0].transAxes)

    # plt.legend(loc='lower left')
    ax[1][0].axis('off')

    return fig, ax


def defining_graph_pos(G):
    pos = dict()
    parallel_pattern_nodes = dict()
    parallelism = nx.get_node_attributes(G, 'parallel')

    ii = 0
    observed_parallel = set()
    for node in list(G.nodes):
        if node in observed_parallel:
            continue
        if parallelism[node]:
            parallel_pattern_nodes[ii] = {node}
            in_pattern_nodes = set(G.in_edges._adjdict[node].keys())
            out_pattern_nodes = set(G.out_edges._adjdict[node].keys())
            Other_nodes = set(parallelism.keys())
            Other_nodes.remove(node)
            for ND in Other_nodes:
                if not parallelism[ND]:
                    continue
                in_pattern_ND = set(G.in_edges._adjdict[ND].keys())
                out_pattern_ND = set(G.out_edges._adjdict[ND].keys())
                if in_pattern_nodes == in_pattern_ND and out_pattern_nodes == out_pattern_ND:
                    parallel_pattern_nodes[ii].add(ND)
                    observed_parallel.add(ND)
                    observed_parallel.add(node)
            ii += 1

    non_parallel_nodes = set(parallelism.keys()).difference(observed_parallel)
    num_locations = len(parallel_pattern_nodes) + len(non_parallel_nodes)

    ordered_nodes = list(G.nodes)
    loc_x = 0
    loc_y = 0
    end = False
    if 'start' in ordered_nodes or 'end' in ordered_nodes:
        pos['start'] = np.array((loc_x, loc_y))
        loc_x += (1 / num_locations)
        end = True
        ordered_nodes.remove('start')
        ordered_nodes.remove('end')

    ordered_nodes.sort()
    for node in ordered_nodes:
        if node in non_parallel_nodes:
            pos[node] = np.array((loc_x, loc_y))

            loc_x += (1 / num_locations)
        else:
            for key in parallel_pattern_nodes:
                if node in parallel_pattern_nodes[key]:
                    loc_y = - (1 / len(parallel_pattern_nodes[key])) / len(parallel_pattern_nodes[key])
                    for ND in parallel_pattern_nodes[key]:
                        pos[ND] = np.array((loc_x, loc_y))
                        loc_y += (1 / len(parallel_pattern_nodes[key]))

                    loc_x += (1 / num_locations)
                    loc_y = 0
                    break
    if end:
        pos['end'] = np.array((loc_x, loc_y))

    return pos


def Single_Pattern_Extender(all_extension_list, chosen_pattern_ID,
                            patient_data, EventLog_graphs, all_variants):
    # Extension_3_patterns = []
    Extended_patterns_at_stage = dict()
    # for chosen_pattern_ID in All_extended_patterns_2:
    new_patterns_for_core = []
    Core_activity = chosen_pattern_ID.split("_")[0]
    print('Core:  ' + Core_activity)
    selected_variants = all_variants[Core_activity]
    # if any(nx.get_edge_attributes(All_extended_patterns_2[chosen_pattern_ID]['pattern'], 'eventually').values()):
    #     return None, None, None
    print(chosen_pattern_ID)
    for idx, case in enumerate(all_extension_list[chosen_pattern_ID]['Instances']['case']):
        Trace_graph = EventLog_graphs[case].copy()
        nodes_values = [Trace_graph._node[n]['value'] for n in Trace_graph.nodes]
        embedded_trace_graph = all_extension_list[chosen_pattern_ID]['Instances']['emb_trace'][idx]
        inside_pattern_nodes = set(Trace_graph.nodes).difference(set(embedded_trace_graph.nodes))
        to_remove = set(Trace_graph.nodes).difference(inside_pattern_nodes)
        chosen_pattern = Trace_graph.copy()
        chosen_pattern.remove_nodes_from(to_remove)

        ending_nodes = {n[0] for n in chosen_pattern.out_degree if n[1] == 0}
        starting_nodes = {n[0] for n in chosen_pattern.in_degree if n[1] == 0}

        case_data = selected_variants[selected_variants['case:concept:name'] == case]
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
                                          value=values[in_node], parallel=parallel[in_node],
                                          color=color[in_node])
                for node in starting_nodes:
                    extended_pattern.add_edge(in_node, node, eventually=False)

            new_embedded_trace_graph = create_embedded_pattern_in_trace(set(extended_pattern.nodes),
                                                                        Trace_graph)
            Extended_patterns_at_stage, new_Pattern_IDs = update_pattern_dict(Extended_patterns_at_stage,
                                                                              extended_pattern,
                                                                              new_embedded_trace_graph,
                                                                              case_data, 'case:concept:name', nm,
                                                                              em,
                                                                              chosen_pattern_ID,
                                                                              new_patterns_for_core)
            if new_Pattern_IDs != "":
                new_patterns_for_core.append(new_Pattern_IDs)

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

            new_embedded_trace_graph = create_embedded_pattern_in_trace(set(extended_pattern.nodes),
                                                                        Trace_graph)
            Extended_patterns_at_stage, new_Pattern_IDs = update_pattern_dict(Extended_patterns_at_stage,
                                                                              extended_pattern,
                                                                              new_embedded_trace_graph,
                                                                              case_data, 'case:concept:name', nm,
                                                                              em,
                                                                              chosen_pattern_ID,
                                                                              new_patterns_for_core)
            if new_Pattern_IDs != "":
                new_patterns_for_core.append(new_Pattern_IDs)

        ## all non-direct nodes
        Eventual_relations_nodes = set(embedded_trace_graph.nodes).difference(
            in_pattern_nodes.union(out_pattern_nodes))
        Eventual_relations_nodes.remove('pattern')

        # Eventually following patterns
        if len(out_pattern_nodes) > 0:
            Eventual_following_nodes = {node for node in Eventual_relations_nodes if
                                        node > max(out_pattern_nodes)}
            for Ev_F_nodes in Eventual_following_nodes:
                Eventual_follow_pattern = chosen_pattern.copy()
                Eventual_follow_pattern.add_node(Ev_F_nodes,
                                                 value=values[Ev_F_nodes], parallel=parallel[Ev_F_nodes],
                                                 color=color[Ev_F_nodes])
                for node in ending_nodes:
                    Eventual_follow_pattern.add_edge(node, Ev_F_nodes, eventually=True)

                Extended_patterns_at_stage, new_Pattern_IDs = update_pattern_dict(Extended_patterns_at_stage,
                                                                                  Eventual_follow_pattern, [],
                                                                                  case_data, 'case:concept:name',
                                                                                  nm, em,
                                                                                  chosen_pattern_ID,
                                                                                  new_patterns_for_core)
                if new_Pattern_IDs != "":
                    new_patterns_for_core.append(new_Pattern_IDs)

        # Eventually preceding patterns
        if len(in_pattern_nodes) > 0:
            Eventual_preceding_nodes = {node for node in Eventual_relations_nodes if
                                        node < min(in_pattern_nodes)}
            for Ev_P_nodes in Eventual_preceding_nodes:
                Eventual_preceding_pattern = chosen_pattern.copy()
                Eventual_preceding_pattern.add_node(Ev_P_nodes,
                                                    value=values[Ev_P_nodes], parallel=parallel[Ev_P_nodes],
                                                    color=color[Ev_P_nodes])
                for node in starting_nodes:
                    Eventual_preceding_pattern.add_edge(Ev_P_nodes, node, eventually=True)

                Extended_patterns_at_stage, new_Pattern_IDs = update_pattern_dict(Extended_patterns_at_stage,
                                                                                  Eventual_preceding_pattern, [],
                                                                                  case_data, 'case:concept:name',
                                                                                  nm, em,
                                                                                  chosen_pattern_ID,
                                                                                  new_patterns_for_core)
                if new_Pattern_IDs != "":
                    new_patterns_for_core.append(new_Pattern_IDs)

        # Extension_3_patterns.extend(new_patterns_for_core)
        if len(new_patterns_for_core) > 0:

            # Extended_patterns_at_stage[chosen_pattern_ID] = dict()
            # Extended_patterns_at_stage[chosen_pattern_ID]['dict'] = Extended_pattern_dictionary
            # Extended_patterns_at_stage[chosen_pattern_ID]['Egraph'] = EventLog_graphs
            # Extended_patterns_at_stage[chosen_pattern_ID]['variants'] = selected_variants

            patient_data[new_patterns_for_core] = 0
            for PID in new_patterns_for_core:
                for CaseID in np.unique(Extended_patterns_at_stage[PID]['Instances']['case']):
                    variant_frequency_case = Extended_patterns_at_stage[PID]['Instances']['case'].count(CaseID)
                    Other_cases = \
                        selected_variants.loc[
                            selected_variants['case:concept:name'] == CaseID, 'case:CaseIDs'].tolist()[
                            0]
                    for Ocase in Other_cases:
                        patient_data.loc[
                            patient_data['case:concept:name'] == Ocase, PID] = variant_frequency_case

    all_extension_list.update(Extended_patterns_at_stage)
    return all_extension_list, Extended_patterns_at_stage, patient_data
