from sklearn.tree import DecisionTreeClassifier
import networkx as nx
import math
import pandas as pd
from scipy.spatial.distance import pdist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from pylab import figure
from sklearn.impute import KNNImputer
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import spearmanr
import networkx.algorithms.isomorphism as iso
from scipy.stats import entropy


def feature_entropy(df, feature):
    # Count the number of instances that have each unique value of the feature
    count = df[feature].value_counts().to_dict()
    # Convert the counts into probabilities
    prob = [count[value] / sum(count.values()) for value in count]

    return entropy(prob)


def calculate_pairwise_case_distance(X_features, num_col, Imput=False):
    Cat_exists = False
    Num_exists = False

    if len(num_col) > 0 and Imput:
        X_features.replace(9999, np.nan, inplace=True)
        Xx = X_features[num_col]
        imputer = KNNImputer(n_neighbors=5)
        imputer.fit(Xx)
        Xx_Imputed = imputer.transform(Xx)
        Imputed_EventLog = pd.DataFrame(Xx_Imputed, columns=num_col)
        X_features[num_col] = Imputed_EventLog

    normalizer = preprocessing.StandardScaler()
    for num in num_col:
        if num == 'leeft' or num == 'open_cases':
            x = X_features[num].values.reshape(-1, 1)
            X_features[num] = normalizer.fit_transform(x)
        else:
            X_features[num] += 0.000005
            num_x = X_features[num].values.reshape(-1, 1)
            X_features[num] = np.log(num_x)

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
        # numeric_dist = 1 - numeric_dist

    if Cat_exists and Num_exists:
        Combined_dist = ((len(cat_col) * cat_dist) + numeric_dist) / (1 + len(cat_col))
        return Combined_dist

    elif Cat_exists and not Num_exists:
        return cat_dist

    elif Num_exists and not Cat_exists:
        return numeric_dist


def update_pattern_dict(Patterns_Dictionary, pattern, embedded_trace_graph,
                        case_data, case_column, node_match, edge_match, core, new_patterns_for_core=None):
    if new_patterns_for_core is None:
        new_patterns_for_core = list(Patterns_Dictionary.keys())
    discovered_patterns = list(Patterns_Dictionary.keys())
    for PID in discovered_patterns:
        if nx.is_isomorphic(pattern, Patterns_Dictionary[PID]['pattern'],
                            node_match=node_match, edge_match=edge_match):
            Patterns_Dictionary[PID]['Instances']['case'].append(case_data[case_column].unique()[0])
            Patterns_Dictionary[PID]['Instances']['emb_trace'].append(embedded_trace_graph)
            new_Pattern_IDs = ""
            break
    else:
        if len(new_patterns_for_core) > 0:
            Pattern_IDs = max([int(s.split("_")[-1]) for s in new_patterns_for_core])
            new_Pattern_IDs = core + "_" + str(Pattern_IDs + 1)
        else:
            new_Pattern_IDs = core + "_1"

        Patterns_Dictionary[new_Pattern_IDs] = {'Instances': {'case': [case_data[case_column].unique()[0]],
                                                              'emb_trace': [embedded_trace_graph]},
                                                'pattern': pattern}

    return Patterns_Dictionary, new_Pattern_IDs

def create_pattern_frame(pattern_list):
    patterns_data = pd.DataFrame(columns=pattern_list)
    patterns_data = patterns_data.transpose()
    patterns_data['patterns'] = patterns_data.index
    patterns_data.reset_index(inplace=True, drop=True)
    return patterns_data


def frequency_measuring_patterns(patterns_data, pattern_list, patient_data, Core_activity):
    for pattern in pattern_list:
        patterns_data.loc[patterns_data['patterns'] == pattern, 'Pattern_Frequency'] \
            = np.sum(patient_data[pattern])

        patterns_data.loc[patterns_data['patterns'] == pattern, 'Case_Support'] = \
            np.count_nonzero(patient_data[pattern])

        patterns_data.loc[patterns_data['patterns'] == pattern, 'Case_Coverage'] = \
            np.count_nonzero(patient_data[pattern]) / len(patient_data)

    return patterns_data


def predictive_measuring_patterns(patterns_data, patient_data, label_col, label_class, Core_activity=None):
    if Core_activity is not None:
        patient_data = patient_data[patient_data[Core_activity] != 0]

    x = patient_data[patterns_data['patterns']]
    if label_col is not None:
        y = patient_data[label_col]
        for pattern in patterns_data['patterns']:
            patterns_data.loc[patterns_data['patterns'] == pattern, 'OutcomeCorrelation'], _ = \
                spearmanr(y, x[pattern])
            _, patterns_data.loc[patterns_data['patterns'] == pattern, 'p_values'] = \
                spearmanr(y, x[pattern])

            patterns_data.loc[patterns_data['patterns'] == pattern, 'MedianOutcome_in'] = np.median(
                y.loc[x[pattern] > 0])
            patterns_data.loc[patterns_data['patterns'] == pattern, 'MedianOutcome_out'] = np.median(
                y.loc[x[pattern] == 0])

    if label_class is not None:
        cat_y_all = patient_data[label_class]
        for pattern in patterns_data['patterns']:
            if len(np.unique(cat_y_all)) < 2:
                label_ranges = [0, 1]
            else:
                label_ranges = np.unique(cat_y_all)

            cat_y = list(patient_data.loc[patient_data[pattern] > 0, label_class])
            cat_y_out = list(patient_data.loc[patient_data[pattern] == 0, label_class])
            patterns_data.loc[patterns_data['patterns'] == pattern,
                              "PositiveOutcome_rate_pattern"] = np.sum(cat_y) / len(cat_y)
            patterns_data.loc[patterns_data['patterns'] == pattern,
                              "PositiveOutcome_rate_anti-pattern"] = np.sum(cat_y_out) / len(cat_y_out)

            cor, _ = spearmanr(cat_y_all, x[pattern])
            if np.isnan(cor):
                patterns_data.loc[patterns_data['patterns'] == pattern, 'OutcomeCorrelation'] = 0
                patterns_data.loc[patterns_data['patterns'] == pattern, 'p_values'] = np.nan
            else:
                patterns_data.loc[patterns_data['patterns'] == pattern, 'OutcomeCorrelation'], _ = \
                    spearmanr(cat_y_all, x[pattern])

                _, patterns_data.loc[patterns_data['patterns'] == pattern, 'p_values'] = \
                    spearmanr(cat_y_all, x[pattern])


        # info_gain = mutual_info_classif(x, cat_y_all, discrete_features=True)
        # patterns_data['Info_Gain'] = info_gain.reshape(-1, 1)

    # if survival:
    #     for pattern in patterns_data['patterns']:
    #         group1_cases = patient_data.loc[patient_data[pattern] > 0, 'case:concept:name']
    #         group2_cases = patient_data.loc[~patient_data['case:concept:name'].isin(group1_cases), 'case:concept:name']
    #
    #         Temp_patient_dataset = patient_data.drop_duplicates(subset=['case:concept:name'])
    #
    #         T = Temp_patient_dataset.loc[Temp_patient_dataset['case:concept:name'].isin(group1_cases), 'LifeTime']
    #         E = Temp_patient_dataset.loc[Temp_patient_dataset['case:concept:name'].isin(group1_cases), 'vit_stat']
    #
    #         T1 = Temp_patient_dataset.loc[Temp_patient_dataset['case:concept:name'].isin(group2_cases), 'LifeTime']
    #         E1 = Temp_patient_dataset.loc[Temp_patient_dataset['case:concept:name'].isin(group2_cases), 'vit_stat']
    #
    #         results = logrank_test(T, T1, event_observed_A=E, event_observed_B=E1)
    #
    #         patterns_data.loc[patterns_data['patterns'] == pattern, 'LR_p'] = results.p_value
    #
    #         if results.p_value < 0.05:
    #             print(pattern)

    return patterns_data


def similarity_measuring_patterns(patterns_data, patient_data, pair_cases, start_search_points,
                                  pairwise_distances_array, Core_activity=False):
    # if Core_activity is not None:
    #     patient_data = patient_data[patient_data[Core_activity] != 0]

    for pattern in patterns_data['patterns']:

        # if Core_activity:
        #     Core_activity = pattern.split("_")[0]
        #     patient_data = patient_data[patient_data[Core_activity] != 0]

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

        patterns_data.loc[patterns_data['patterns'] == pattern, 'Case_Distance'] = \
            np.mean([pairwise_distances_array[ind] for ind in selected_pair_index])

    return patterns_data


def create_pattern_attributes(patient_data, label_col, label_class, case_id, Core_activity, pattern_list,
                              num_col, binary_col):
    patterns_data = create_pattern_frame(pattern_list)

    ## Frequency-based measures
    patterns_data = frequency_measuring_patterns(patterns_data, pattern_list, patient_data, Core_activity)

    ## Discriminative measures
    patterns_data = predictive_measuring_patterns(patterns_data, patient_data, label_col, label_class, Core_activity)

    ## clustering based on patterns
    X_features = patient_data.drop([case_id, label_col], axis=1)
    normalizer = preprocessing.StandardScaler()
    for num in num_col:
        # X_features[num] += 0.000005
        x = X_features[num].values.reshape(-1, 1)
        # X_features[num] = np.log(x)
        X_features[num] = normalizer.fit_transform(x)

    for pattern in patterns_data['patterns']:
        patient_data.loc[patient_data[pattern] > 0, pattern] = 1
        cluster = patient_data[pattern]

        core_cluster_without_pattern = np.mean(patient_data.loc[patient_data[pattern] == 0, num_col])
        core_cluster_with_pattern = np.mean(patient_data.loc[patient_data[pattern] > 0, num_col])

        if sum(cluster) == len(patient_data):
            patterns_data.loc[patterns_data['patterns'] == pattern, 'CaseDistance'] = 0
        else:
            patterns_data.loc[patterns_data['patterns'] == pattern, 'CaseDistance'] = \
                math.dist(core_cluster_without_pattern, core_cluster_with_pattern)

    normalizer = preprocessing.MinMaxScaler()
    x = patterns_data['CaseDistance'].values.reshape(-1, 1)
    patterns_data['CaseDistance'] = normalizer.fit_transform(x)

    return patterns_data


def create_pattern_attributes_IKNL(patient_data, label_col, label_class, Core_activity, pattern_list,
                                   pairwise_distances_array, pair_cases, start_search_points):
    patterns_data = create_pattern_frame(pattern_list)
    ## Frequency-based measures
    patterns_data = frequency_measuring_patterns(patterns_data, pattern_list, patient_data, Core_activity)

    ## Discriminative measures
    patterns_data = predictive_measuring_patterns(patterns_data, patient_data, label_col, label_class, Core_activity)

    ## Similarity measures
    patterns_data = similarity_measuring_patterns(patterns_data, patient_data, pair_cases, start_search_points,
                                                  pairwise_distances_array, Core_activity)

    return patterns_data


def create_pattern_attributes_BPIC12(patient_data, label_col, label_class, Core_activity, pattern_list):
    patterns_data = create_pattern_frame(pattern_list)
    ## Frequency-based measures
    patterns_data = frequency_measuring_patterns(patterns_data, pattern_list, patient_data, Core_activity)

    ## Discriminative measures
    patterns_data = predictive_measuring_patterns(patterns_data, patient_data, label_col, label_class, Core_activity)

    ## Similarity measures
    if Core_activity is not None:
        patient_data = patient_data[patient_data[Core_activity] != 0]
    for pattern in patterns_data['patterns']:
        in_pattern_mean = np.mean(patient_data.loc[patient_data[pattern] > 0, 'AMOUNT_REQ'])
        if len(patient_data[patient_data[pattern] > 0]) > 0 and len(patient_data[patient_data[pattern] == 0]) > 0:
            out_pattern_mean = np.mean(patient_data.loc[patient_data[pattern] == 0, 'AMOUNT_REQ'])
            patterns_data.loc[patterns_data['patterns'] == pattern, 'Case_Distance'] = \
                np.abs(in_pattern_mean - out_pattern_mean)
        else:
            patterns_data.loc[patterns_data['patterns'] == pattern, 'Case_Distance'] = 0

    normalizer = preprocessing.MinMaxScaler()
    x = patterns_data['Case_Distance'].values.reshape(-1, 1)
    patterns_data['Case_Distance'] = normalizer.fit_transform(x)

    return patterns_data


def threeD_ploting(df, D1, D2, D3, folder_address, color_act_dict=None, activity_level=False):
    if activity_level:
        fig = plt.figure(figsize=[12, 6])
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.set_xlabel(D1)
        ax.set_ylabel(D2)
        ax.set_zlabel(D3)
        for ticker, row in df.iterrows():
            ax.scatter(row[D1], row[D2], row[D3], c=color_act_dict[row['patterns']])
        ax2 = fig.add_subplot(1, 2, 2)
        for v in df['patterns']:
            ax2.scatter([], [], c=color_act_dict[v], label=v)
        ax2.legend()
        ax2.axis('off')
        plt.savefig(folder_address + "pattern/3Dplot_CoreActivities.png")
    else:
        fig = figure(figsize=[8, 8])
        ax = fig.add_subplot(projection='3d')
        for ticker, row in df.iterrows():
            ax.scatter(row[D1], row[D2], row[D3])
            ax.text(row[D1], row[D2], row[D3], row['patterns'])

        ax.set_xlabel(D1)
        ax.set_ylabel(D2)
        ax.set_zlabel(D3)
        plt.savefig(folder_address + "pattern/3Dplot_%s.png" % row['patterns'].split('_')[0])

    plt.show()
    plt.close(fig)


def create_embedded_pattern_in_trace(inside_pattern_nodes, Trace_graph):
    start_node = min(inside_pattern_nodes)
    end_node = max(inside_pattern_nodes)
    embedded_trace_graph = Trace_graph.copy()
    embedded_trace_graph.add_node('pattern', value='pattern', parallel=False, color='#000000')
    in_nodes_for_pattern = set(embedded_trace_graph.pred[start_node].keys())
    out_nodes_for_pattern = set(embedded_trace_graph.succ[end_node].keys())
    for ii in in_nodes_for_pattern:
        embedded_trace_graph.add_edge(ii, 'pattern', eventually=False)
    for out in out_nodes_for_pattern:
        embedded_trace_graph.add_edge('pattern', out, eventually=False)

    embedded_trace_graph.remove_nodes_from(inside_pattern_nodes)

    return embedded_trace_graph


def plot_patterns(Patterns_Dictionary, pattern_id, color_act_dict, pattern_attributes, folder_address):
    fig, ax = plt.subplots(1, 2, figsize=[14, 6])
    # fig = figure(figsize=[8, 8])
    # ax = fig.add_subplot()

    pattern_features = pattern_attributes[pattern_attributes['patterns'] == pattern_id]
    info_text = ""
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
                           node_color=colors, node_size=sizes, ax=ax[0])

    text = nx.draw_networkx_labels(Patterns_Dictionary[pattern_id]['pattern'], pos, values, ax=ax[0])
    for _, t in text.items():
        t.set_rotation('vertical')

    nx.draw_networkx_edges(Patterns_Dictionary[pattern_id]['pattern'], pos, arrows=True,
                           width=2, edge_color=edge_styles, ax=ax[0])

    plt.title('Pattern ID: ' + str(pattern_id))
    plt.axis('off')

    # # place a text box in upper left in axes coord

    for v in np.unique(nodes_values):
        if v in ['start', 'end']:
            continue
        plt.scatter([], [], c=color_act_dict[v], label=v)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
    plt.text(0.2, 0.9, info_text, fontsize=12,
             verticalalignment='top', bbox=props, transform=ax[1].transAxes)

    plt.legend(loc='lower left')
    plt.savefig(folder_address + "pattern/Pattern_%s.png" % pattern_id)
    plt.show()
    plt.close(fig)


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


def plot_patterns_dashboard(Patterns_Dictionary, pattern_id, color_act_dict, pattern_attributes):
    fig, ax = plt.subplots(3, 6, figsize=[60, 30])

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
                           node_color=colors, node_size=sizes, ax=ax[0, 0])

    nx.draw_networkx_edges(Patterns_Dictionary[pattern_id]['pattern'], pos, arrows=True,
                           width=2, edge_color=edge_styles, ax=ax[0, 0])

    for v in np.unique(nodes_values):
        if v in ['start', 'end']:
            continue
        ax[0, 0].scatter([], [], c=color_act_dict[v], label=v)

    ax[0, 0].axis('off')
    ax[0, 0].legend(loc='upper left', prop={'size': 20})
    # ax[0, 0].legend(prop={'size': 50})

    return fig, ax


def Pattern_Extender(All_extended_patterns_2, patient_data, EventLog_graphs, data, case_id, activity):
    Extension_3_patterns = []
    Extended_patterns_at_stage = dict()
    for chosen_pattern_ID in All_extended_patterns_2:
        new_patterns_for_core = []
        Core_activity = chosen_pattern_ID.split("_")[0]
        print('Core:  ' + Core_activity)
        filtered_cases = data.loc[data[activity] == Core_activity, case_id]
        filtered_main_data = data[data[case_id].isin(filtered_cases)]
        if any(nx.get_edge_attributes(All_extended_patterns_2[chosen_pattern_ID]['pattern'], 'eventually').values()):
            continue
        print(chosen_pattern_ID)
        for idx, case in enumerate(All_extended_patterns_2[chosen_pattern_ID]['Instances']['case']):
            Trace_graph = EventLog_graphs[case].copy()
            nodes_values = [Trace_graph._node[n]['value'] for n in Trace_graph.nodes]
            embedded_trace_graph = All_extended_patterns_2[chosen_pattern_ID]['Instances']['emb_trace'][idx]
            inside_pattern_nodes = set(Trace_graph.nodes).difference(set(embedded_trace_graph.nodes))
            to_remove = set(Trace_graph.nodes).difference(inside_pattern_nodes)
            chosen_pattern = Trace_graph.copy()
            chosen_pattern.remove_nodes_from(to_remove)

            ending_nodes = {n[0] for n in chosen_pattern.out_degree if n[1] == 0}
            starting_nodes = {n[0] for n in chosen_pattern.in_degree if n[1] == 0}

            case_data = filtered_main_data[filtered_main_data[case_id] == case]
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

                new_embedded_trace_graph = create_embedded_pattern_in_trace(set(extended_pattern.nodes), Trace_graph)
                Extended_patterns_at_stage, new_Pattern_IDs = update_pattern_dict(Extended_patterns_at_stage,
                                                                                  extended_pattern,
                                                                                  new_embedded_trace_graph,
                                                                                  case_data, case_id, nm,
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
                                                                                  case_data, case_id, nm,
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
                                                                                      case_data, case_id,
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
                                                                                      case_data, case_id,
                                                                                      nm, em,
                                                                                      chosen_pattern_ID,
                                                                                      new_patterns_for_core)
                    if new_Pattern_IDs != "":
                        new_patterns_for_core.append(new_Pattern_IDs)

        Extension_3_patterns.extend(new_patterns_for_core)
        if len(new_patterns_for_core) > 0:
            patient_data[new_patterns_for_core] = 0
            for PID in new_patterns_for_core:
                for CaseID in np.unique(Extended_patterns_at_stage[PID]['Instances']['case']):
                    variant_frequency_case = Extended_patterns_at_stage[PID]['Instances']['case'].count(CaseID)
                    patient_data.loc[patient_data['case:concept:name'] == CaseID, PID] = variant_frequency_case

    return Extension_3_patterns, Extended_patterns_at_stage, patient_data

def Classifiers_kFold_results(All_extended_patterns, patient_data, test_list, train_list, label_class):
    ML_model = DecisionTreeClassifier()
    result_dict = dict()
    for Obj in All_extended_patterns:
        ACC, Fscore, AUC = [], [], []
        for fold in range(len(test_list)):
            train_X = patient_data[patient_data['case:concept:name'].isin(train_list[fold + 1])]
            test_X = patient_data[patient_data['case:concept:name'].isin(test_list[fold + 1])]

            train_y = train_X[label_class]
            test_y = test_X[label_class]

            train_X = train_X[All_extended_patterns[Obj]]
            test_X = test_X[All_extended_patterns[Obj]]

            ML_model.fit(train_X, train_y)
            predicted = ML_model.predict(test_X)

            ACC.append(accuracy_score(test_y, predicted))
            Fscore.append(f1_score(test_y, predicted, average='weighted'))
            # AUC.append(roc_auc_score(test_y, predicted, multi_class="ovr"))

        result_dict[Obj] = np.mean(Fscore)
        result_dict[Obj + "_std"] = np.std(Fscore)
        print('Performance for patterns obtained using %s' % Obj)
        print('Mean Accuracy: %.3f (%.3f)' % (np.mean(ACC), np.std(ACC)))
        print('Mean f-score: %.3f (%.3f)' % (np.mean(Fscore), np.std(Fscore)))
        # print('Mean AUC: %.3f (%.3f)' % (np.mean(AUC), np.std(AUC)))
        print('---------------------------------------------------------------')

    return result_dict
