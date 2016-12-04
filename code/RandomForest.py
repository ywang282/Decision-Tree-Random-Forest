#!/usr/bin/python

import sys
import pprint
import copy
import random
#todo
import sklearn
#todo
from sklearn import tree

def get_node_dict(node):
    '''
    DEBUG

    '''
    node_dict = {'terminate':-1, 'label': -1, 'attribute':-1, 'children':{}}
    node_dict['terminate'] = node.terminate
    node_dict['label'] = node.label
    node_dict['attribute'] = node.attribute
    return node_dict

def debug_print(node):
    '''
    DEBUG
    
    '''
    # traverse the node:
    if node is None:
        return {}
    node_dict = get_node_dict(node)
    if len(node.children.keys()) != 0:
        for key in node.children.keys():
            node_dict['children'].setdefault(key, {})
            node_dict['children'][key] = debug_print(node.children[key])
    return node_dict


def get_decision(attr_pairs, forest):
    '''
    run the classifier and get the decision for each tuple

    '''
    attr_pair_dict = {}
    for attr_pair in attr_pairs:
        attr = attr_pair.split(':')[0]
        val = attr_pair.split(':')[1]
        attr_pair_dict[attr] = val


    results = []
    for tree in forest:
        curr = tree
        while(curr.terminate==False):
            split_attr = curr.attribute
            value = attr_pair_dict[split_attr]
            if value not in curr.children:
                value = curr.children.keys()[len(curr.children.keys())-1]
            curr = curr.children[value]
        results.append(int(curr.label))
    return max(set(results), key=results.count)


def test_classifier(testing_file_path, forest):
    '''
    wrapper function to run the classifier and get the decisions for all the tuples

    '''
    with open(testing_file_path) as fin:
        lines = fin.read().splitlines()

    count = 0 
    for line in lines:
        if len(line)<1:
            continue
        label = int(line.split()[0])
        attr_pairs = line.split()[1:]
        decision = get_decision(attr_pairs, forest)
        if label != decision:
            count += 1
    print ('mine:', 100-float(count)*100/len(lines))


def run_standard(training_file_path, testing_file_path):
    '''
    run the sklearn module on training set and testing set

    '''
    count = 0 
    # read the training set
    with open(training_file_path) as fin:
        train_lines = fin.read().splitlines()

    x = []
    y = []

    for line in train_lines:
        if len(line)<1:
            continue
        label = line.split()[0]
        pairs = line.split()[1:]
        sub_list = []
        for pair in pairs:
            sub_list.append(int(pair.split(':')[1]))
        x.append(sub_list)
        y.append(int(label))
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x, y)

    # read the testing set
    with open(testing_file_path) as fin:
        test_lines = fin.read().splitlines()

    x_x = []

    for line in test_lines:
        if len(line)<1:
            continue
        label = line.split()[0]
        pairs = line.split()[1:]
        sub_list = []
        for pair in pairs:
            sub_list.append(int(pair.split(':')[1]))
        x_x.append(sub_list)

    result = clf.predict(x_x)

    for i in range(len(result)):
        label = int(test_lines[i].split()[0])
        decision = result[i]
        if label != decision:
            count += 1
    print ('standard:', 100-float(count)*100/len(test_lines))


def compute_gini_val(sub_dict):
    '''
    compute the gini score on a subset D

    '''
    score = 1
    count = sub_dict['counts']
    for label in sub_dict.keys():
        if label == 'counts':
            continue
        score -= (float(sub_dict[label])/count)**2
    return score


def compute_gini_split(data_dict, attribute):
    '''
    compute the gini score on a splitting attribute

    '''
    score = 0
    for val in data_dict['attr_val_pairs'][attribute].keys():
        if val == 'counts':
            continue
        D = data_dict['attr_val_pairs'][attribute]['counts']
        D_val = data_dict['attr_val_pairs'][attribute][val]['counts']
        coeff = float(D_val)/D
        sub_dict = data_dict['attr_val_pairs'][attribute][val]
        gini_val = compute_gini_val(sub_dict)
        score += coeff*gini_val
    return score


def compute_gini_index(data_dict, attribute_list):
    '''
    compute the index with the lowest gini score on the splitting attributes

    '''
    gini_score = []
    for attribute in attribute_list:
        score = compute_gini_split(data_dict, attribute)
        gini_score.append(score)
    min_index = gini_score.index(min(gini_score))
    return min_index


def build_reverse_index(lines):
    '''
    build the reverse index for fast access of the training set data

    '''
    data_dict = {
        'labels' : {},
        'attr_val_pairs' : {}
    }
    for line in lines:
        if len(line)<1:
            continue
        label = line.split(' ')[0]
        attr_val_pairs = line.split(' ')[1:]

        # set reverse index for labels
        data_dict['labels'].setdefault(label, 0)
        data_dict['labels'][label] += 1

        # set reverse index for attr_val_pairs 
        for attr_val_pair in attr_val_pairs:
            attribute = attr_val_pair.split(':')[0]
            value = attr_val_pair.split(':')[1]
            data_dict['attr_val_pairs'].setdefault(attribute, {'counts': 0})
            data_dict['attr_val_pairs'][attribute]['counts'] += 1
            data_dict['attr_val_pairs'][attribute].setdefault(value, {'counts': 0})
            data_dict['attr_val_pairs'][attribute][value]['counts'] += 1
            data_dict['attr_val_pairs'][attribute][value].setdefault(label, 0)
            data_dict['attr_val_pairs'][attribute][value][label] += 1

    return data_dict


def buildtree(data_dict, attribute_list, num_selected_attributes, lines, Node):
    '''
    build the decision tree

    '''
    # check for terminate condition
    if len(attribute_list) < num_selected_attributes:
        inverse = [(value, key) for key, value in data_dict['labels'].items()]
        return Node(terminate=True, label=max(inverse)[1])

    if len(attribute_list) == 0:
        inverse = [(value, key) for key, value in data_dict['labels'].items()]
        return Node(terminate=True, label=max(inverse)[1])

    if len(data_dict['labels'].keys()) == 1:
        return Node(terminate=True, label=data_dict['labels'].keys()[0])

    # todo if2) The current instance set is empty

    # select a random subset of the attributes
    random.shuffle(attribute_list)
    attribute_subset = (copy.copy(attribute_list))[:num_selected_attributes]
    gini_index = compute_gini_index(data_dict, attribute_subset)
    attribute = attribute_subset[gini_index]
    node = Node(attribute=attribute)

    # create children for the node
    attribute_list.pop(gini_index)
    values = data_dict['attr_val_pairs'][attribute].keys()
    for val in values:
        if val == 'counts':
            continue
        # truncate the lines
        new_lines = []
        for line in lines:
            if len(line)<1:
                continue
            attr_val_pairs = line.split(' ')[1:]
            for attr_val_pair in attr_val_pairs:
                attr = attr_val_pair.split(':')[0]
                value = attr_val_pair.split(':')[1]
                if attr == attribute and value == val:
                    new_lines.append(line)
        new_data_dict = build_reverse_index(new_lines)
        subtree = buildtree(new_data_dict, copy.copy(attribute_list), num_selected_attributes, new_lines, Node)
        node.children.setdefault(val, None)
        node.children[val] = subtree
    return node


def construct_decision_tree(num_attribute_fraction, num_tuple, lines):
    '''
    wrapper function for constructing the decision tree

    '''
    class Node():
        def __init__(self, terminate=False, attribute=None, label=-1):
            self.terminate = terminate
            self.attribute = attribute
            self.label = label
            self.children = {}

    # random select a subset of lines
    subset_lines = []
    for i in range(num_tuple):
        random_int = random.randint(0, num_tuple-1)
        subset_lines.append(copy.copy(lines[random_int]))
    data_dict = build_reverse_index(subset_lines)
    attribute_list = data_dict['attr_val_pairs'].keys()
    num_selected_attributes = int(len(attribute_list)*num_attribute_fraction)
    root = buildtree(data_dict, attribute_list, num_selected_attributes, lines, Node)
    return root


def construct_random_forest(num_trees, num_attribute_fraction, num_tuple, lines):
    '''
    wrapper function to construct the random forest

    '''
    forest = []
    for i in range(num_trees):
        tree = construct_decision_tree(num_attribute_fraction, num_tuple, lines)
        forest.append(tree)
    return forest


def main():
    # open the training set file and store the data in memory
    training_file_path = sys.argv[1]
    with open(training_file_path) as fin:
        lines = fin.read().splitlines()

    # todo
    num_trees = 20
    num_attribute_fraction = 0.4
    num_tuple = int(len(lines)/10)
    forest = construct_random_forest(num_trees, num_attribute_fraction, num_tuple, lines)

    # open the training set file and store the data in memory
    testing_file_path = sys.argv[2]
    test_classifier(testing_file_path, forest)

    # compare the result with the standard sklearn
    run_standard(training_file_path, testing_file_path)


if __name__ == "__main__":
    main()