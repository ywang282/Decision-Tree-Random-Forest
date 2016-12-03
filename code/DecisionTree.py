#!/usr/bin/python

import sys
import pprint


class node():
	def __init__(self):
		self.terminate = False
		self.attribute = None
		self.children = None




def store_training_data_in_memory(file_path):
	with open(file_path) as fin:
		lines = fin.read().splitlines()

	data_dict = {
		'labels' : {},
		'attr_val_pairs' : {}
	}
	for line in lines:
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


def main():
    # open the training set file and store the data in memory
    training_file_path = sys.argv[1]
    train_data_dict = store_training_data_in_memory(training_file_path)
    pprint.pprint(train_data_dict)

    # construct the decision tree
    decision_tree = construct_decision_tree(train_data_dict)

    # open the training set file and store the data in memory
    testing_file_path = sys.argv[2]
    # TODO

if __name__ == "__main__":
    main()