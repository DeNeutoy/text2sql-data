#!/usr/bin/env python3
# Adapted by Jonathan K. Kummerfeld from https://github.com/clab/dynet/blob/master/examples/tagger/bilstmtagger.py

from typing import List, Dict, Any
import argparse
from collections import Counter
import pickle
import random
import json
import sys

print("Running:\n"+ ' '.join(sys.argv))

import numpy as np

## Command line argument handling and default configuration ##

parser = argparse.ArgumentParser(description='A simple template-based text-to-SQL system.')

# IO
parser.add_argument('data', help='Data in json format', nargs='+')
parser.add_argument('--query_split', help='Use the query split rather than the question split', action='store_true')
parser.add_argument('--no_vars', help='Run without filling in variables', action='store_true')
parser.add_argument('--use_all_sql', help='Default is to use first SQL only, this makes multiple instances.', action='store_true')
parser.add_argument('--do_test_eval', help='Do the final evaluation on the test set (rather than dev).', action='store_true')
parser.add_argument('--split', help='Use this split in cross-validation.', type=int)

args = parser.parse_args()
JsonDict = Dict[str, Any]
## Input ##

def get_template(sql_tokens: List[str],
                 sql_variables: List[Dict[str, str]],
                 sent_variables: Dict[str, str]):
    template = []
    for token in sql_tokens:
        if (token not in sent_variables) and (token not in sql_variables): # I think this second clause is always false
            template.append(token)
        elif token in sent_variables:
            if sent_variables[token] == '':
                # sentence variables can be empty. If so,
                # find the variable in the sql_variables and use that name.
                example = None
                for variable in sql_variables:
                    if variable['name'] == token:
                        example = variable['example']
                assert example is not None
                template.append(example)
            else:
                template.append(token)
        elif token in sql_variables:
            example = None
            for variable in sql_variables:
                if variable['name'] == token:
                    example = variable['example']
            assert example is not None
            template.append(example)
    return template

def insert_variables(sql: str,
                     sql_variables: List[Dict[str, str]],
                     sent: str,
                     sent_variables: Dict[str, str]):

    """
    sql: The string sql query.
    sql_variables: The variables extracted from the sentence and sql query.
        e.g. [{'example': 'arizona', 'location': 'both', 'name': 'var0', 'type': 'state'}] 
    sent: The string of the sentence.
    sent_variables: The variable in the sentence and it's actual string. e.g {'var0': 'texas'}
    """
    tokens = []
    tags = []
    for token in sent.strip().split():
        if (token not in sent_variables) or args.no_vars:
            tokens.append(token)
            tags.append("O")
        else:
            assert len(sent_variables[token]) > 0
            for word in sent_variables[token].split():
                tokens.append(word)
                tags.append(token)

    sql_tokens = []
    for token in sql.strip().split():
        # Split variables into  " variable " token sequences.
        if token.startswith('"%') or token.startswith("'%"):
            sql_tokens.append(token[:2])
            token = token[2:]
        elif token.startswith('"') or token.startswith("'"):
            sql_tokens.append(token[0])
            token = token[1:]

        if token.endswith('%"') or token.endswith("%'"):
            sql_tokens.append(token[:-2])
            sql_tokens.append(token[-2:])
        elif token.endswith('"') or token.endswith("'"):
            sql_tokens.append(token[:-1])
            sql_tokens.append(token[-1])
        else:
            sql_tokens.append(token)

    template = get_template(sql_tokens, sql_variables, sent_variables)

    return (tokens, tags, ' '.join(template))

def get_tagged_data_for_query(data: JsonDict):
    # If we're splitting based on SQL queries,
    # we assign whole splits of questions which 
    # have a similar SQL template to the same split.
    dataset: str = data['query-split']
    for sent_info in data['sentences']:
        # Instead, if we're using the question split,
        # we take the split according to the individual question.
        if not args.query_split:
            dataset = sent_info['question-split']

        # Some datasets require splits because they are small.
        if args.split is not None:
            if str(args.split) == str(dataset):
                dataset = "test"
            else:
                dataset = "train"

        for sql in data["sql"]:
            sql_vars = data['variables']
            text = sent_info['text']
            text_vars = sent_info['variables']

            yield (dataset, insert_variables(sql, sql_vars, text, text_vars))

            # Some questions might have multiple equivelent SQL statements.
            # By default, we just use the first one. TODO(MARK) - Use the shortest?
            if not args.use_all_sql:
                break

train = []
dev = []
test = []
for filename in args.data:
    with open(filename) as input_file:
        data = json.load(input_file)
        index= 0

        if type(data) != list:
            data = [data]
        for example in data:
            index+=1
            if index == 2:
                break
            for dataset, instance in get_tagged_data_for_query(example):
                if dataset == 'train':
                    train.append(instance)
                elif dataset == 'dev':
                    if args.do_test_eval:
                        train.append(instance)
                    else:
                        dev.append(instance)
                elif dataset == 'test':
                    test.append(instance)
                elif dataset == 'exclude':
                    pass
                else:
                    assert False, dataset
        