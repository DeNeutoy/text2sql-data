#!/usr/bin/env python3
# Adapted by Jonathan K. Kummerfeld from https://github.com/clab/dynet/blob/master/examples/tagger/bilstmtagger.py





from typing import List, Dict, Any
import argparse
from collections import Counter
import pickle
import random
import json
import sys
import os

from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor
from parsimonious.nodes import RegexNode
import sqlparse
import numpy as np

JsonDict = Dict[str, Any]
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

def get_template(sql_tokens: List[str],
                 sql_variables: Dict[str, str],
                 sent_variables: Dict[str, str]):
    template = []
    for token in sql_tokens:
        if (token not in sent_variables) and (token not in sql_variables):
            template.append(token)
        elif token in sent_variables:
            # This is the case that we have the variable
            # in the sql variables but not the sentence variables.
            # Apparently this is denoted with a "".
            if sent_variables[token] == '':
                template.append(sql_variables[token])
            else:
                template.append(token)
        elif token in sql_variables:
            template.append(sql_variables[token])
    return template

def get_tokens_and_tags(sentence: List[str],
                        sent_variables: Dict[str, str]):
    """
    sentence: The string of the sentence.
    sent_variables: The variable in the sentence and it's actual string. e.g {'var0': 'texas'}
    """
    tokens = []
    tags = []
    for token in sentence:
        if (token not in sent_variables) or args.no_vars:
            tokens.append(token)
            tags.append("O")
        else:
            assert len(sent_variables[token]) > 0
            for word in sent_variables[token].split():
                tokens.append(word)
                tags.append(token)
    return tokens, tags

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
    split_sentence = sent.strip().split()
    tokens, tags = get_tokens_and_tags(split_sentence, sent_variables)

    sql_tokens = []
    for token in sql.strip().split():
        # Split variables into  " variable " token sequences.
        # TODO This is bizare, why are we spliting it like this
        # if token.startswith('"%') or token.startswith("'%"):
        #     sql_tokens.append(token[:2])
        #     token = token[2:]
        # elif token.startswith('"') or token.startswith("'"):
        #     sql_tokens.append(token[0])
        #     token = token[1:]

        # if token.endswith('%"') or token.endswith("%'"):
        #     sql_tokens.append(token[:-2])
        #     sql_tokens.append(token[-2:])
        # elif token.endswith('"') or token.endswith("'"):
        #     sql_tokens.append(token[:-1])
        #     sql_tokens.append(token[-1])
        token = token.replace('"', "'").replace("%", "")
        if token.endswith("(") and len(token) > 1:
            sql_tokens.append(token[:-1])
            sql_tokens.append(token[-1])

        # Split column names but not numbers
        #elif "." in token and not all([x.isnumeric() for x in token.split(".")]):
        #    toks = token.split(".")
        #    sql_tokens.append(toks[0])
        #    for t in toks[1:]:
        #        sql_tokens.append(".")
        #        sql_tokens.append(t)
        else:
            sql_tokens.append(token)

    template = get_template(sql_tokens, sql_variables, sent_variables)

    return (tokens, tags, ' '.join(template))

def get_tagged_data_for_query(data: JsonDict):
    # If we're splitting based on SQL queries,
    # we assign whole splits of questions which 
    # have a similar SQL template to the same split.
    dataset_split: str = data['query-split']
    for sent_info in data['sentences']:
        # Instead, if we're using the question split,
        # we take the split according to the individual question.
        if not args.query_split:
            dataset_split = sent_info['question-split']

        # Some datasets require splits because they are small.
        if args.split is not None:
            if str(args.split) == str(dataset_split):
                dataset_split = "test"
            else:
                dataset_split = "train"

        # Loop over the different sql statements with equivelent semantics
        for sql in data["sql"]:
            sql_variables = {}
            for variable in data['variables']:
                sql_variables[variable['name']] = variable['example']
            
            text = sent_info['text']
            text_vars = sent_info['variables']

            yield (dataset_split, insert_variables(sql, sql_variables, text, text_vars))

            # Some questions might have multiple equivelent SQL statements.
            # By default, we just use the first one. TODO(MARK) - Use the shortest?
            if not args.use_all_sql:
                break


def get_train_dev_test_splits(filename: str):
    train = []
    dev = []
    test = []
    for filename in args.data:
        with open(filename) as input_file:
            data = json.load(input_file)
            if type(data) != list:
                data = [data]
            for example in data:
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
    return train, dev, test

# Changes:
# Added AVG, SUM to agg_func
# added optional AS to table_name
# Made where_clause optional in query
# Add query to pos_value (TODO check this, very unclear if it is the correct way to handle this)
# Added optional DISTINCT inside agg
# added <> binary op
# added biexpr to agg to support "SELECT TABLE.COLUMN / TABLE2.COLUMN2 FROM ..."
# added optional extra agg clauses connected to each other by a binaryop, fixes e.g SUM ( STATEalias0.POPULATION ) / SUM ( STATEalias0.AREA )
# Added optional nested brackets inside DISTINCT for aggregates (common in yelp)

# still TODO: 
# GROUP BY, ORDER BY
# sort out difference between variable and string matching
# tablename in (tablename1, tablename2)
# 
# JOIN, seems hard.

# TODO:
# string set isn't an expr, it should only be in in_expr
# not all functions can take * as an argument.    
# orderby_clause   = ORDER BY order_clause
#order_clause     = ordering_term (ws "," ordering_term)*
#    ordering_term    = ws expr (ASC / DESC)?
#    limit            = LIMIT ws number

SQL_GRAMMAR2 = Grammar(
    r"""
    stmt             = (query ws ";") / (query ws)
    query            = (ws select_core groupby_clause orderby_clause ws limit) / (ws select_core groupby_clause ws limit) / (ws select_core orderby_clause ws limit) / (ws select_core groupby_clause) / (ws select_core orderby_clause) / (ws select_core)
    select_core      = (select_with_distinct select_results from_clause where_clause) / (select_with_distinct select_results from_clause) / (select_with_distinct select_results where_clause) / (select_with_distinct select_results)
    select_with_distinct = (SELECT DISTINCT) / SELECT
    select_results   = (ws select_result ws "," ws select_results) / (ws select_result)
    select_result    = sel_res_all_star / sel_res_tab_star / sel_res_val / sel_res_col

    sel_res_tab_star = name ".*"
    sel_res_all_star = "*"
    sel_res_val      = (expr AS wsp name) / expr
    sel_res_col      = col_ref (AS wsp name)

    from_clause      = FROM source
    source           = (ws single_source ws "," ws source) / (ws single_source)
    single_source    = source_table / source_subq
    source_table     = table_name (AS wsp name)
    source_subq      = ("(" ws query ws ")" AS ws name) / ("(" ws query ws ")")
    where_clause     = (WHERE wsp expr where_conj) / (WHERE wsp expr)
    where_conj       = (AND wsp expr where_conj) / (AND wsp expr) 
    
    groupby_clause   = (GROUP BY group_clause having_clause) / (GROUP BY group_clause)
    group_clause     = (ws expr ws "," group_clause) / (ws expr)
    having_clause    = HAVING ws expr

    orderby_clause   = ORDER BY order_clause
    order_clause     = (ordering_term ws "," order_clause) / ordering_term
    ordering_term    = (ws expr ordering) / (ws expr)
    ordering         = ASC / DESC
    limit            = LIMIT ws number

    col_ref          = (table_name "." column_name) / column_name
    table_name       = name
    column_name      = name
    ws               = ~"\s*"i
    wsp              = ~"\s+"i
    name             = ~"[a-zA-Z]\w*"i

    expr             = in_expr / like_expr / between_expr / binary_expr / unary_expr / source_subq / value / string_set
    like_expr        = value wsp LIKE ws string
    in_expr          = (value wsp NOT IN wsp expr) / (value wsp IN wsp expr)
    between_expr     = value BETWEEN wsp value AND wsp value
    binary_expr      = value ws binaryop ws expr
    unary_expr       = unaryop expr
    value            = parenval / number / boolean / function / col_ref / string
    parenval         = "(" ws expr ws ")"
    function         = (fname ws "(" ws DISTINCT ws arg_list_or_star ws ")") / (fname ws "(" ws arg_list_or_star ws ")")
    arg_list_or_star = arg_list / "*"
    arg_list         = (expr ws "," ws arg_list) / expr
    number           = ~"\d*\.?\d+"i
    string_set       = ws "(" ws string_set_vals ws ")"
    string_set_vals  = (string ws "," ws string_set_vals) / string
    string           = ~"\'.*?\'"i
    fname            = "COUNT" / "SUM" / "MAX" / "MIN" / "AVG"
    boolean          = "true" / "false"
    binaryop         = "+" / "-" / "*" / "/" / "=" / "<>" / ">=" / "<=" / ">" / "<" / ">" / "and" / "AND" / "or" / "OR"
    binaryop_no_andor = "+" / "-" / "*" / "/" / "=" / "<>" / "<=" / ">" / "<" / ">" 
    unaryop          = "+" / "-" / "not" / "NOT"
    
    SELECT   = ws "SELECT"
    FROM     = ws "FROM"
    WHERE    = ws "WHERE"
    AS       = ws "AS"
    AND      = ws "AND"
    DISTINCT = ws "DISTINCT"
    GROUP    = ws "GROUP"
    ORDER    = ws "ORDER"
    BY       = ws "BY"
    ASC      = ws "ASC"
    DESC     = ws "DESC"
    BETWEEN  = ws "BETWEEN"
    IN       = ws "IN"
    NOT      = ws "NOT"
    HAVING   = ws "HAVING"
    LIMIT    = ws "LIMIT"
    LIKE     = ws "LIKE"
    """
)

class SQLVisitor(NodeVisitor):
    grammar = SQL_GRAMMAR2

    def __init__(self):
        self.prod_acc = []

        for nonterm in self.grammar.keys():
            if nonterm != 'stmt':
                self.__setattr__('visit_' + nonterm, self.add_prod_rule)

    def generic_visit(self, node, visited_children):
        self.add_prod_rule(node)

    def add_prod_rule(self, node, children=None):
        if node.expr.name and node.expr.name != 'ws':
            rule = '{} ->'.format(node.expr.name)

            if isinstance(node, RegexNode):
                rule += '"{}"'.format(node.text)

            for child in node.__iter__():
                if child.expr.name != '':
                    rule += ' {}'.format(child.expr.name)
                else:
                    rule += ' {}'.format(child.expr._as_rhs())


            self.prod_acc = [rule] + self.prod_acc

    def visit_stmt(self, node, children):
        self.add_prod_rule(node)
        return self.prod_acc



def parse_dataset(filename: str):
    train, dev, test = get_train_dev_test_splits(filename)

    num_queries = 0
    num_parsed = 0
    sql_visitor = SQLVisitor()

    tests = ["SELECT TABLEalias0.ROW FROM TABLE AS TABLEalias0 WHERE TABLEalias0.ROW2 > 3 ORDER BY TABLEalias0.ROW ;",
            "SELECT TABLEalias0.ROW FROM TABLE AS TABLEalias0 ORDER BY TABLEalias0.ROW ;"]

    for (tokens, tags, template) in train + dev + test:
        num_queries += 1
        sql_visitor = SQLVisitor()
        query = template.strip("\n")
        try:
            prod_rules = sql_visitor.parse(query)
            num_parsed += 1
        except Exception as e:
            if "GROUP" not in template:
                print()
                print(e)
                print(" ".join(tokens))
                print(sqlparse.format(template, reindent=True))
    print(f"Parsed {num_parsed} out of {num_queries} queries, coverage {num_parsed/num_queries}")

parse_dataset(args.data)