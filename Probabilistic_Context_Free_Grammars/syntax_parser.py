import json
import numpy as np
from collections import defaultdict


class RuleWriter(object):
    """
    This class is for writing rules in a format 
    the judging software can read
    Usage might look like this:

    rule_writer = RuleWriter()
    for lhs, rhs, prob in out_rules:
        rule_writer.add_rule(lhs, rhs, prob)
    rule_writer.write_rules()

    """
    def __init__(self):
        self.rules = []

    def add_rule(self, lhs, rhs, prob):
        """Add a rule to the list of rules
        Does some checking to make sure you are using the correct format.

        Args:
            lhs (str): The left hand side of the rule as a string
            rhs (Iterable(str)): The right hand side of the rule. 
                Accepts an iterable (such as a list or tuple) of strings.
            prob (float): The conditional probability of the rule.
        """
        assert isinstance(lhs, str)
        assert isinstance(rhs, list) or isinstance(rhs, tuple)
        assert not isinstance(rhs, str)
        nrhs = []
        for cl in rhs:
            assert isinstance(cl, str)
            nrhs.append(cl)
        assert isinstance(prob, float)

        self.rules.append((lhs, nrhs, prob))

        
    def write_rules(self, filename="q1.json"):
        """Write the rules to an output file.

        Args:
            filename (str, optional): Where to output the rules. Defaults to "q1.json".
        """
        json.dump(self.rules, open(filename, "w"))


# helper function to traverse and update to dictionaries
def lookup(sent):
    # save value following the term
    val_list = []
    # extract term
    term = sent[0]

    # case where value is string and recursion stops
    if isinstance(sent[1],str):
        val_list.append(sent[1])
    # traverse
    else:
        for result in sent[1:]:
            val_list.append(result[0])
            # recurssion
            lookup(result)


    # update the number of syntax translation occurance
    if (term, tuple(val_list)) in syntax_num:
        syntax_num[(term, tuple(val_list))] += 1
    else:
        syntax_num[(term, tuple(val_list))] = 1
        
    # update the number of term occurance
    if term in term_num:
        term_num[term] += 1
    else:
        term_num[term] = 1

    return


# load the parsed sentences
psents = json.load(open("parsed_sents_list.json", "r"))
# psents = json.load(open("smalltest.json", "r"))
# instantiate rulewriter
rule_writer = RuleWriter()
# record the number of syntax translation occurance
syntax_num = {}
# record the number of word occurance
term_num = {}

#traverse all sentences
for sent in psents:
    lookup(sent)

# iterate over key and values of dictionary to write rules
for k, v in syntax_num.items():
    # extract parent and child from keys
    par, chd = k
    #compute probability and write rules 
    rule_writer.add_rule(par, chd, v / term_num[par])
    
# save to files
rule_writer.write_rules("q1.json")


out = json.load(open("q1.json", "r"))

for s in out[:10]:
   print(s)
