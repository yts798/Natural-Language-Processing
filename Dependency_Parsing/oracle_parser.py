import json
from dependency_parser import DepContext, DepParser


def normalise_word(word):
    """Normalise a word for feature extraction

    Args:
        word (str): input word

    Returns:
        str: normalised output word
    """
    return word.strip().lower()


def extract_features(ctx):
    """Extract some simple features from a DepContext object
    the features represent the current state.
    They would be useful for training a classifier. Note that training a
    classifier is not required for this question.

    Args:
        ctx (DepContext): The current dependency parser context

    Returns:
        list(str): a list of terms which represent the current context
    """
    #NOTE: You should not modify this function

    # get the top of the stack and the next word in the buffer
    stack_id = ctx.stack_top()
    if ctx.is_buffer_empty():
        buff_id = 0
    else:
        buff_id = ctx.next_buffer_word_id()

    # convert the ids to word strings
    stack_word = ctx.word_id_to_str(stack_id)
    buff_word = ctx.word_id_to_str(buff_id)

    # build a feature list from this context
    feats = [
                normalise_word(stack_word)+'_stack', 
                normalise_word(buff_word)+'_buffer',
            ]

    return feats


def parse_gold(dp, gold):
    """Generates a list of actions that generates a ground truth (gold) parse
    also returns a list of features extracted using the extract_features function

    Args:
        dp (DepParser): An initialised dependency parser object
        gold (list(tuple(int,int))): A ground truth parse for the sentence used to initialise dp

    Returns:
        list(str), list(list(str)): first return value is the list of actions 
        performed these are one of "left", "right", "reduce", "shift" ,
        the second return value is the list of features (called states in the code here).
        The list of actions and list of features should be in the order that the actions were performed
    """
    actions = []
    states = []

    # implementation of unlabelled dependency parsing.

    while True:
        # empty stack break directly
        if dp.ctx.is_buffer_empty():
            break
        
        # initialise
        ft = extract_features(dp.ctx)
        
        i = dp.ctx.stack_top()
        j = dp.ctx.next_buffer_word_id()
        
        # pre-calculate reduce condition is met
        reduce_met = False
        for x in range(i):
            if (j,x) in gold:
                reduce_met = True
            elif (x,j) in gold:
                reduce_met = True
                
        # try four operations in the requiredorder
        if (j,i) in gold:
            # try left arc
            if dp.left_arc():
                actions.append("left")

        elif (i,j) in gold:
            # try right arc
            if dp.right_arc():
                actions.append("right")
                
        # try reduce
        elif reduce_met and dp.reduce():
            actions.append("reduce")
            
        # try shift
        elif dp.shift():
            actions.append("shift")
        # nothing works
        else:
            break
            
        # save features
        states.append(ft)
    return actions, states






def run_small_test():
    """Runs a small test to help you check your algorithm is outputting in the correct format
    """
    rules = [('[ROOT]', 'A'),
            ('A', 'B'),
            ('B', 'C'),
            ('C', 'B')]

    test_sentences = ['AB']
    ground_truth = [
        [(0, 1), (1, 2)]
    ]

    # get the actions and features
    for sent,gold in zip(test_sentences, ground_truth):
        dp = DepParser(rules, list(sent))
        actions, states = parse_gold(dp, gold)
        print("Action:", actions)
        print("Features:", states)
        print("")

        # the expected output is:
        # Action: ['right', 'right']
        # Features: [['[root]_stack', 'a_buffer'], ['a_stack', 'b_buffer']]
        assert actions == ['right', 'right']
        assert states == [['[root]_stack', 'a_buffer'], ['a_stack', 'b_buffer']]


def run_large_test():
    """Runs a large test using a real set of rules. And parses of english sentences.
    The solutions are not provided as this is primarily for marking purposes.
    """

    # read in the rule set
    rules = json.load(open("rule_set.json", "r"))
    rules = [(r[0], r[1]) for r in rules["rules"]]

    # read in the sentences and their ground truth parses
    sents = json.load(open("gold_parse.json", "r"))
    test_sentences = sents["seqs"]
    test_gold = sents["parse"]

    # get the actions and states for all sentences
    good_count = 0
    fail_count = 0
    for sent,parse in zip(test_sentences, test_gold):
        # remove the label from the ground truth parse then run the parser
        parse = [(p[0], p[1]) for p in parse]
        dp = DepParser(rules, sent[1:])
        actions, states = parse_gold(dp, parse)

        sdeps = sorted(dp.ctx.get_all_dependencies())
        sparse = sorted(parse)
        if str(sdeps) != str(sparse):
            fail_count += 1
            if fail_count <= 3:
                print("Fail case %d" % fail_count)
                print(sent)
                print("Actions:", actions)
                print("Features:", states[:3], "....")
                print("Parser Dependencies:", end="")
                for i,j in sdeps:
                    print((dp.ctx.word_id_to_str(i), dp.ctx.word_id_to_str(j)), end="")
                print("")
                print("Ground Truth Dependencies:", end="")
                for i,j in sparse:
                    print((dp.ctx.word_id_to_str(i), dp.ctx.word_id_to_str(j)), end="")
                print("")
                print("--------")
        else:
            good_count += 1
            if good_count <= 3:
                print("Success case %d" % good_count)
                print("Sentence:", " ".join(sent))
                print("Actions:", actions)
                print("Features:", states[:3], "....")
                print("--------")

    print("Total fails: %d, Total success: %d" % (fail_count, good_count))


def main():
    # run a small test to check your model is working
    run_small_test()

    # run a much larger test
    run_large_test()

if __name__ == "__main__":
    main()

