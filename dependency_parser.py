class DepContext(object):
    """This class stores the current state of the dependency parser.
    WARNING: part of this question will be automatically graded
             changing this class could result in the grading failing.
             Also make sure you use the methods provided rather than 
             directly modifying the attributes.
    """

    def __init__(self, input_words):
        # prepend the root tag to the list of input words
        self.__words = ["[ROOT]"] + input_words

        # the input buffer is represented as a list of word ids
        # we start from 1 because the root with id 0 is already on the stack
        self.__word_ids = list(range(1, len(self.__words)))
        # we reverse the list because removing from the end 
        # of a list is more efficient than from the front
        self.__word_ids.reverse()

        # the list of dependencies found by the system so far
        self.__deps = []
        # the stack with the id of the root already added
        self.__stack = [0]

    def root_id(self):
        """Returns the id of the root

        Returns:
            int: the id of the root node
        """
        return 0

    def word_id_to_str(self, id):
        """Converts a word id to the string of the word.

        Args:
            id (int): The word id to convert

        Returns:
            str: the word represented by id
        """
        return self.__words[id]

    def next_buffer_word_id(self):
        """Gets the id of the next word in the buffer (aka input sentence).

        Returns:
            int : id of the next word in the buffer
        """
        return self.__word_ids[-1]
    
    def pop_buffer(self):
        """Removes and returns the next word id from the buffer.

        Returns:
            int : the word removed from the buffer
        """
        return self.__word_ids.pop()

    def is_buffer_empty(self):
        """Checks if the buffer is empty.

        Returns:
            boolean : returns True if the buffer is empty, False otherwise
        """
        if self.__word_ids:
            return False
        return True

    def stack_top(self):
        """Get the word id at the top of the stack

        Returns:
            int: Word id at the top of the stack
        """
        return self.__stack[-1]

    def push_stack(self, v):
        """Add a word id to the top of the stack

        Args:
            v (int): add a word id to the top of the stack
        """
        assert isinstance(v, int)
        self.__stack.append(v)
    
    def pop_stack(self):
        """Remove and return the top word id from the stack

        Returns:
            int: The word id just removed from the stack
        """
        return self.__stack.pop()

    def is_stack_empty(self):
        """Check if the stack is empty

        Returns:
            boolean : True if the stack is empty, False otherwise
        """
        if self.__stack:
            return False
        return True
    
    def add_dependency(self, src, tgt):
        """Stores a dependency relation from src to tgt
        src -> tgt

        Args:
            src (int): The word id that is the source of the dependency
            tgt (int): The word id that is the target of the dependency
        """
        assert isinstance(src, int)
        assert isinstance(tgt, int)
        self.__deps.append((src, tgt))
    
    def has_dependency_with_target(self, tgt):
        """Check if a word id is already the target of a dependency
        returns True iff tgt it is the target of a stored dependency

        Args:
            tgt (int): the word id to check

        Returns:
            boolean : True if tgt is the target of a dependency, False otherwise
        """
        assert isinstance(tgt, int)
        for a,b in self.__deps:
            if b == tgt:
                return True
        return False

    def get_all_dependencies(self):
        """Return a list of all the dependencies

        Returns:
            list(tuple(int, int)): list of dependencies
        """
        return list(self.__deps)


class DepParser(object):
    def __init__(self, rules, input_words):

        # create the context object
        # this stores the current state of the parser
        self.ctx = DepContext(input_words)

        # the rules of the grammar
        # each rule: lhs -> rhs 
        # is a tuple (lhs, rhs)
        self.rules = set(rules)

    def in_rules(self, rule):
        return rule in self.rules

    def left_arc(self):
        """Check if a left arc can be performed. 
        If so perform the left arc by modifying self.ctx


        Returns:
            boolean: True if left arc was performed, else False
        """
        # empty stack return false directly
        if self.ctx.is_buffer_empty():
            return False
        
        else:
            # extract values
            v_i = self.ctx.stack_top()
            v_j = self.ctx.next_buffer_word_id()
            w_i = self.ctx.word_id_to_str(v_i)
            w_j = self.ctx.word_id_to_str(v_j)

            #check the two conditions:
            #whether (vj->vi) is in the rule
            #whether vi has no dependency
            
            # check condition 1 in slides
            if not self.ctx.word_id_to_str(v_i) == 'ROOT':
                # check condition 2 in slides
                if self.in_rules((w_j, w_i)):
                    # check condition 3 in slides
                    if not self.ctx.has_dependency_with_target(v_i):
                        # all met! perform actions
                        self.ctx.add_dependency(v_j, v_i)
                        self.ctx.pop_stack()
                        return True
        # not met cases
        return False


    def right_arc(self):
        """Check if a right arc can be performed. 
        If so perform the right arc by modifying self.ctx


        Returns:
            boolean: True if right arc was performed, else False
        """
        # empty stack return false directly
        if self.ctx.is_buffer_empty():
            return False
        
        else:
            # extract values
            v_i = self.ctx.stack_top()
            v_j = self.ctx.next_buffer_word_id()
            w_i = self.ctx.word_id_to_str(v_i)
            w_j = self.ctx.word_id_to_str(v_j)

            # check condition 1 in slides
            if self.in_rules((w_i, w_j)):
                # check condition 2 in slides
                if not self.ctx.has_dependency_with_target(v_j):
                    # all met! perform actions
                    self.ctx.add_dependency(v_i, v_j)
                    self.ctx.pop_buffer()
                    self.ctx.push_stack(v_j)
                    return True
        # not met cases
        return False

    def reduce(self):
        """Check if reduce can be performed. 
        If so perform reduce by modifying self.ctx


        Returns:
            boolean: True if reduce was performed, else False
        """
        # empty stack return false directly
        if self.ctx.is_stack_empty():
            return False
        
        # not empty stack
        else:
            # check if top value from stack has dependency with target
            if self.ctx.has_dependency_with_target(self.ctx.stack_top()):
                # pop
                self.ctx.pop_stack()
                return True
            else:
                return False
            
        return False


    def shift(self):
        """Check if shift can be performed. 
        If so perform shift by modifying self.ctx


        Returns:
            boolean: True if shift was performed, else False
        """

        # check that there is still a wordid left in the buffer
        if not self.ctx.is_buffer_empty():

            # get the next word from the buffer, and remove it from the buffer
            wid = self.ctx.pop_buffer()

            # add the word to the top of the stack
            self.ctx.push_stack(wid)

            return True
        
        return False


def parse(dp):
    """Given an already initialised DepParser object
    run the parser until no operation can be performed.
    Should implement the operations in the order:
    left-arc, right-arc, reduce, shift. If an operation succeeds
    then start trying from left-arc again. This is the deterministic
    method mentioned in lectures.

    Args:
        dp (DepParser): the dependency parser object to parse

    Returns:
        boolean: True if the buffer is empty when no more operations can be performed,
        False if the buffer is not empty when no more operations can be performed.
    """
    while True:
        # flag variable to determine whether to continue
        cont = False
        # try four operations in the required order
        for oper in [dp.left_arc, dp.right_arc, dp.reduce, dp.shift]:
            if oper():
                cont = True
                break
                
        # should not continue
        if not cont:
            if dp.ctx.is_buffer_empty():
                return True
            else:
                return False


def main():

    # set of rules in a simple test grammar
    # each rule: lhs -> rhs   is a tuple (lhs, rhs)
    # eg: [ROOT] -> A
    # A -> B
    # ....
    rules = [('[ROOT]', 'A'),
            ('A', 'B'),
            ('B', 'C'),
            ('C', 'B')]

    # a set of test sentences in the simple grammar
    test_sentences = ['AB',
                    'CBA',
                    'ABC',
                    'ABCBC',
                    'BAC']

    # run the dependency parser on each of the test sentences
    for sent in test_sentences:
        dp = DepParser(rules, list(sent))
        res = parse(dp)

        print("Return value:", res)
        print("Dependencies:", dp.ctx.get_all_dependencies())
    
    # TIP:
    # The output for "AB" should be:
    # Return value: True
    # Dependencies: [(0, 1), (1, 2)]


if __name__ == "__main__":
    main()

