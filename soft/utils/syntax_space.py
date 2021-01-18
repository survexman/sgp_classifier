import copy
import random

from anytree import Node, RenderTree


class SyntaxSpace:

    def __init__(self, operators):
        self.operators = operators

    def get_term_operators(self):
        return [o for o in self.operators if o.is_term()]

    def get_types_by_return_type(self, return_type):
        return {o.o_type for o in self.operators if o.return_type == return_type}

    def get_operators_by_return_type(self, return_type):
        return [o for o in self.operators if o.return_type == return_type]

    def get_operator_by_name(self, name):
        for o in self.operators:
            if o.o_name == name:
                return o
        return None

    def get_random_operator(self, return_type, allowed_types = None, except_types = None, is_term = False):

        if not allowed_types:
            allowed_types = []

        if not except_types:
            except_types = []

        candidates = self.get_operators_by_return_type(return_type)

        if len(allowed_types) == 0:
            allowed_types = self.get_types_by_return_type(return_type)

        operators = [o for o in candidates if o.o_type not in except_types and o.o_type in allowed_types]

        if is_term:
            operators = [o for o in operators if o.is_term()]

        if len(operators) == 0:
            return None

        operator = copy.deepcopy(random.choice(operators))

        return operator

    def evaluate(self, signature: str, **kwargs):
        eval_str = signature
        for k, v in kwargs.items():
            eval_str = eval_str.replace(k, "[" + ",".join(str(bit) for bit in v) + "]")
        context = {}
        for o in self.operators:
            context[o.o_name] = o.func
        eval_str = eval_str.replace('\n', '')
        result = eval(eval_str, context)
        return result

    def generate_random_tree(self, return_type, min_height, max_height, starts_with = None, put_before_top = None):

        if not starts_with:
            root_node = Node(self.get_random_operator(return_type, except_types = self.get_term_operators()))
        else:
            root_node = Node(self.get_operator_by_name(starts_with))

        is_put_before_top = False

        while True:
            is_completed = True

            for _, _, node in RenderTree(root_node):
                o_type = node.name.o_type

                if node.name.arg_number == len(node.children):
                    continue

                if node.name.is_term():
                    continue
                children = []

                for input_type in node.name.input_types:
                    same_type_chain_len = 0
                    chain_node = node

                    while chain_node and chain_node.name.o_type == o_type:
                        same_type_chain_len += 1
                        chain_node = chain_node.parent

                    if put_before_top and not is_put_before_top:
                        child_operator = put_before_top
                        is_put_before_top = True
                    elif same_type_chain_len < min_height[o_type]:
                        child_operator = self.get_random_operator(input_type, allowed_types = [o_type])
                    elif same_type_chain_len >= max_height[o_type]:
                        child_operator = self.get_random_operator(input_type, except_types = [o_type])
                        if not child_operator:
                            child_operator = self.get_random_operator(input_type, is_term = True)
                    else:
                        child_operator = self.get_random_operator(input_type)

                    child_operator.update_value()
                    children.append(Node(child_operator))

                node.children = tuple(children)
                is_completed = False

            if is_completed:
                break

        return root_node
