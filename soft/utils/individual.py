import copy
import random

from anytree import RenderTree


class Individual:
    fitness_function = None

    def __init__(self, gene_tree, structure):
        self.gene_tree = gene_tree
        self.structure = structure
        self.signature = None
        self.fitness = None
        self.set_up()

    def set_up(self):
        self.signature = self.set_signature()
        self.fitness = self.__class__.fitness_function(self)

    @classmethod
    def set_fitness_function(cls, func):
        cls.fitness_function = func

    def set_signature(self):
        sign = "*"
        for _, _, node in RenderTree(self.gene_tree):
            node_signature = str(node.name)
            sign = sign.replace("*", node_signature, 1)
        return sign

    def __str__(self):
        return f"{self.signature} : {self.fitness}"

    def evaluate(self, **kwargs):
        return self.structure.evaluate(self.signature, **kwargs)

    def print(self):
        for pre, _, node in RenderTree(self.gene_tree):
            print("%s%s" % (pre, str(node.name)))

    def count_genes(self):
        count = 0
        for _, _, _ in RenderTree(self.gene_tree):
            count += 1
        return count

    def get_random_node(self):
        random_num = random.randint(0, self.count_genes() - 1)
        i = 0
        for _, _, node in RenderTree(self.gene_tree):
            if random_num == i:
                return node
            i += 1

    def get_random_node_by_operator_type(self, operator_type):
        candidates = self.get_nodes_by_operator_type(operator_type)
        if len(candidates) == 0:
            return None
        return random.choice(candidates)

    def get_nodes_by_operator_type(self, operator_type):
        nodes = []
        for _, _, node in RenderTree(self.gene_tree):
            if node.name.o_type == operator_type:
                nodes.append(node)
        return nodes

    def replace_node(self, old_node, new_node):
        new_node = copy.deepcopy(new_node)
        if old_node.is_root:
            self.gene_tree = new_node
        else:
            children = list(old_node.parent.children)
            for i in range(len(children)):
                if str(children[i].name) == str(old_node.name):
                    children[i] = new_node
                    break
            old_node.parent.children = children
