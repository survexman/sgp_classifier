class Operator:

    def __init__(self, o_name, o_type, func, input_types, return_type, mutation_rule = None):
        self.o_name = o_name
        self.o_type = o_type
        self.func = func
        self.arg_number = len(input_types)
        self.input_types = input_types
        self.return_type = return_type
        self.mutation_rule = mutation_rule
        self.value = None

    def update_value(self):
        if self.is_term():
            self.value = self.func()

    def is_term(self):
        return self.arg_number == 0

    def signature(self, *args):

        if self.value:
            return str(self.value)

        if not self.value and self.is_term():
            self.update_value()
            return str(self.value)

        function_parameters = ", ".join([f"{args[i]}" for i in range(self.arg_number)])
        sign = f"{self.o_name}({function_parameters})"
        return sign

    def __str__(self):
        return self.signature(*(['*'] * self.arg_number))
