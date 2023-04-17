import re

# Define the input language (e.g., "x = 3")
pattern = re.compile(r"([a-zA-Z]+|\d+|\+|\-|\*|\/|\=|\()")

# Define a dictionary to store variables and their values
variables = {}

def parse_input(input_string):
    # Parse the input string and create an internal data structure
    tokens = pattern.findall(input_string)
    return tokens

def evaluate_expression(tokens):
    # Evaluate the expression recursively, following order of operations
    if len(tokens) == 1:
        if tokens[0] in variables:
            return variables[tokens[0]]
        else:
            try:
                return int(tokens[0])
            except ValueError:
                return float(tokens[0])
    elif len(tokens) == 3:
        op1, operator, op2 = tokens
        if operator == "+":
            return evaluate_expression([op1]) + evaluate_expression([op2])
        elif operator == "-":
            return evaluate_expression([op1]) - evaluate_expression([op2])
        elif operator == "*":
            return evaluate_expression([op1]) * evaluate_expression([op2])
        elif operator == "/":
            return evaluate_expression([op1]) / evaluate_expression([op2])
    else:
        # Handle nested expressions
        pass

def assign_variable(tokens):
    # Assign the value of the expression to a variable
    variable = tokens[0]
    expression = tokens[2:]
    variables[variable] = evaluate_expression(expression)

def evaluate_assignment(tokens):
    # Evaluate an assignment statement
    if tokens[1] == "=":
        assign_variable(tokens)

def main():
    # Read input from the user
    print("Add expressions (type 'quit' to exit):")
    while True:
        input_string = input("> ")
        if input_string == "quit":
            break

        # Parse the input and evaluate the expressions
        tokens = parse_input(input_string)
        if "=" in tokens:
            evaluate_assignment(tokens)
        elif "print" in tokens:
            variables_to_print = [token for token in tokens if token in variables]
            values_to_print = [str(variables[token]) for token in variables_to_print]
            print(" ".join(values_to_print))
        else:
            result = evaluate_expression(tokens)
            print(result)

    # Print the final variable values
    print(variables)

if __name__ == "__main__":
    main()
