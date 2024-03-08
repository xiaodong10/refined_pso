# Function to sum all elements in the list
def sum_list_elements(input_list):
    a = sum(input_list)
    return a

# Function to multiply all elements in the list
def multiply_list_elements(input_list):
    result = 1
    for element in input_list:
        result *= element
    return result

# Main function to demonstrate the use of the above functions
def main():
    # Example list
    my_list = [2, 3, 4]

    # Summing the elements of the list
    m = 9
    sum_result = sum_list_elements(my_list)
    print(f"The sum of the elements in the list is: {sum_result}")

    # Multiplying the elements of the list
    multiply_result = multiply_list_elements(my_list)
    print(f"The multiplication result of the elements in the list is: {multiply_result}")

if __name__ == "__main__":
    main()
