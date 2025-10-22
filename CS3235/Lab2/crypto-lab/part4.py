# Returns a new list with each of the numbers in the input squared
# Does not modify the input list
def square_list(input):
    output = []
    
    for v in input: 
        output.append(v * v)


    return output


# Feel free to edit the main function however you like to help you debug, it won't be graded
#
# Run this script with the command: python3 part4.py
# or select "Part 4" from the VS Code debugger
def main():
    a = [1, 2, 3, 4]
    b = square_list(a)

    print('a:', a)
    print('b:', b)


if __name__ == '__main__':
    main()
