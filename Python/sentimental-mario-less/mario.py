def height(prompt):
    while True:
        try:
            value = int(input(prompt))
            if 1 <= value <= 8:
                return value
            else:
                print("please enter a positive value betwwen 1 and 8")
        except ValueError:
            print("Invalid input")


def helf_pyramid(height):
    for i in range(height):
        print(" " * (height - i - 1) + "#" * (i + 1))


# Main program
height = height("height: ")
helf_pyramid(height)
