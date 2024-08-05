from cs50 import get_float


def get_change():
    while True:
        change = get_float("change: ")
        if change >= 0:
            return change
        else:
            print("please enter a non-negative value")


def calculate_coin(change):
    cents = round(change * 100)
    coins = 0

    # calculate the number of quarters
    coins = coins + cents // 25
    cents = cents % 25
    # calculate the number of dimes
    coins = coins + cents // 10
    cents = cents % 10
    # calculate the number of nickels
    coins = coins + cents // 5
    cents = cents % 5
    # calculate the number of pennis
    coins = coins + cents

    return coins


# Main
change = get_change()
coins = calculate_coin(change)
print(coins)
