from random import randint

def play():
    random_int = randint(0,100)

    while True:
        user_guess = int(input("what number did we guess (0-100)?"))

        if user_guess == random_int:
            print(f"You found the number ({random_int}).Congrats!")
            break

        if user_guess < random_int:
            print("You number is less.")
            continue

        if user_guess > random_int:
            print("You number is more.")
            continue

if __name__ == '__main__':
    play()