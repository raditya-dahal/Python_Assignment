#Question Number 2


from random import choice

shopping_cart = []
choice = None
while choice != "3":
    print("Would you like to \n Add or \n Remove items or \n Quite?:", end="")
    choice = input()
    if choice == "1":
        item_name = input("What will be added?: ")
        shopping_cart.append(item_name)
    elif choice == "2":
        if shopping_cart:
            print(f"There are {len(shopping_cart)} items in the shopping cart.")
            index = input("Which item is deleted?: ")
            if index.isdigit():
                index = int(index)
                if 0 <= index < len(shopping_cart):
                    shopping_cart.pop(index)
                else:
                    print("Incorrect selection")
            else:
                print("Incorrect selection")
        else:
            print("The list is empty")
    elif choice == "3":
        print("The following items remain in the shopping cart:")
        for item in shopping_cart:
            print(item)

    else:
        print("Incorrect selection")
