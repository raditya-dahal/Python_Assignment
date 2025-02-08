#Question No 3

productPrice = [10, 14, 22, 33, 44, 13, 22, 55, 66, 77]
totalSum = 0
choice = None

print ("Supermarket \n =============")

while choice != "0":
    choice = input("Select product number (1-10) and 0 to Quit: ")

    if choice.isdigit():
        choice = int(choice)
        if choice == 0:
            break
        elif 1<= choice <= 10:
            Price = productPrice[choice-1]
            totalSum += Price
            print (f"Product: {choice} Price: {Price}")
        else:
            print ("Invalid choice. Please try again between 1-10 or 0 to quit.")
    else:
        print ("Invalid input. Please enter number")

print (f"Total Price: {totalSum}")
payment = int(input("Payment: "))
if payment >= totalSum:
    print(f"Change: {payment - totalSum}")
else:
    print("Insufficient payment")
