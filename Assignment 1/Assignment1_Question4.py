def print_with_commas(sentence):
    for i in range(len(sentence)):
        if sentence[i] == " ":
            print(",", end="")
        else:
            print(sentence[i], end="")
    print()


def print_words_on_new_lines(sentence):
    # Print each word on a new line
    current_word = ""
    for char in sentence:
        if char == " ":
            print(current_word)
            current_word = ""
        else:
            current_word += char
        print(current_word)

sentence = input("Please enter sentence:")
print_with_commas(sentence)
print_words_on_new_lines(sentence)