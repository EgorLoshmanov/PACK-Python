line = input()
lst_words = line.split()

def find_longest_word(lst):
    longest_word = lst_words[0]
    for word in lst_words:
        if len(word) > len(longest_word):
            longest_word = word
    return longest_word

print(find_longest_word(line))