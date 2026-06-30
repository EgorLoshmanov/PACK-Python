dct = {"быстрый": "скорый", "красивый": "прекрасный", "умный": "разумный"} # Пример возможного словаря

words = ''
line = input()
words = line.split()

for word in words:
    if word in dct:
        print(dct[word], end=' ')
    else:
        print(word, end=' ')