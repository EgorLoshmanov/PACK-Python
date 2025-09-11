f = open("example.txt")

cnt_lines = 0
cnt_words = 0
cnt_letter = 0
for line in f:
    cnt_lines += 1
    cnt_words += len(line.split())
    for letter in line:
        if letter != '\n':
            cnt_letter += 1

print(f"Строк: {cnt_lines}")
print(f"Слов: {cnt_words}")
print(f"Букв: {cnt_letter}")

f.close()