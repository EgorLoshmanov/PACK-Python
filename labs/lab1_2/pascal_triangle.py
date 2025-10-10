import argparse

parser = argparse.ArgumentParser()
parser.add_argument("h", type=int)
arg = parser.parse_args()

pascal_triangle = []
for i in range(arg.h):
    row = [1] * (i + 1)
    for j in range(i + 1):
        if j != 0 and j != i:
            row[j] = pascal_triangle[i-1][j-1] + pascal_triangle[i-1][j]

    pascal_triangle.append(row)

max_width = len(str(max(pascal_triangle[-1]))) 
len_last_row = len(pascal_triangle[-1]) 
for row in pascal_triangle:
    cells = [f"{x:^{max_width}}" for x in row]
    line = ' '.join(cells) 
    indent = ((len_last_row - len(row)) * (max_width + 1)) // 2
    print(' ' * indent + line)



