import os
print('Hello World')

x = 3
y = -1.9
text = 'hello'
array_numbers = [1, 2, 8, 9]

text = '60'
print('x is : ', x)
x = int(text)
print('x is : ', x)

x = 30
text = str(x)
print('x is : ', x)

x = 3 ** 2
print('x is : ', x)

x = 5
y = 2
r = x % y
print('r is : ', r)

print(os.getcwd())

print(os.chdir('../'))

print(os.getcwd())

a = 10
b = 6

if a == b:
    print("a = b")
elif a > b:
    print("a > b")
else:
    print("a < b")

move = 'LULRRRDDLLURDLRURRRRRUUUUU'
x = 0
y = 0

for j in range(len(move)):
    print(move[j])
    match move[j]:
        case 'L':
            x -= 1
        case 'R':
            x += 1
        case 'U':
            y -= 1
        case 'D':
            y += 1

print((x, y))

#position = [x, y]
position = [0, 0]

move = 'LULRRRDDLLURDLRURRRRRUUUUU'
def update(list):
    print(list)
    position[list[0]] = position[list[0]] + list[1]

switcher={
 'L': [0, -1],
 'R': [0, 1],
 'U': [1, 1],
 'D': [1, -1]
}

for j in range(len(move)):
    print(move[j])
    update(switcher[move[j]])

print(position)
