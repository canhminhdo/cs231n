def quicksort(arr):

    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


print(quicksort([3, 6, 8, 10, 1, 2, 1]))

animals = ['cat', 'dog', 'horse']

for x in animals:
    print(x)

for idx, x in enumerate(animals):
    print(idx, x)

nums = [0, 1, 2, 3, 4, 5]

squares = [x ** 2 for x in nums]

print(squares)

even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares)

d = {'cat': 'cute', 'dog': 'furry'}
print(d.get('cat'))

print(d.get('dog', 'N/A'))
if 'cat' in d:
    print(d.get('cat'))

for animal in d:
    print(d[animal])

print(d.items())

print(range(10))


