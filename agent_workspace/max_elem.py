def find_max(arr):
    max_val = arr[0]
    for num in arr:
        if num > max_val: 
            max_val = num
    return max_val

res = find_max([2, 5, 7, 2, 0, 5, 6, 7])
assert res == 7
print('MAX_Elem')