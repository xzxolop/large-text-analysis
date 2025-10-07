def print_arr(arr, title, sz = -1):
    size = len(arr)
    if(sz >= 0):
        size = sz
    print(f"\n{title}:")
    for i in range(size):
        print(i, ":", arr[i])
    print("")