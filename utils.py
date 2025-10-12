def print_arr(arr, title, sz = -1):
    size = len(arr)
    if(sz >= 0):
        size = sz
    print(f"\n{title}:")
    for i in range(size):
        print(i, ":", arr[i])
    print("")

def count_word_matches(search_word, arr):
    cnt = 0
    for word in arr:
        if word == search_word:
            cnt += 1
    return cnt
