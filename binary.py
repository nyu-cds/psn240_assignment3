__author__ = 'Sudhir'
from itertools import permutations
from itertools import repeat

def zbits(n, k):
    result = set()
    for item in permutations(list(repeat(0,k))+list(repeat(1,n-k)), n):
        result.add(''.join(map(str, item)))
    return result

if __name__ == '__main__':
    assert zbits(4, 3) == {'0100', '0001', '0010', '1000'}
    assert zbits(4, 1) == {'0111', '1011', '1101', '1110'}
    assert zbits(5, 4) == {'00001', '00100', '01000', '10000', '00010'}