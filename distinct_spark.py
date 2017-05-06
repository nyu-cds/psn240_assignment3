from pyspark import SparkContext
from operator import add
import re

# remove any non-words and split lines into separate words
# finally, convert all words to lowercase
def splitter(line):
    line = re.sub(r'^\W+|\W+$', '', line)
    return map(str.lower, re.split(r'\W+', line))

if __name__ == '__main__':
    sc = SparkContext("local", "distinct words")
    text = sc.textFile('pg2701.txt')
    words = text.flatMap(splitter)
    words_mapped = words.map(lambda x: (x,1))
    sorted_map = words_mapped.sortByKey()
    #counts = sorted_map.reduceByKey(add)
    counts = sorted_map.reduceByKey(lambda accum, n: 1)
    #Either of the statements print the count of distinct words in the input text
    #print(counts.count())
    print(counts.values().sum())