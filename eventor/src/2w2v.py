'''
Created on Feb 20, 2018

@author: judeaax
'''
from re import split
from sys import argv

new_lines = []
already_seen = set()
with open(argv[1]) as f:
    lines = f.read().splitlines()
    for i in range(len(lines)):
        print(i / len(lines))
        values = split('\s+', lines[i])
        if len(values) == 1:
            continue
        if values[0] == '':
            values = values[1:]
        if len(values) == 400:
            # empty string
            continue
        if len(values) != 401:
            print("skip ", str(len(values)), str(values))
            continue
        if values[0] in already_seen:
            print("skip ealready seen: ", values[0])
            continue
        already_seen.add(values[0])
        new_lines.append(" ".join(values))

print("Writing to ", argv[2])

with open(argv[2], 'w') as f:
    f.write('%d %d\n' % (len(new_lines), 400))
    f.write('\n'.join(new_lines))
