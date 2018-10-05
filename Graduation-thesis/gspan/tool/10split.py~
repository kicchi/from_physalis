# coding: utf-8


import sys 
 
argvs = sys.argv  
argc = len(argvs)

if (argc != 4):  
    print 'Usage: # python %s inputFilename makeFilename [h or q or else]' % argvs[0]
    print 'h = half, q = quarter, else : any integer (> 0)'
    quit()

N = 0
if argvs[3] == 'h':
    N = 2
elif argvs[3] == 'q':
    N = 4
else:
    N = int(argvs[3])
    
def not_remove(c):
    if c % N == 0:
        return True
    else:
        return False

print argvs[2]
g_count = 0
new_gnum = 0
f = open(argvs[1],'r')
fw = open(argvs[2],'w')
while True:
    line = f.readline()
    if line[0:1] == "t":
        #print line.replace('\n','').split(' ')[3],
        while line != "\n":
            if not_remove(g_count):
                print line,
                fw.write(line)
            line = f.readline()
        if not_remove(g_count):
            print
            fw.write('\n')          
    else:
        break
    g_count += 1

print g_count
fw.close()
