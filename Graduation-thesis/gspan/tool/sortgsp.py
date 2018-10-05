# coding: utf-8


import sys 
 
argvs = sys.argv  
argc = len(argvs)

if (argc != 3):  
    print 'Usage: # python %s inputFilename makeFilename' % argvs[0]
    quit()
    


print argvs[2]
g_count = 0
new_gnum = 0
f = open(argvs[1],'r')
fp = open("positive",'w')
fn = open("negative",'w')
fw = open(argvs[2],'w')
while True:
    line = f.readline()
    if line[0:1] == "t":
        if line.split(' ')[3] == '1':
            while line != "\n":
                #print line,
                fp.write(line)
                line = f.readline()
            #print
            fp.write('\n')
        else:
            while line != "\n":
                #print line,
                fn.write(line)
                line = f.readline()
            #print
            fn.write('\n')
    else:
        break    
fp.close()
fn.close()

for line in open("negative","r"):
    print line,
    fw.write(line)
for line in open("positive","r"):
    print line,
    fw.write(line)

    
fw.close()
