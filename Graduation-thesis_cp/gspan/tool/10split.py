# coding: utf-8


import sys 
import subprocess
argvs = sys.argv  
argc = len(argvs)
mkdirshape = "mkdir -p %s"


if (argc != 2):  
    print 'Usage: # python %s inputFilename' % argvs[0]
    quit()

    
def is_test(c,i):
    if c % 10 == i:
        return True
    else:
        return False

filename = argvs[1]
#frontname = filename.split('/')
dire = ""
for l in filename.split('/')[0:len(filename.split('/'))-1]:
    dire += l + '/'
name = filename.split('/')[len(filename.split('/'))-1]
head = name.split('.gsp')[0]
print "---------------info-------------------"
print "directory      :",dire
print "filename       :",name
print "make file head :",head
print "--------------------------------------"
print "make buckets for 10-fold cross validation ..."
place = dire+head+"buckets"
mkd = mkdirshape % (place)
print mkd
stdout_data,stderr_data = subprocess.Popen(mkd.strip().split(" "),stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()

for i in range(10):
    print "make bucket",i,"...",
    train = head+"_train"+str(i)+'.gsp'
    test  = head+"_test"+str(i)+'.gsp'
    print train,test
    g_count = 0
    f = open(filename,'r')
    ftr = open(place+'/'+train,'w')
    fte = open(place+'/'+test,'w')
    while True:
        line = f.readline()
        if line[0:1] == "t":
            #print g_count,i
            while line != "\n":
                if is_test(g_count,i):
                    fte.write(line)
                else:
                    ftr.write(line)
                line = f.readline()
            if is_test(g_count,i):
                fte.write('\n') 
            else:
                ftr.write('\n')          
        else:
            break
        g_count += 1
    f.close()
    ftr.close()
    fte.close()
