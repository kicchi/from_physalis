# -*- coding: utf-8 -*-
import sys
import re



if(len(sys.argv) != 4):
    print 'Usage: # python %s datalist dataplace labellist' % sys.argv[0]
    quit()
listdata = sys.argv[1]
dataplace = sys.argv[2]
listlabel = sys.argv[3]
if __name__ == '__main__':
    data_list = open(listdata,'r')
    label_list = open(listlabel,'r')
    llist = label_list.readline().split()
    #print llist
    datacount = 0
    for data in data_list:
        datafile = data.replace('\n','')
        print "t", "#", datacount,
        if llist[datacount] == '1':
            print llist[datacount], datafile
        else:
            print "-1",datafile
        datacount += 1
        graph = open(dataplace+"/"+datafile,'r')
        for sline in graph:
            #print sline.replace('\n','').split(">")
            if sline[5:9] ==  "edge":
                print "e",re.split('[><" n]',sline)[8],re.split('[><" n]',sline)[12],"1"
            elif sline[5:9] ==  "node":
                print "v",re.split('[><" n]',sline)[9],
            elif sline[17:24] ==  "v_label":
                print re.split('[><" n]',sline)[11]
                #print re.split('[><" n]',sline)
        print
        graph.close()
    data_list.close()
    
    
