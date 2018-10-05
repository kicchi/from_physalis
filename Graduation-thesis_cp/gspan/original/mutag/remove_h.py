#coding: UTF-8

import re
import sys
import csv
import codecs
sys.stdout = codecs.getwriter('utf8')(sys.stdout)


fi = open('mutag.gsp','r')
fo = open('mutag_wH.gsp','w')

dict = {}

g_count = 0
v_count = 0
e_count = 0

flag = True

while 1:
    line = fi.readline()
    if not line:
        break
    if line[:1]=="t":
        print dict
        print
        g_count +=1
        fo.write(line)
        print line,
        dict ={}
        continue
    if line[:1]=="v":
        if line.split()[2]=="1":
            dict[line.split()[1]]=1
            continue
    if line[:1]=="e":
        if dict.has_key(line.split()[1])==True:
            continue
        if dict.has_key(line.split()[2])==True:
            continue
    fo.write(line)
    print line,
print g_count
"""
    while flag:
        if len(line) > 1:
            if dict.has_key(line[:-1]) == False:
                flag = False
                print "not found"
                break
            flag = False
        line = fi.readline()
    if g_act == 0:
        while 1:
            look = fi.readline()
            if look[:4] == "$$$$":
                flag = True
                g_act = -1
                #print "Unspecified"
                break
    if not line:
        break
    if line[:6] == "M  END":
        while 1:
            look = fi.readline()
            if look[:4] == "$$$$":
                flag = True
                fo.write("\n")
                v_count = 0
                e_count = 0
                break
    if len(line) == 40:
        fo.write( "t # 0 " + str(g_act)+"\n")
        g_count += 1
        v_count = int(line[0:3])
        e_count = int(line[3:6])
        h_list = []
        for v in range(v_count):
            line = fi.readline()
            if dict.has_key(line[31:33]) == True:
                if dict[line[31:33]] != 0:
                    fo.write("v " + str(v) + " " + str(dict[line[31:33]]) + "\n")
                else:
                    h_list.append(v)
            else:
                dict[line[31:33]] = len(dict)
                fo.write("v " + str(v) + " " + str(dict[line[31:33]])+"\n")
        #print h_list
        for e in range(e_count):
            line = fi.readline()
            if not (int(line[0:3])-1 in h_list  or  int(line[3:6])-1 in h_list):
                fo.write("e " + str(int(line[0:3])-1) + " " \
                         + str(int(line[3:6])-1) + " " + str(int(line[8:9]))+"\n")
        """
