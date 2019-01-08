#===========================================================
# a tool to convert mnist data to tinn format training data
#
# Copyright(c) Aixi Wang
#===========================================================
f1 = open('train.bin','rb')
f2 = open('labels.bin','rb')

# skip header
f1.read(16)
# skip header
f2.read(8)

for i in range(30000):
    d = f1.read(784)
    n = f2.read(1)
    if len(d) == 784 and len(n) == 1:
        s1 = ''
        for x in d:
            s1 +=str(ord(x)/255.0) + ' '

        lb = [0,0,0,0,0,0,0,0,0,0]
        lb[ord(n)] = 1.0
        s2 = ''
        for y in lb:
            s2 += str(y) + ' '
        s3 = s1 + s2
        print s3.rstrip(' ')
    	#s = raw_input()
    else:
        time.sleep(1)






