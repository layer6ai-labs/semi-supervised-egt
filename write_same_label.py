import sys
import pickle as pkl

q = open(sys.argv[1], "rb")
q = pkl.load(q)

x = open(sys.argv[2], "rb")
label_to_x = pkl.load(x)

#label_to_x = {}
#for k in x:
#    lb = x[k]
#    if lb not in label_to_x:
#        label_to_x[lb] = [k]
#    else:
#        label_to_x[lb].append(k)


o = open("label.txt", "w")
for k in q:
    lb = q[k]
    if type(lb) == list:
        written = False
        for l in lb:
            if l in label_to_x:
                if not written:
                    o.write(k + ",")
                    written = True
                score = 10000000
                for kk in label_to_x[l]:
                    o.write(kk + " " + str(score)  + " ")
                    score -= 1
        if written:
            o.write("\n")
            o.flush()

    elif lb in label_to_x:
        o.write(k + ",")
        score = 10000000
        for kk in label_to_x[lb]:
            o.write(kk + " " + str(score)  + " ")
            score -= 1
        o.write("\n")
        o.flush()
o.close()

