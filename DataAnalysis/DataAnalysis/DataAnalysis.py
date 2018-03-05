from sklearn.utils import shuffle


x = [[1,1,1],[2,2,2],[3,0,3],[4,0,4]]
y = [1,2,3,4]
p = ["1","2","3","4"]

xs,ys,ps = shuffle(x,y,p)

for xx,yy,pp in zip(x,y,p):
    print()

k = []
k[0] = 1

print()









