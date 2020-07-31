

class Word(str):
    def __eq__(self, other):
        return len(self) == len(other)

w1 = Word("aaa")
w2 = Word("bbb")
w3 = "aaa"
w4 = "bbb"

print(w1==w2)
print(w3==w4)
print(w1.isupper())


class CountList():
    def __init__(self, x):
        self.x = x
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        return X
    def getself(self):
        return self.x

c1 = CountList([1,3,5,7,9])
c2 = CountList([2,4,6,8,10])

# 调用
# print(c1)  ## 3
# print(len(c1))
# c2[1]  ## 4
# c1[1] + c2[1] 	## 7
# c1.count  ## {0:0,1:2,2:0,3:0,4:0}
# c2.count  ## {0:0,1:2,2:0,3:0,4:0}
print(c1)
print(c1.getself())
print(c1[0])

if isinstance(c1,CountList):
    print("YES")
