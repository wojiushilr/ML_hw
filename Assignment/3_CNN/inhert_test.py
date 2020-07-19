#!/usr/bin/python
# -*- coding: UTF-8 -*-
from torch._six import container_abcs
from itertools import repeat

class FooParent(object):
    def __init__(self):
        self.parent = 'I\'m the parent.'
        # print('Parent')

    def bar(self, message):
        print("%s from Parent" % message)


class FooChild(FooParent):
    def __init__(self):
        # super(FooChild,self) 首先找到 FooChild 的父类（就是类 FooParent），然后把类 FooChild 的对象转换为类 FooParent 的对象
        super(FooChild, self).__init__()
        #print('Child')
        self.child = "I\'m the child."

    def bar(self, message):
        super(FooChild, self).bar(message)
        print('Child bar fuction')
        print(self.parent)
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse



if __name__ == '__main__':
    fooChild = FooChild()
    print(fooChild.parent)
    print(fooChild.child)
    # fooChild.bar('HelloWorld')
    _single = _ntuple(1)
    _pair = _ntuple(2)
    _triple = _ntuple(3)
    _quadruple = _ntuple(4)
    kernel_size = _pair(3)
    print(kernel_size)