# -剑指offer刷题代码及思路(Python,牛客网)-

## 1 . 题目描述
## 在一个二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

# -*- coding:utf-8 -*-

class Solution:
    # array 二维列表
    def Find(self, target, array):
        m = len(array)
        n = len(array[0])
        i,j = m-1,0  #从左下角开始遍历
        
        while i>=0 and j<=n-1:
            k = array[i][j]
            if k == target:
                return True
            elif k>target:
                i-=1
            else:
                j+=1
        return False

## 2 . 题目描述
## 请实现一个函数，将一个字符串中的空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。
# -*- coding:utf-8 -*-
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        # write code here
        ss = s.split(' ')
        to_s = '%20'.join(ss)
        return to_s

## 3. 题目描述
## 输入一个链表，从尾到头打印链表每个节点的值。
class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
    def printListFromTailToHead(self, listNode):
        # write code here
        result = []
        if listNode==None:
            return result
        while listNode.next!=None:
            result.append(listNode.val)
            listNode = listNode.next
        result.append(listNode.val)
        return result[::-1]

## 4. 题目描述
## 输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        # write code here
        # 参数合法性判断
        if len(pre) == 0 :
            return None
        # 前序遍历的第一个结点一定是根结点
        root = pre[0]
        idx = tin.index(root)
        tr = TreeNode(root)
        # 递归构造左子树和右子树
        tr.left = self.reConstructBinaryTree(pre[1 : idx+1], tin[:idx])
        tr.right = self.reConstructBinaryTree(pre[1 + idx:], tin[idx+1:])
        return tr

5. 题目描述
用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。
https://www.jianshu.com/p/806109a2fd3b
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []
    def push(self, node):
        # write code here
        self.stack1.append(node)
    def pop(self):
        if self.stack2:
            return self.stack2.pop()
        elif self.stack1:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
            return self.stack2.pop()
        else:
            return None

6.题目描述
把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。 输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。 NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。

# -*- coding:utf-8 -*-
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        # write code here
        if len(rotateArray)==0:
            return 0
        else:
            min_v = None
            for i in rotateArray:
                if min_v == None or min_v>i:
                    min_v = i
        return min_v

7.题目描述
大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项。
n<=39
# -*- coding:utf-8 -*-
class Solution:
    def Fibonacci(self, n):
        # write code here
        lhm=[[0,1],[1,1]]
        rhm=[[0],[1]]
        em=[[1,0],[0,1]]
        #multiply two matrixes
        def matrix_mul(lhm,rhm):
            #initialize an empty matrix filled with zero
            result=[[0 for i in range(len(rhm[0]))] for j in range(len(rhm))]
            #multiply loop
            for i in range(len(lhm)):
                for j in range(len(rhm[0])):
                    for k in range(len(rhm)):
                        result[i][j]+=lhm[i][k]*rhm[k][j]
            return result

        def matrix_square(mat):
            return matrix_mul(mat,mat)
        #quick transform
        def fib_iter(mat,n):
            if not n:
                return em
            elif(n%2):
                return matrix_mul(mat,fib_iter(mat,n-1))
            else:
                return matrix_square(fib_iter(mat,n/2))
        return matrix_mul(fib_iter(lhm,n),rhm)[0][0]

http://www.jb51.net/article/81506.htm
https://blog.csdn.net/g_congratulation/article/details/52734306

题目描述
一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloor(self, number):
        # write code here
        lhm=[[0,1],[1,1]]
        rhm=[[1],[1]]
        em=[[1,0],[0,1]]
        #multiply two matrixes
        def matrix_mul(lhm,rhm):
            #initialize an empty matrix filled with zero
            result=[[0 for i in range(len(rhm[0]))] for j in range(len(rhm))]
            #multiply loop
            for i in range(len(lhm)):
                for j in range(len(rhm[0])):
                    for k in range(len(rhm)):
                        result[i][j]+=lhm[i][k]*rhm[k][j]
            return result

        def matrix_square(mat):
            return matrix_mul(mat,mat)
        #quick transform
        def fib_iter(mat,n):
            if not n:
                return em
            elif(n%2):
                return matrix_mul(mat,fib_iter(mat,n-1))
            else:
                return matrix_square(fib_iter(mat,n/2))
        return matrix_mul(fib_iter(lhm,number),rhm)[0][0]
https://www.cnblogs.com/luckyjason/p/5319379.html

题目描述
一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloorII(self, number):
        # write code here
        return 2**(number-1)

我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？

# -*- coding:utf-8 -*-
class Solution:
    def rectCover(self, number):
        # write code here
????????res = [0,1,2]
????????while len(res) <= number:
????????????res.append(res[-1] + res[-2])
????????return res[number]

题目描述
输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
http://www.cnblogs.com/zzxx-myblog/p/6435930.html
# -*- coding:utf-8 -*-
class Solution:
    def NumberOf1(self, n):
        # write code here
        count = 0
        if n < 0:
            n = n & 0xffffffff##是先把负数变成整数的表示方法？
        while n:
            count += 1
            #将整数与其减１后的数字做位于，能起到消掉最右边１的效果，循环后，如果整数不为０，那么一共有多少位为１，循环计数就为几
            n = (n-1)&n
        return count
https://www.2cto.com/kf/201709/677685.html

题目描述
输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。
# -*- coding:utf-8 -*-
class Solution:
    def reOrderArray(self, array):
        # write code here
        even_nums = []
        odd_nums =[]
        for i in array:
            if i % 2 ==1:
                odd_nums.append(i)
            else:
                even_nums.append(i)
        return odd_nums+even_nums

题目描述
输入一个链表，输出该链表中倒数第k个结点。
https://blog.csdn.net/qq_33431368/article/details/79251960
class Solution:
    def FindKthToTail(self, head, k):
        # write code here
        val_list = []
        while head !=None:
            val_list.append(head)
            head = head.next
        if k>len(val_list) or k<1:
            return 
        return val_list[-k]
http://blog.sina.com.cn/s/blog_1678238d80102wvfb.html

题目描述
输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        # write code here
        h=ListNode(0)
        to_h = h
        while pHead1 and pHead2:
            if pHead1.val<=pHead2.val:
                h.next=pHead1
                pHead1=pHead1.next
            else:
                h.next=pHead2
                pHead2=pHead2.next
            h = h.next
        if pHead1:      
                h.next=pHead1
        if pHead2: 
                h.next=pHead2
        return to_h.next

题目描述
输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）
class Solution:
    def HasSubtree(self, pRoot1, pRoot2):
        # write code here
        if pRoot1 == None or pRoot2 == None:
            return False

        result = False
        if pRoot1.val == pRoot2.val:
            result = self.isSubtree(pRoot1, pRoot2)
        if result == False:
            result = self.HasSubtree(pRoot1.left, pRoot2) | self.HasSubtree(pRoot1.right, pRoot2)
        return result

    def isSubtree(self, root1, root2):
        if root2 == None:
            return True
        if root1 == None:
            return False
        if root1.val == root2.val:
            return self.isSubtree(root1.left, root2.left) & self.isSubtree(root1.right, root2.right)
        return False

https://blog.csdn.net/u010005281/article/details/79460325

题目描述
操作给定的二叉树，将其变换为源二叉树的镜像。
class Solution:
    # 返回镜像树的根节点
    def Mirror(self, root):
        # write code here
        if not root:
            return root
        if not root.right and not root.left:
            return root             
        if root.right or root.left:
            t=root.right
            root.right=root.left
            root.left=t
        self.Mirror(root.right)
        self.Mirror(root.left)
        return root

题目描述
输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.
class Solution:
    # matrix类型为二维列表，需要返回列表
    def printMatrix(self, matrix):
        # write code here
        result = []
        while(matrix):
            result+=matrix.pop(0)
            if not matrix or not matrix[0]:
                break
            matrix = self.turn(matrix)
        return result
    def turn(self,matrix):
        num_r = len(matrix)
        num_c = len(matrix[0])
        newmat= [[matrix[j][i] for j in range(num_r)] for i in range(num_c)]
        newmat.reverse()
        return newmat

题目描述
定义栈的数据结构，请在该类型中实现一个能够得到栈最小元素的min函数。
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.stack = []
        self.min_stack = []
    def push(self, node):
        if not self.min_stack or node <= self.min_stack[-1]:
            self.min_stack.append(node)
        self.stack.append(node)
    def pop(self):
        if self.top() == self.min_stack[-1]:
            self.min_stack.pop()
        self.stack.pop()
    def top(self):
        return self.stack[-1]
     
    def min(self):
        return self.min_stack[-1]
http://www.revotu.com/coding-interviews-python-solutions.html

题目描述
输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4，5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）
class Solution:
    def IsPopOrder(self, pushV, popV):
        stack = []
        index = 0
        for v in pushV:
            stack.append(v)
            while stack and stack[-1] == popV[index]:
                stack.pop()
                index += 1
        return len(stack) == 0



https://www.jianshu.com/p/fa9dcbc88a8e
https://www.cnblogs.com/txlstars/p/5712170.html
http://www.revotu.com/coding-interviews-python-solutions.html

