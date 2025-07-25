import numpy as np

'''
矩阵array的定义
'''
arr = np.array([[1,2,3],
                [4,5,6]])

a = np.zeros((2,3))  # [[0. 0. 0.]
                     #  [0. 0. 0.]]
b = np.ones((2,3))   # [[1. 1. 1.]
                     #  [1. 1. 1.]]
c = np.empty((2,3))  # 生成未定义的array,会有随机数据

d = np.arange(10,20,2) # 和python的range类似
                       # [10 12 14 16 18]
e = np.arange(10,21,2).reshape((2,3)) # reshape规定矩阵形状
# [[10 12 14]
#  [16 18 20]]
f = np.linspace(1,10,5) # 生成[1，10]之间等距离，5个元素的序列
# [ 1.    3.25  5.5   7.75 10.  ]

# 拷贝
mycopy1 = np.array(arr)     # 深拷贝
mycopy2 = np.copy(arr)      # 深拷贝
mycopy3 = np.asarray(arr)   # 浅拷贝


'''
array的属性
'''
print(arr)
print(arr.shape)# (2, 3)
print(arr.dtype)# int64
print(arr.size)# 6
print(arr.ndim)# 2
# [[1 2 3]
#  [4 5 6]]

'''
数据的合并
'''
cars1 = np.array([5, 10, 12, 6])
cars2 = np.array([5.2, 4.2])
cars = np.concatenate([cars1, cars2])
print(cars)
# [ 5.  10.  12.   6.   5.2  4.2]



# 二维的叠加
test1 = np.array([5, 10, 12, 6])
test2 = np.array([5.1, 8.2, 11, 6.3])
# 首先需要把它们都变成二维，下面这两种方法都可以加维度
test1 = np.expand_dims(test1, 0)
test2 = test2[np.newaxis, :]

print("test1加维度后 ", test1)
print("test2加维度后 ", test2)
# 然后再在第一个维度上叠加
all_tests = np.concatenate([test1, test2])
print("扩展后\n", all_tests)

# test1加维度后  [[ 5 10 12  6]]
# test2加维度后  [[ 5.1  8.2 11.   6.3]]
# 扩展后
#  [[ 5.  10.  12.   6. ]
#   [ 5.1  8.2 11.   6.3]]




a = np.array([
[1,2,3],
[4,5,6]
])
b = np.array([
[7,8],
[9,10]
])

print(np.concatenate([a,b], axis=1))  # 这个没问题
# print(np.concatenate([a,b], axis=0))  # 这个会报错



a = np.array([[1,2],
              [3,4]])
b = np.array([[5,6],
              [7,8]])
print("竖直合并\n", np.vstack([a, b]))
print("水平合并\n", np.hstack([a, b]))
# 竖直合并
#  [[1 2]
#   [3 4]
#   [5 6]
#   [7 8]]
# 水平合并
#  [[1 2 5 6]
#   [3 4 7 8]]


'''
矩阵运算
'''
# dot矩阵点积，即矩阵直接相乘
a = np.array([
[1, 2],
[3, 4]
])
b = np.array([
[5, 6],
[7, 8]
])

print(a.dot(b))
# [[19 22]
#  [43 50]]
print(np.dot(a, b))
# [[19 22]
#  [43 50]]

'''
数据分析
'''
a = np.array([150, 166, 183, 170])
print("最大：", np.max(a))
print("最小：", a.min())
print('累加:',a.sum())

a = np.array([150, 166, 183, 170])
print("累乘：", a.prod())
print("总数：", a.size)

a = np.array([0, 1, 2, 3])
print("非零总数：", np.count_nonzero(a))

month_salary = [1.2, 20, 0.5, 0.3, 2.1]
print("平均工资：", np.mean(month_salary))
print("工资中位数：", np.median(month_salary))
print("标准差：", np.std(month_salary))

a = np.array([150.1, 166.4, 183.7, 170.8])
print("ceil:", np.ceil(a))
print("floor:", np.floor(a))
# ceil: [151. 167. 184. 171.]
# floor: [150. 166. 183. 170.]

a = np.array([150.1, 166.4, 183.7, 170.8])
print("clip:", a.clip(160, 180))
# clip: [160.  166.4 180.  170.8]