import math

a = math.atan(0.189)
#print(a)

a = math.atan(1)
#print(a)

k_b = 0.1893
#k_i = 0.1681
k_i = 0.0170
values_1 = abs((k_b-k_i)/(1+k_b*k_i))
print(values_1)
values_2 = math.atan(values_1)
print(values_2)
# 0.0205
# 0.1701
