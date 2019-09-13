自己实现的FM算法，参数说明：
------------------------------------------------
dim:fm的维度n

dim_lat:隐层v的列数维度k

σ：初始化v的时候，使用正态分布，μ=0，σ=1

alpha_w：w的learning rate中参数1

alpha_v:v的learning rate中参数1

β_w：w的learning rate中参数2

β_v:v的learning rate中参数2

λ_w1:w的一范数

λ_w2:w的二范数

λ_v1:v的一范数

λ_v2:v的二范数

zws：w的梯度加成Δ梯度后的值。也就是实际使用的梯度。

nws：冲量法计算的Δ梯度。也就是说nws是中间用来修正zws的临时变量。

zvs：v的梯度加成Δ梯度后的值。也就是实际使用的梯度。

nvs：冲量法计算的Δ梯度。也就是说nvs是中间用来修正zvs的临时变量。

梯度公式如下两种：
--------------------------------------------------------------
第一种：

fore = (self.beta_v + np.sqrt(self._nvs[i, f])) / self.alpha_v + self.lambda_v2

sign_zvs = -1. if self._zvs[i, f] < 0. else 1.

self.V[i, f] = -1. / fore * (self._zvs[i, f] - sign_zvs * self.lambda_v1)

第二种：

prediction = self.predict(sample)

base_grad = prediction - label

gradient = base_grad * sample[i]

sigma = (np.sqrt(self._nws[i] + gradient ** 2) - np.sqrt(self._nws[i])) / self.alpha_w

self._zws[i] += gradient - sigma * self.weights[i]

self._nws[i] += gradient ** 2