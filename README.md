# MXNet
A Model-Driven Deep Neural Network for MRI Reconstruction with Unknown Sampling Pattern

## Abstract
Against the magnetic resonance imaging (MRI) reconstruction task, current deep learning based methods have achieved promising performance. Nevertheless, most of them are confronted with two main problems: 1) For most current MRI reconstruction methods, the down-sampling pattern is generally preset in advance, which makes it hard to flexibly handle the complicated real scenarios where the training data and the testing data are obtained under different sampling settings, thus constraining the model generalization capability. 2) They have not fully incorporated the physical imaging mechanism between the down-sampling pattern estimation and high-resolution MRI reconstruction into deep network design for this specific task.
To alleviate these issues, we propose a model-driven MRI reconstruction network called MXNet, which considers the relationship between the undersampling pattern and imaging by encoding the mask into the network.
Specifically, based on the MR physical imaging process, we first jointly optimize the down-sampling pattern and MRI reconstruction network. Then, based on the proposed optimization algorithm and the deep unfolding technique, we correspondingly construct the deep network where the physical imaging mechanism for MRI reconstruction is fully embedded into the entire learning process. Based on different settings between training data and testing data, with both consistent and inconsistent down-sampling patterns, extensive experiments comprehensively substantiate the effectiveness of our proposed MXNet in detail reconstruction as well as its fine generality. Moreover, we provide detailed model analysis and validate that our proposed framework shows fine generality and it can still accomplish superior performance when the downsampling mask is accurately available.

## Challenges in MRI Reconstruction
Despite deep learning promising results, these methods typically struggle with generalization across different or unknown sampling settings and often overlook the integration of the physical imaging mechanism of MRI into the network design. 

## Modeling of MXNet
A degradation process of MRI is 

$$Y=M \odot F X+{\varepsilon}$$

MRI Reconstruction model is 

$$
\min_{M, X} || M \odot F X - Y ||_{F}^2 + \lambda_1 R_1 (M) + \lambda_2 R_2 (X)
$$


To address the problem, firstly utilize relaxation techniquesto relax the constraints of $M_{i j} \in\{0,1}$, to allow $M_{i j} \in [0,1]$. Then, we introduce an auxiliary variable $Z$, defined as $Z=F X$. Then it can be transformed into the following optimization problem:

$$
\min_{Z, M, X} || M \odot Z - Y ||_{F}^2 + \lambda_1 R_1 (M) + \lambda_2 R_2 (X)
$$

$$
s.t. Z - F X = 0,  (1 - M) \odot Y = 0,
$$

**1) Updating $Z$:**

$$Z_{n+1}=\frac{\alpha{F} X_{n}+ M_{n} \odot Y}{\alpha{I} +  M_{n}^2}$$

**2) Updating $M$:**

$$
M_{n+1} = Prox_{\lambda_1}\left(\frac{\beta Y^2 + Y \odot Z_n}{\beta Y^2 + Z_{n+1}^2 + \omega}\right)
$$


**3) Updating $X$:**

$$
X_{n+1} = Prox_{\lambda_2/ \alpha}\left(F^{-1} Z_{n+1}\right)
$$


## Network architecture
![net-pic](https://github.com/sunliyangna0705/MXNet/blob/main/net.jpg)

## Experimental Results
setting 1: Consistent training-testing setting with noiseless

setting 2: Consistent training-testing setting with noisy

setting 3: Inconsistent training-testing setting with noiseless

setting 4: Inconsistent training-testing setting with noisy

**MRI Reconstruction under different setting**

![set1-brain](https://github.com/sunliyangna0705/MXNet/blob/main/PICS/set1-brain.jpg)
![set2-brain](https://github.com/sunliyangna0705/MXNet/blob/main/PICS/set2-brain.jpg)
![set3-brain](https://github.com/sunliyangna0705/MXNet/blob/main/PICS/set3-brain.jpg)
![set4-brain](https://github.com/sunliyangna0705/MXNet/blob/main/PICS/set4-brain.jpg)
![set1-knee](https://github.com/sunliyangna0705/MXNet/blob/main/PICS/set1-knee.jpg)
![set2-knee](https://github.com/sunliyangna0705/MXNet/blob/main/PICS/set2-knee.jpg)
![set3-knee](https://github.com/sunliyangna0705/MXNet/blob/main/PICS/set3-knee.jpg)
![set4-knee](https://github.com/sunliyangna0705/MXNet/blob/main/PICS/set4-knee.jpg)

**Downsampling patterns learned and the effects on image reconstruction in each stage**

![mask](https://github.com/sunliyangna0705/MXNet/blob/main/PICS/MX-mask.jpg)



