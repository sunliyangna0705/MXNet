# MXNet
A Model-Driven Deep Neural Network for MRI Reconstruction with Unknown Sampling Pattern

## Abstract
Against this magnetic resonance imaging (MRI) reconstruction task, current deep learning based methods have achieved promising performance. Nevertheless, most of them are confronted with two main problems: 1) For most of current MRI reconstruction methods, the down-sampling pattern is generally preset and known in advance, which makes it hard to flexibly handle the complicated real scenario where the training data and the testing data are obtained under different sampling settings, thus constraining the model generalization capability. 2) They have not fully incorporated the physical imaging mechanism between the down-sampling pattern estimation and high-resolution MRI reconstruction into deep network design for this specific task.
To alleviate these issues, in this paper, we propose a model-driven MRI reconstruction network with unknown sampling pattern, called MXNet. Specifically, based on the MRI physical imaging process, we first jointly optimize the down-sampling pattern and the high-resolution MRI reconstruction. Then based on the proposed optimization algorithm and the deep unfolding technique, we correspondingly construct the deep network where the physical imaging mechanism for MRI reconstruction is fully embedded into the entire learning process. Based on different settings between training data and testing data, including consistent and inconsistent down-sampling patterns, extensive experiments comprehensively substantiate the effectiveness of our proposed MXNet in detail reconstruction as well as its fine generality, beyond the current state-of-the-art MRI reconstruction methods.

## Challenges in MRI Reconstruction
The learning of the down-sampling pattern $M$ is heuristic, and it cannot finely reflect the degradation process underlying the acquired undersampled measurement $y$, leading to the limited performance improvement; 2) The physical generation mechanism has not been fully embedded into network design for the joint optimization of $M$ and $X$.

## Modeling of MXNet
A degradation process of MRI is $Y=M \odot F X+{\varepsilon}$

MRI Reconstruction model is $ min _{M, X}{\Vert M \odot F X - Y \Vert }_{{F}}^2+\lambda_{1}\mathcal{R}_1(M)+\lambda_{2} \mathcal{R}_2(X)$

To address the problem, firstly utilize relaxation techniquesto relax the constraints of $\M_{i j} \in\{0,1\}$, to allow $\M_{i j} \in [0,1]$. Then, we introduce an auxiliary variable $\Z$, defined as $\Z=\F \X$. Then Eq. (\ref{optimize}) can be transformed into the following optimization problem:
$$
\min _{\Z, \M, \X}{\Vert\M \odot \Z-\Y\Vert_{{F}}^2+\lambda_{1} \mathcal{R}_1(\M)+\lambda_{2} \mathcal{R}_2(\X)} \\
%\text { s.t. } \quad \Z-\F \X=0, \\
\text { s.t. } \Z-\F \X=0,  \quad (1-\M) \odot \Y=0,
% &\quad (1-\M) \odot \Y=0,
$$
**1) Updating $Z$:**
$Z_{n+1}=\frac{\alpha{F} X_{n}+ M_{n} \odot Y}{\alpha{I} +  M_{n}^2}$
**2) Updating $M$:**
$\M_{n+1} = \text{Prox}_{\lambda_{1}}\left(\frac{\beta \Y^2+ \Y \odot \Z_{n}}{{\beta \Y^2 +\Z_{n+1}}^2+\omega}\right) $
**3) Updating $X$:**
$X_{n+1} = \text{Prox}_{\lambda_{2}/\alpha}\left(\F^{-1} \Z_{n+1}\right) $
## Network architecture
