# MXNet
A Model-Driven Deep Neural Network for MRI Reconstruction with Unknown Sampling Pattern

## Abstract
Against this magnetic resonance imaging (MRI) reconstruction task, current deep learning based methods have achieved promising performance. Nevertheless, most of them are confronted with two main problems: 1) For most of current MRI reconstruction methods, the down-sampling pattern is generally preset and known in advance, which makes it hard to flexibly handle the complicated real scenario where the training data and the testing data are obtained under different sampling settings, thus constraining the model generalization capability. 2) They have not fully incorporated the physical imaging mechanism between the down-sampling pattern estimation and high-resolution MRI reconstruction into deep network design for this specific task.
To alleviate these issues, in this paper, we propose a model-driven MRI reconstruction network with unknown sampling pattern, called MXNet. Specifically, based on the MRI physical imaging process, we first jointly optimize the down-sampling pattern and the high-resolution MRI reconstruction. Then based on the proposed optimization algorithm and the deep unfolding technique, we correspondingly construct the deep network where the physical imaging mechanism for MRI reconstruction is fully embedded into the entire learning process. Based on different settings between training data and testing data, including consistent and inconsistent down-sampling patterns, extensive experiments comprehensively substantiate the effectiveness of our proposed MXNet in detail reconstruction as well as its fine generality, beyond the current state-of-the-art MRI reconstruction methods.

## Challenges in MRI Reconstruction
The learning of the down-sampling pattern $\M$ is heuristic, and it cannot finely reflect the degradation process underlying the acquired undersampled measurement $y$, leading to the limited performance improvement; 2) The physical generation mechanism has not been fully embedded into network design for the joint optimization of $\M$ and $\X$.

## Modeling of MXNet
A degradation process of MRI is 
\begin{equation}\label{degradation}
\Y=\M \odot \F \X+\bm{\varepsilon},
\end{equation}
