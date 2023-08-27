# Wasserstein Conditional Deep-Convolutional GAN with Gradient Penalty

This repository contains the scripts and document for my International
Baccalaureate Extended Essay in Computer Science (which I took at Higher level).
This essay was submitted during the May 2023 examination session and I received
a grade A for it.

In summary, I combined the concepts of a Conditional GAN, a Wasserstein GAN with
gradient penalty, and the Deep Convolutional GAN together to generate images from
the EMNIST dataset that consists of alphanumeric characters. I then investigated
the effect of constricting the training dataset to see how that would impact
the fidelity of the generated images which is quatified by the Fréchet Inception
Distance (FID score).

## Model Architecture (cDCWGAN-gp)
This model was implemented using the Pytorch framework and can be found in
`src/models.py`. The model architecture is illustrated in the diagram below.  

<img src="imgs/architecture.png" title="" alt="GAN architecture" data-align="center">


## Training and Evaluation Procedure
Part I: Training the GAN
1. Train the GAN on 112,800 samples for 15 epochs
2. Once training has completed, save the generator state
3. Repeat step 1-3 for all 10 dataset sizes (replace 112,800 with the appropriate value)

Part II: Gathering FID Scores
1. Load the checkpoint of the GAN trained on 112,800 samples
2. Generate samples using the saved generator
3. Calculate the FID score given the generated samples and real images from the dataset
4. Repeat for all saved generator states

Training this model required powerful hardware, so a cloud virtual machine was used with 8
Intel Xeon E5-2623 v4 cores, 30 GB of RAM, and an Nvidia Quadro P4000 with 8GB of vRAM.
This was able to sufficiently train each model configuration, taking approximately 16 hours to
do so including FID calculation.


The hyperparameters used during the training of this model are shown below. The
python Pytorch script can be found at `src/train.py`.

- $\lambda = 10$ gradient penalty coefficient 
- $n_{\text{critic}}$ ratio of critic to generator training iterations
- $\alpha = 1e-4$ learning rate of optimiser
- $\beta_1 = 0.0$ $\beta_2 = 0.9$ Adam optimiser parameters

<img src="imgs/training.png" title="" alt="GAN training" data-align="center">

The following loss function was constructed by combining the techniques from each of the papers.
$C_{\text{loss}} = \underbrace{C(G(z,y),y) - C(x,y)}_{\text{Wasserstein distance estimate}} +
\underbrace{\lambda(\| \nabla C(\tilde x, y) \|_2 -1)^2}_{\text{Gradient penalty term}}$
The ‘Wasserstein distance estimate’ remains the same from the original WGAN paper \cite{wgan},
however the generator $G(z,y)$ and critic $C(x,y)$, also take in the labels $y$, since the GAN
is conditional. The next element term is the gradient penalty, which uses the norm of the
gradients of the critic’s weights, when inputting an interpolation of a real and generated image.
The interpolation is done using a random number, $\epsilon\in [0,1]$, resulting in an interpolated
image $\tilde x = \epsilon x + (1-\epsilon)\hat x$, where $x$ and $\hat x$ are real and generated images.

## Results
### Quantitative Data
| % of Total | Samples |  FID Score  | $\Delta$ FID |
|:----------:|:-------:|:-----------:|:------------:|
|         10 |   11280 |  63.9696288 |              |
|         20 |   22560 | 43.81612586 |      20.1535 |
|         30 |   33840 | 39.08653845 |     4.729587 |
|         40 |   45120 | 37.73255476 |     1.353984 |
|         50 |   56400 | 35.50103584 |     2.231519 |
|         60 |   67680 | 32.09195388 |     3.409082 |
|         70 |   78960 | 28.34171738 |     3.750237 |
|         80 |   90240 | 31.41788896 |     3.076172 |
|         90 |  101520 | 30.47313068 |     2.891379 |
|        100 |  112800 | 27.58175179 |     2.891379 |


<img src="imgs/plot.png" title="" alt="FID plot" data-align="center">

As is visible in the graph 1, the FID score clearly decreases as the number of training samples
increases (indicating increased fidelity). It is also important to notice that the drop in FID
is significantly higher at lower numbers of training samples, than at higher numbers (between
11,280 and 22,560 FID drops by 20.1 FID points).

### Qualitative Data
### Full 100% Training Set
<img src="imgs/112800.png" title="" alt="FID plot" data-align="center">

### Reduced 10% Training Set
<img src="imgs/11280.png" title="" alt="FID plot" data-align="center">

## References
The references can be found in the `references.bib` file.
