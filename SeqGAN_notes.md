## SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient

1. Link: https://arxiv.org/pdf/1609.05473.pdf
2. Experiment code: https://github.com/LantaoYu/SeqGAN

-----

### Problem


Generating sequential synthetic data that mimics the real one is an important problem in unsupervised learning. But there is a problem: the model generates a sequence iteratively and predicts next token conditioned on its previously predicted ones that may be never observed in the training data. Such a discrepancy between training and inference can incur accumulatively along with the sequence and will become prominent as the length of sequence increases. 

General adversarial net (GAN) proposed by (Goodfellow and others 2014) is a promising framework for solving the above problem. Specifically, in GAN a discriminative net D learns to distinguish whether a given data instance is real or not, and a generative net G learns to confuse D by generating high quality data. This approach has been successful and been mostly applied in computer vision tasks of generating samples of natural images. 

Unfortunately, applying GAN to generating sequences has two problems. Firstly, GAN is designed for generating real-valued, continuous data but has difficulties in directly generating sequences of discrete tokens, such as texts. The reason is that in GANs, the generator starts with random sampling first and then a determistic transform, govermented by the model parameters. As such, the gradient of the loss from D w.r.t. the outputs by G is used to guide the generative model G (paramters) to slightly change the generated value to make it more realistic. If the generated data is based on discrete tokens, the “slight change” guidance from the discriminative net makes little sense because there is probably no corresponding token for such slight change in the limited dictionary space. Secondly, GAN can only give the score/loss for an entire sequence when it has been generated; for a partially generated sequence, it is non-trivial to balance how good as it is now and the future score as the entire sequence.


### Solution

In paper, to address the above two issues, authors consider the sequence generation procedure as a sequential decision making process. The generative model is treated as an agent of reinforcement learning (RL); the state is the generated tokens so far and the action is the next token to be generated.
Authors employ a discriminator to evaluate the sequence and feedback the evaluation to guide the learning of the generative model.

### Details

#### Sequence Generative Adversarial Nets

Given a dataset of real-world structured sequences, train a θ-parameterized generative model Gθ to produce a sequence Y1:T = (y1,...,yt,...,yT),yt ∈ Y, where Y is the vocabulary of candidate tokens. Authors interpret this problem based on reinforcement learning. In timestep t, the state s is the current produced tokens (y1 , . . . , yt−1 ) and the action a is the next token yt to select. 

Additionally, authors also train a φ-parameterized discriminative model Dφ to provide a guidance for improving generator Gθ. Dφ(Y1:T) is a probability indicating how likely a sequence Y1:T is from real sequence data or not.

<img src="https://github.com/AlexPopenych/notes/blob/master/Снимок%20экрана%202019-03-31%20в%2022.28.06.png">

Discriminative model Dφ is trained by providing positive examples from the real sequence data and negative examples from the synthetic sequences generated from the generative model Gθ. At the same time, the generative model Gθ is updated by employing a policy gradient and MC search on the basis of the expected end reward received from the discriminative model Dφ.

#### SeqGAN via Policy Gradient

The objective of the generator model (policy) Gθ (yt |Y1:t−1 ) is to generate a sequence from the start state s0 to maximize its expected end reward:

<img src="https://github.com/AlexPopenych/notes/blob/master/Снимок%20экрана%202019-03-31%20в%2022.11.53.png">

where RT is the reward for a complete sequence.

<img src="https://github.com/AlexPopenych/notes/blob/master/Снимок%20экрана%202019-03-31%20в%2022.12.39.png" width="80" height="30"> - is the action-value function of a sequence, i.e. the expected accumulative reward starting from state s, taking action a, and then following policy Gθ.

Then consider the estimated probability of being real by the discriminator Dφ (Y1:T) as the reward.

<img src="https://github.com/AlexPopenych/notes/blob/master/Снимок%20экрана%202019-03-31%20в%2022.13.39.png">


However, the discriminator only provides a reward value for a finished sequence. To evaluate the action-value for an intermediate state, authors apply Monte Carlo search with a roll-out policy Gβ to sample the unknown last T − t tokens.

<img src="https://github.com/AlexPopenych/notes/blob/master/Снимок%20экрана%202019-03-31%20в%2022.14.55.png">

where Y1:t = (y1, . . . , yt) and Yt+1:T is sampled based on the roll-out policy Gβ and the current state. In experiment, Gβ is set the same as the generator, but one can use a simplified version if the speed is the priority. To reduce the variance and get more accurate assessment of the action value, authors run the roll-out policy starting from current state till the end of the sequence for N times to get a batch of output samples. 

<img src="https://github.com/AlexPopenych/notes/blob/master/Снимок%20экрана%202019-03-31%20в%2022.15.40.png">

Once have a set of more realistic generated sequences, retrain the discriminator model and update discriminator parameters:

<img src="https://github.com/AlexPopenych/notes/blob/master/Снимок%20экрана%202019-03-31%20в%2022.43.57.png">

In summary, Algorithm 1 shows full details of the pro- posed SeqGAN. At the beginning of the training, authors use the maximum likelihood estimation (MLE) to pre-train Gθ on training set S. The supervised signal from the pre- trained discriminator is informative to help adjust the gener- ator efficiently.

### Experiments
#### Synthetic Data Experiments

<img src="https://github.com/AlexPopenych/notes/blob/master/Снимок%20экрана%202019-03-31%20в%2022.50.12.png">

#### Real-world Scenarios

* Chinese poem generation
* Obama political speech generation
* Music Generation

<img src="https://github.com/AlexPopenych/notes/blob/master/Снимок%20экрана%202019-03-31%20в%2022.50.23.png">
