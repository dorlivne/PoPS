# PoPS: Policy Pruning and Shrinking for Deep Reinforcement Learning
This repository contains the PoPS framework code, as presented in "PoPS: Policy Pruning and Shrinking for Deep Reinforcement Learning". The paper has been accepted for publication in the IEEE Journal of Selelcted Topics in Signal Processing. A preliminary version is available at arXiv (https://arxiv.org/abs/2001.05012).  
if this code is used in your research please cite our paper:  [BibTeX](https://github.com/dorlivne/PoPS#please-cite-our-paper)

## Special Notes 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
Any code/data piece supplementary to this repository must be used in accordance to its own license and terms
datasets has the instructions for obtaining the datasets

## Requirements
We ran the code on Ubuntu 18.04 LTS , it should also work on Windows (although downloading the emulator for atari is tricky on Windows)
We used Python 3.6 with the following packages: tensorflow, numpy, gym, Box2D

## Introduction
The recent success of deep neural networks (DNNs) for function approximation in reinforcement learning has triggered the development of Deep Reinforcement Learning (DRL) algorithms in various fields, such as robotics, computer games, natural language processing, computer vision, sensing systems, and wireless networking. Unfortunately, DNNs suffer from high computational cost and memory consumption, which limits the use of DRL algorithms in systems with limited hardware resources. 

In recent years, pruning algorithms have demonstrated considerable success in reducing the redundancy of DNNs in classification tasks. However, existing algorithms suffer from a significant performance reduction in the DRL domain. In this paper, we develop the first effective solution to the performance reduction problem of pruning in the DRL domain, and establish a working algorithm, named Policy Pruning and Shrinking (PoPS), to train DRL models with strong performance while achieving a compact representation of the DNN. The framework is based on a novel iterative policy pruning and shrinking method that leverages the power of transfer learning when training the DRL model. We present an extensive experimental study that demonstrates the strong performance of PoPS using the popular Cartpole, Lunar Lander, and Pong environments. Finally, we develop an open source software for the benefit of researchers and developers in related fields.

## Modules
1) *configs.py* - file holds the configuration for each environment and model, parameters (such as the target sparsity and the pruning frequency for the pruning procedure) that affect the initial training phase, and the PoPS procedure.
it also holds the dynamic learning rate function for each architecture, we noticed that the learning rate should be higher when the architecture is smaller.
there is no need to use configs.py if you are planning on implementing "PoPS" on a new environment.

2) *model.py* - file contains the model architecture for each environment such as class DQNPong, class CartPoleDQN, class ActorLunarLander, and CriticLunarLander. These models are used for the initial training phase, and follow the DQNAgent interface. Each model is associated with a Student version that inherits it, such as StudentPong. The Student version is adjusted for the PoPS algorithm such that the loss function is the KL-Divergence loss function (described in the paper)and the architecture is a dynamic architecture which is defined by the redundancy measures and the *_calculate_sizes_according_to_redundancy* function each student has which basically defines each layer size by the redundancy measures.

3) *train.py* -  contains functions that execute the policy distillation training procedure as well as the IPP's pruning and fine-tuning steps. The functions are well documented and are used by a variety of models and environments. in short, *train_student* function is responsible on training and optionally prune the model and *fit_supervised* is responsible on preforming the entire policy distillation training procedure.

4) *prune.py* - contains the IPP module orchestrating the pruning phase in the PoPS algorithm as detailed in the paper.
              it contains the function *iterative_pruning_policy_distilliation* which takes a trained model, and prune it with IPP
              it outputs information regarding the preformance of the model during the pruning process and a sparse model which is saved               in the given path(as stated in the script).
              
5) *utils.py* - a collection of helpful utils that are used for plotting graphs or histograms, using the tensorflow interface with the pruning framework in a more convenient manner and etc.

  
  
 ## Please Cite Our Paper
    @ARTICLE{livne2020PoPS,
        author  = {Livne, Dor and Cohen, Kobi},
        journal = {to appear in the IEEE Journal of Selected Topics in Signal Processing},
        title   = {PoPS: Policy Pruning and Shrinking for Deep Reinforcement Learning},
        year 	= {2020},
        volume 	= {__},
        number 	= {__},
        pages 	= {__-__},
        doi 	= {__},
        ISSN 	= {__},
        month 	= {__},
    }



