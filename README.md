# PoPS: Policy Pruning and Shrinking for Deep Reinforcement Learning
This repository contains the PoPS framework code, as presented in "PoPS: Policy Pruning and Shrinking for Deep Reinforcement Learning". 

# Special Notes 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
Any code/data piece supplementary to this repository must be used in accordance to its own license and terms
datasets has the instructions for obtaining the datasets

# Requirements
We ran the code on Ubuntu 18.04 LTS , it should also work on Windows (although downloading the emulator for atari is tricky on Windows)
We used Python 3.6 with the following packages: tensorflow, numpy, gym, Box2D

# Introduction
The recent success of deep neural networks (DNNs) for function approximation in reinforcement learning has triggered the development of Deep Reinforcement Learning (DRL) algorithms in various fields, such as robotics, computer games, natural language processing, computer vision, sensing systems, and wireless networking. Unfortunately, DNNs suffer from high computational cost and memory consumption, which limits the use of DRL algorithms in systems with limited hardware resources. 

In recent years, pruning algorithms have demonstrated considerable success in reducing the redundancy of DNNs in classification tasks. However, existing algorithms suffer from a significant performance reduction in the DRL domain. In this paper, we develop the first effective solution to the performance reduction problem of pruning in the DRL domain, and establish a working algorithm, named Policy Pruning and Shrinking (PoPS), to train DRL models with strong performance while achieving a compact representation of the DNN. The framework is based on a novel iterative policy pruning and shrinking method that leverages the power of transfer learning when training the DRL model. We present an extensive experimental study that demonstrates the strong performance of PoPS using the popular Cartpole, Lunar Lander, and Pong environments. Finally, we develop an open source software for the benefit of researchers and developers in related fields.

# Modules
1) The \textit{configs.py} file holds the configuration for each environment and model, parameters (such as the target sparsity and the pruning frequency for the pruning procedure) that affect the initial training phase, and the PoPS procedure.


