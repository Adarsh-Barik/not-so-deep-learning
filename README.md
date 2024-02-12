<div>
  <h1 align="center">Not-So-Deep Learning</h1>
    <h3 align="center">End-to-end implementation of a Convolutional Neural Network</h3>
</div>

<br/>

<p align="center">
    <img src="webimg/cover.png" alt="Cover">
</p>

<h2> </h2>

This is a minicourse to learn basics of application oriented deep-learning. In this minicourse we will learn an end-to-end implementation of a convolutional neural network (CNN) in pytorch. The course is divided in 3 lectures, each taking roughly 1-1.5 hours to complete.

## Task
We aim to learn the flow profile of a radial flow from a camera captured image. All images are generated synthetically. The input image looks like below:
<img src="images/input/input_0.jpg" alt="input">
Our task is to learn the corresponding flow profile which looks like below:
<img src="images/output/output_0.jpg" alt="input">

The CNN not only learns the flow profile but it also implicitly figures out the location of the source.

## Preliminaries

Before we begin, it might help to complete the following tasks:
1. Clone the git repository.
2. Learn how to get minima of a one variable function. For example: f(x) = x^2, f(x) = x^3
3. Install python and the following packages: torch, torchvision, numpy, matplotlib
4. Run 'hello_world.py'
5. Most importantly, have a problem that you want to solve using deep learning. It need not be an original problem. It can be a toy version of a problem from a research paper. You should focus more on a formalizing the problem. You would need answers to the following questions:
    1. What is your input? - images, texts, videos?
    2. What is your output? - images, texts, videos, number?
    3. How many (input, output) pairs do you have?
    4. What are the dimensions of your input and output?
    5. Do you have a basic understanding of input-output relationship? It could be a conjecture which might or might not be correct.

## Lecture 1
1. Example: Decide when to give loan as a bank
2. Example: Decide how much loan to give as a bank
3. Supervised and unsupervised learning, focus on Supervised learning
4. Loss functions
5. Minimization of loss functions: convex, nonconvex
6. Important aspects: initialization, learning rate
7. Perspective: Machine learning is a task to design good loss functions
8. What are artificial neural networks?
9. Overfitting, underfitting and benefits of cross validation
10. Not all data are the same? - preprocessing is important
11. Skewed datasets, badly scaled datasets
12. Make sure 'hello_world.py' runs

### Homework
Work on the problem (at least a toy version) of problem of your choice. If you do not have one in mind, then take the input images from the input folder and construct a CNN that figures out the coordianates of the source. You can use 'data/seed_pts.dat' as your output data for training/validation.

## Lecture 2
1. Introduction to CNN
2. Operators that help: kernel, stride, padding, maxpooling etc
3. Introduction to pytorch
4. Develop the CNN architecture for the problem: Autoencoders
5. Go through the code, line-by-line
6. Run the code for various learning rates
7. Plot beutiful images

## Lecture 3
1. Any queries?
2. Discussion on the homework problem
3. Issues and debugging
4. Why did it work or why did it not

## Useful references <a name="credits" />
1. CNN in pytorch: https://medium.com/thecyphy/train-cnn-model-with-pytorch-21dafb918f48
2. Autoencoders in pytorch: https://stackoverflow.com/questions/65260432/how-to-create-a-neural-network-that-takes-in-an-image-and-ouputs-another-image
3. Autoencoders for colorizing images: https://github.com/Keramatfar/image-colorizing-with-auto-encoder/blob/main/01.ipynb

