This markdown file contains a set of notes derived from the [YouTube series on deep learning](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=LmaIb4KT4XYY7Tke) by [3Blue1Brown](https://youtube.com/@3blue1brown?si=wxfEm81NdWnHvIoJ).
The series offers a visual and intuitive explanation of the fundamentals of deep learning, from the basics of neural networks to the intricacies of backpropagation and gradient descent. These notes aim to be a useful reference for anyone looking to grasp the foundational ideas in machine learning.

> [!NOTE]
> Example used:
> Throughout this course, we will focus on a neural network designed for recognizing handwritten digits. Below is an overview of its structure:
> 1. Input Layer: Composed of 784 neurons, each representing a pixel in a 28x28 grayscale image.
> 		- The activation of each neuron corresponds to the pixel's intensity, ranging from 0 (black) to 1 (white).
> 	
> 		![[Image1.png]]
> 	
> 2. Hidden Layers: The network includes two hidden layers, each consisting of 16 neurons.
> 		- The number of neurons and layers is flexible and can be adjusted depending on the specific requirements of the network.
> 
> 3. Output Layer : Contains 10 neurons, each representing a digit from 0 to 9.
> 
> This example will serve as a foundation for explaining various aspects of neural network architecture and training.

# [What is a Neural Network](https://youtu.be/aircAruvnKk?si=bMUz1Nv04V_4JTEf)

## Neuron

A **neuron** in a neural network functions as a computational unit that outputs a value between 0 and 1, known as its **activation**. More accurately, each neuron can be seen as a function that processes inputs from neurons in the preceding layer and generates a single output value within that range.

In our example, the input layer comprises 784 neurons, each corresponding to a pixel in the input image. The output layer consists of 10 neurons, each one representing a digit for classification.

## Layered structure

In an ideal scenario, we might expect that each neuron in the penultimate layer corresponds to specific subcomponents of digits, such as loops (which might indicate the numbers 8 or 9) or vertical lines (which could suggest 1 or 7). Moving from the third layer to the final one would then involve learning how these subcomponents combine to form specific digits.

However, to identify these subcomponents, we must first recognize more basic patterns. For instance, a loop could be broken down into smaller edges that form its structure. Ideally, each neuron in the penultimate layer would correspond to these smaller edges.

When an image is processed, the neurons corresponding to the detected edges are activated, which in turn trigger neurons associated with the subcomponents. Ultimately, this activates the neuron associated with the recognized digit.

![[Image2.png]]

The goal is to develop a system that progressively builds complexity by combining simple elements: pixels into edges, edges into patterns, and patterns into recognizable digits.

## Edge detection example

Consider the scenario where we want a neuron in the second layer to detect an edge in a specific region of the image. To achieve this, we need to determine the network parameters that would allow it to recognize this edge pattern, along with other pixel patterns or more complex structures, such as loops formed by multiple edges.

### Weights

To accomplish this, we assign a **weight** to each connection between the neuron in the second layer and the neurons in the first layer. These weights are simply numbers. The neuron computes the **weighted sum** of the activations from the first layer, using these weights to prioritize certain inputs.

The calculation looks like this:

$$
w_1a_1 + w_2a_2 + w_3a_3 + w_4a_4 + \dots + w_na_n 
$$

If we set most of the weights for pixels to zero, except for positive weights in the region of interest, the weighted sum would primarily reflect the pixel values in that specific area.

To detect an edge, we might assign **negative weights** to the surrounding pixels. This approach ensures that the sum is maximized when the central pixels are bright, while the surrounding pixels are darker, thereby highlighting the presence of an edge.

![[Image3.png]]

### Activation Function

When calculating a weighted sum, the result could be any real number. However, for this network, we need the activations to fall within the range of 0 to 1. To achieve this, we apply a function that compresses the values of the real number line into the interval [0, 1].

A commonly used function for this purpose is the **sigmoid function**, also known as the logistic curve. This function ensures that highly negative inputs produce values close to 0, highly positive inputs produce values near 1, and values around 0 produce a smooth transition between the two extremes.

![[Image4.png]]

Thus, the activation of a neuron represents how positive or negative the relevant weighted sum is.

### Bias

In some cases, you may not want the neuron to activate when the weighted sum exceeds 0. For instance, you might want it to activate only when the sum is greater than 10, meaning you prefer a bias towards inactivation.

To implement this, we add a constant value, such as -10, to the weighted sum before applying the sigmoid function. This constant is referred to as the **bias** and helps adjust the threshold at which the neuron activates.

### Conclusion

The weights determine the pixel patterns that a neuron in the second layer responds to, while the bias controls the threshold for neuron activation. Essentially, the bias dictates how high the weighted sum must be for the neuron to activate meaningfully.

This concept applies to each neuron in the second layer. Every neuron is connected to all 784 input neurons from the first layer, with each connection assigned its own weight. Additionally, each neuron in the second layer has a bias added to the weighted sum before it is passed through the sigmoid function.

For a neuron in the second layer, the activation can be expressed as:

$$
a_{n}^{(1)} = \sigma \left( w_{0,0}a_{0}^{(0)} + w_{0,1}a_{1}^{(0)} + \dots + w_{0,783}a_{783}^{(0)} + b_n \right)
$$

Where:
- $w_{j,k}$  is the weight connecting neuron $k$ from layer $j$,
- $a_{x}^{(y)}$ is the activation of neuron $x$ from layer $y$,
- $b_n$ is the bias for neuron $n$,
- $\sigma$ is the sigmoid function.

When we talk about learning, we refer to the process of adjusting these weights and biases to solve the task at hand.

### Notation and Linear Algebra

A more compact way to represent the connections between layers is to use linear algebra:

- Organize the activations from one layer into a vector.
- Arrange the weights as a matrix, where each row represents the connections between neurons in the previous and next layers.
  
Taking the weighted sum of activations corresponds to one term in the matrix-vector product.

- Biases are organized into a vector and added to the matrix-vector product.
- The sigmoid function is applied element-wise to the resulting vector.

This can be written as:

```math
$$
\begin{bmatrix}
    a_{0}^{(1)} \\ a_{1}^{(1)} \\ \vdots \\ a_{n}^{(1)}
\end{bmatrix} =\sigma\left(\begin{bmatrix}
    w_{0,0} & w_{0,1}  & \dots  & w_{0,n} \\
    w_{1,0} & w_{1,1}  & \dots  & w_{1,n} \\
    \vdots & \vdots  & \ddots & \vdots \\
    w_{k,0} & w_{k,1}  & \dots  & w_{k,n}
\end{bmatrix}
\begin{bmatrix}
    a_{0}^{(0)} \\ a_{1}^{(0)} \\ \vdots \\ a_{n}^{(0)}
\end{bmatrix} + \begin{bmatrix}
    b_{0} \\ b_{1} \\ \vdots \\ b_{n} 
\end{bmatrix}\right)
$$
```

Once we express the weights, activations, and biases as their own symbols, the transition between layers can be captured concisely as:

$$
a^{(1)} = \sigma(Wa^{(0)} + b)
$$

# [Gradient descent, how neural networks learn](https://youtu.be/IHZwWFHWa-w?si=_gaZbo9wDWswBoKW)

Conceptually, each neuron is connected to every neuron in the previous layer. The **weights** in the weighted sum determine the strength of these connections, while the **bias** helps indicate whether the neuron tends to be active or inactive.

Initially, we randomly assign values to all the weights and biases. Naturally, the network will perform poorly at first, as it's essentially making random decisions.

## Cost Function

To improve the network's performance, we define a **cost function**, which provides a way for the computer to measure how far off its predictions are. For instance, in the case of classifying a digit as "3," the network should ideally have an activation of 1 for the neuron representing "3" and activations of 0 for all other neurons.

![[Image5.png]]

To calculate the cost for a single training example, we sum the squares of the differences between the actual output activations and the desired values. A lower value indicates a better performance.

![[Image6.png]]

However, to evaluate the overall performance, we compute the **average cost** over the entire training dataset. When we talk about the network learning, we're referring to minimizing this cost function. 

The goal is to adjust the weights and biases to reduce the cost, and the process of minimizing the cost function guides how the network learns to make better predictions. This function takes all the weights and biases as inputs and produces a single output: the cost.

### Example with One Input

To simplify, let's consider a function that has only one input and one output, rather than imagining a function with thousands of inputs. The objective is to minimize the value of this function. 

A practical approach is to begin with any input and determine in which direction you should move to lower the output. If you can calculate the slope of the function at your current point, you can decide whether to shift left or right: move left if the slope is positive and right if the slope is negative.

![[Image7.png]]

By repeating this process, we gradually move toward a **local minimum** of the function, much like a ball rolling down a hill. However, it's important to note that the local minimum we reach might not be the smallest possible value, or **global minimum**, of the cost function, depending on where we started.
## Gradient Descent

In this scenario, the input space can be imagined as the **xy-plane**, with the **cost function** represented as a surface above it. Instead of merely considering the slope of a function at a single point, we now need to determine which direction in the input space will reduce the cost function most rapidly. This is the **downhill direction**.

![[Image8.png]]

In multivariable calculus, the **gradient** of a function points in the direction of the steepest ascent, indicating where you would step to increase the function’s output most rapidly. By taking the negative of this gradient, we find the direction that will **decrease** the function most quickly. The length of the gradient vector represents how steep that descent is.

![[Image9.png]]

The key algorithm for minimizing the function involves computing the gradient direction, taking a small step downhill, and repeating this process iteratively.

### Neural Network Case

The same principle applies to a neural network, though it deals with many more inputs—up to 13,000 weights and biases in our example. These can be organized into a **giant column vector**. The **negative gradient of the cost function** is a vector indicating the most effective adjustments (increases or decreases) to those weights and biases to minimize the cost.

![[Image10.png]]

The repeated process of adjusting inputs by a small amount in the direction of the negative gradient is known as **gradient descent**.

One way to think about this gradient vector is that it encodes the relative importance of each weight and bias, highlighting which changes will provide the most efficient improvement in minimizing the cost function.

# Analyzing the Network

Earlier discussion, we suggested that each layer in a neural network learns to detect certain patterns. However, this is not quite what's happening in our case.

The **weights** that connect neurons from the first layer to a neuron in the second layer can be visualized as the specific pixel patterns that the second layer neuron is learning to recognize. 

When we analyze the actual weights from the first to the second layer, rather than identifying distinct edges or well-defined patterns, the weights often appear almost random, with only vague patterns emerging in the center.

![[Image11.png]]

This suggests that, within the vast 13,000-dimensional space of possible weights and biases, the network has found a **local minimum**. While this local minimum allows the network to classify most images successfully, the patterns it picks up are not as distinct or interpretable as we might expect. The network’s internal representations remain somewhat elusive, indicating that it doesn't necessarily align with our intuitive understanding of pattern recognition.

# [What is Backpropagation Really Doing?](https://youtu.be/Ilg3gGewQ5U?si=Hzca9jRyuZgowY2N)

## Recap

Backpropagation is an algorithm designed to compute the gradient, as discussed earlier. It helps determine how a single training example influences the adjustment of weights and biases, both in terms of direction (whether they should increase or decrease) and in relative magnitude (how much each change will reduce the cost function most effectively).

For instance, imagine the negative gradient has been calculated. If the component for the first weight is 3.2, and for the second weight it's 0.1, this means the cost function is **32 times more sensitive** to changes in the first weight than the second. In other words, small adjustments to the first weight will have a much larger impact on reducing the cost than the same adjustments to the second weight.

![[Image12.png]]

## Walkthrough Example

Let's focus on a specific training example, like an image of the digit **2**.

We want the network to classify this image correctly as a "2." To do this, we need to adjust the weights and biases, as we can't directly change the neuron activations. Our aim is to influence the **output layer** so that the activation for the neuron representing "2" increases, while the activations for the other output neurons decrease. 

The size of these adjustments should be proportional to how far each neuron’s current value is from its desired target. For example, the increase to the neuron representing "2" should be more significant than the decrease to the neuron representing "8" if the "8" neuron is already close to the correct activation value.

By doing this, backpropagation ensures that the adjustments made by this single training example effectively reduce the network’s overall error.

![[Image13.png]]

Let’s now focus on a specific neuron, the one whose activation we want to increase. There are three main factors that can contribute to increasing that activation:

1. **Increasing the bias**: This directly influences the activation threshold of the neuron.
2. **Adjusting the weights**: Strengthening the connections between this neuron and the previous layer.
3. **Changing activations from the previous layer**: Adjusting how much influence previous neurons have on this neuron.

### Adjusting the Weights

The effect of adjusting weights depends on how bright (active) the neurons in the previous layer are. Neurons with stronger activations in the previous layer have a greater influence on this neuron, as the weights connected to them are multiplied by higher activation values. Consequently, increasing the weight for a connection to a brighter neuron has a larger impact on the final cost function than increasing the weight of a connection to a dimmer neuron, for this particular training example.

### Adjusting the Activations

Even though we can't directly manipulate activations, we can indirectly affect them by adjusting the weights and biases. If the neurons in the second layer that are connected to the digit "2" neuron through **positive weights** get brighter, and those connected through **negative weights** get dimmer, then the activation of the "2" neuron will increase.

However, we also want to reduce the activation of all other output neurons in the last layer (i.e., the neurons corresponding to digits that aren't "2"). Each of these other output neurons has its own preferences for how the activations in the second-to-last layer should change. This means the adjustments needed for the digit "2" neuron are combined with the adjustments required by all other neurons, weighted by their influence on the output layer and their contribution to the overall error.

In summary, backpropagation efficiently coordinates these desires by adjusting the weights and biases so that the cost function is reduced across all neurons, focusing on the most significant connections for each individual training example.

![[Image14.png]]

By summing all the desired changes from each training example, we get a list of nudges that need to be applied to the weights and biases in the second-to-last layer. Then, this process is applied recursively, moving backward through the network, adjusting all weights and biases accordingly.

However, if we only used the adjustments from one training example, say an image of a "2," the network would become biased toward classifying all inputs as a "2." Therefore, we must apply this backpropagation process for every training example. Each training example provides its own set of desired adjustments, and we average these changes to determine the final adjustments to the weights and biases.

![[Image15.png]]

### Stochastic Gradient Descent

In practice, calculating the full gradient of the cost function over all training examples for every single step is computationally expensive and slow. To speed this up, a method called **stochastic gradient descent** is used.

Here's how it works:
1. **Shuffle the training data**: Randomly mix up the training examples.
2. **Mini-batches**: Divide the shuffled data into smaller mini-batches, each containing, for example, 100 training examples.
3. **Compute the gradient for each mini-batch**: Instead of computing the gradient for the entire training set, compute it for each mini-batch and adjust the weights and biases accordingly.

![[Image16.png]]

Although this mini-batch gradient is not as precise as calculating the gradient for the full dataset, it provides a good approximation while significantly speeding up the process. The network's path to minimizing the cost function becomes less precise (more like a "drunk man stumbling down a hill" rather than carefully calculating each step) but it is much faster.

![[Image17.png]]

By repeatedly going through all the mini-batches and making these approximate gradient steps, the network still converges toward a local minimum of the cost function, meaning it will still do a very good job on the training data.

This approach allows neural networks to learn efficiently even with massive datasets, while still performing gradient descent effectively.

# [Backpropagation calculus](https://youtu.be/tIeHLnjs5U8?si=rcdvlzbeN5FT-7pj)

## The Chain Rule

To begin, let's consider a simple neural network where each layer contains a single neuron:

![[Image18.png]]

Our objective is to understand how sensitive the cost function is to the variables within this network.

We will focus on the connection between the last two neurons. The superscripts $L$ and $L-1$ denote the layers in which these neurons are located.

![[Image19.png]]

Let's say that the desired value for the last activation, for a given training example, is $y$. This value $y$ could be either 0 or 1.
The cost function $C_0$ for a single training example can be defined as:

$$
C_0(...) = (a^{(L)} - y)^2
$$

where $a^{(L)}$ is the activation of the last layer, given by the sigmoid function:

$$
a^{(L)} = \sigma(z^{(L)})
$$

Furthermore, $z^{(L)}$ is calculated using the weights $w^{(L)}$ and the biases $b^{(L)}$ from the previous layer:

$$
z^{(L)} = w^{(L)} a^{(L-1)} + b^{(L)}
$$

Here, $w^{(L)}$ represents the weights connecting the neurons from layer $L-1$ to layer $L$, $a^{(L-1)}$ is the activation of the neurons in the previous layer, and $b^{(L)}$ is the bias term for the neurons in layer $L$.

![[Image20.png]]

### Sensitivity to the Weight

The first objective is to determine how sensitive the cost function $C$ is to small changes in the weight $w^{(L)}$. In other words, we want to calculate the derivative of $C$ with respect to $w^{(L)}$:

$$
\frac{\partial C_0}{\partial w^{(L)}}
$$

Here, $\partial w^{(L)}$ represents a small change in $w^{(L)}$, and $\partial C_0$ denotes the corresponding change in the cost. Our goal is to find their ratio.

Since $C$ and $w$ are interconnected, we can apply the chain rule:

$$
\frac{\partial C_0}{\partial w^{(L)}} = 
\frac{\partial z^{(L)}}{\partial w^{(L)}}
\frac{\partial a^{(L)}}{\partial z^{(L)}}
\frac{\partial C_0}{\partial a^{(L)}}
$$

Multiplying these three ratios yields the sensitivity of $C$ to small changes in $w^{(L)}$. Let’s compute the relevant derivatives.

1. **Derivative of Cost with Respect to Activation:**

$$
\frac{\partial C_0}{\partial a^{(L)}} = 2(a^{(L)} - y)
$$

   The magnitude of this derivative is proportional to the difference between the network’s output and the target value. Thus, if the output is significantly different from the target, even small changes can greatly impact the final cost function.

3. **Derivative of Activation with Respect to $z$:**

$$
\frac{\partial a^{(L)}}{\partial z^{(L)}} = \sigma'(z^{(L)})
$$

   Here, the derivative of $a^{(L)}$ with respect to $z^{(L)}$ is simply the derivative of the sigmoid function (or whichever nonlinearity is used).

5. **Derivative of Cost with Respect to Weight:**

$$
\frac{\partial z^{(L)}}{\partial w^{(L)}} = a^{(L-1)}
$$

   This means that the effect of a small nudge to the weight on the last layer is influenced by the activation of the previous neuron.

Combining all these components, we obtain:

```math
$$
\frac{\partial C_0}{\partial w^{(L)}} = 
\frac{\partial z^{(L)}}{\partial w^{(L)}}
\frac{\partial a^{(L)}}{\partial z^{(L)}}
\frac{\partial C_0}{\partial a^{(L)}} 
= a^{(L-1)} \times \sigma'(z^{(L)}) \times 2(a^{(L)} - y)
$$
```

This expression represents the derivative of the cost for a specific training example. Since the overall cost function is the average of costs across multiple training examples, its derivative requires averaging this expression over all training examples:

```math
$$
\underbrace{\frac{\partial C}{\partial w^{(L)}}}_{\begin{array}{c}
\text{Derivative of} \\ 
\text{full cost function}
\end{array}} 
= \overbrace{\frac{1}{n}\sum^{n-1}_{k=0}{\frac{\partial C_k}{\partial w^{(L)}}}}^{\begin{array}{c}
\text{Average of all} \\ 
\text{training examples}
\end{array}}
$$
```

Ultimately, this is just one component of the gradient vector, which is constructed from the partial derivatives of the cost function with respect to all the weights and biases:

```math
$$
\nabla C = \begin{bmatrix}
    \frac{\partial C}{\partial w^{(1)}} \\ 
    \frac{\partial C}{\partial b^{(1)}} \\ 
    \vdots \\ 
    \frac{\partial C}{\partial w^{(L)}} \\ 
    \frac{\partial C}{\partial b^{(L)}}
\end{bmatrix}
$$
```

### Sensitivity to the Bias

While $\frac{\partial C_0}{\partial w^{(L)}}$ represents just one of the many partial derivatives needed for our calculations, it accounts for more than 50% of the effort. The sensitivity of the cost function to the bias is almost identical; we simply replace $\frac{\partial z^{(L)}}{\partial w^{(L)}}$ with $\frac{\partial z^{(L)}}{\partial b^{(L)}}$. In this case, the derivative simplifies to 1.

```math
$$
\frac{\partial C_0}{\partial b^{(L)}} = 
\frac{\partial z^{(L)}}{\partial b^{(L)}}
\frac{\partial a^{(L)}}{\partial z^{(L)}}
\frac{\partial C_0}{\partial a^{(L)}} 
= 1 \times \sigma'(z^{(L)}) \times 2(a^{(L)} - y)
$$
```

### Sensitivity to the Activation of the Previous Layer

Next, we can evaluate how sensitive the cost function is to the activation of the previous layer. This is where the concept of backpropagation becomes relevant.

```math
$$
\frac{\partial C_0}{\partial a^{(L-1)}} =
\frac{\partial z^{(L)}}{\partial a^{(L-1)}}
\frac{\partial a^{(L)}}{\partial z^{(L)}}
\frac{\partial C_0}{\partial a^{(L)}} 
= w^{(L)} \times \sigma'(z^{(L)}) \times 2(a^{(L)} - y)
$$
```

Here, the derivative $\frac{\partial z^{(L)}}{\partial a^{(L-1)}}$, which reflects the sensitivity of $z$ to the activation of the previous layer, is equal to the weight $w^{(L)}$.

Although we cannot directly influence the activation of the previous layer, keeping track of this sensitivity is beneficial. We can continue to apply the chain rule recursively to determine how the cost function is influenced by the preceding weights and biases.

![[Image21.png]]

## Layers with Multiple Neurons

When incorporating multiple neurons into the layers of a neural network, the overall structure remains similar, though additional indices are needed to keep track of the various elements:

![[Image22.png]]

The chain rule expression that describes how sensitive the cost is to a specific weight retains its basic form:

```math
$$
\frac{\partial C_0}{\partial w^{(L)}_{jk}} = 
\frac{\partial z^{(L)}_j}{\partial w^{(L)}_{jk}}
\frac{\partial a^{(L)}_j}{\partial z^{(L)}_j}
\frac{\partial C_0}{\partial a^{(L)}_j}
$$
```

However, the derivative of the cost with respect to one of the activations in layer $L-1$ changes in this context. In this case, each neuron influences the cost function through multiple pathways. Specifically, a neuron affects all neurons in the subsequent layer, and we need to aggregate these contributions. This results in the following expression:

```math
$$
\frac{\partial C_0}{\partial a^{(L-1)}_{k}} = \underbrace{\sum^{n_{L}}_{j=0}
\frac{\partial z^{(L)}_j}{\partial a^{(L-1)}_{k}}
\frac{\partial a^{(L)}_j}{\partial z^{(L)}_j}
\frac{\partial C_0}{\partial a^{(L)}_j}}_{\text{Sum over layer } L}
$$
```

This summation accounts for the influence of the activation $a^{(L-1)}_{k}$ on the cost through all neurons $j$ in layer $L$.

# Wrapping up

This deep dive into the mathematics behind neural networks has allowed us to understand the fundamental concepts. We've explored the intricacies of backpropagation, delving into the chain rule and its application in calculating gradients. We've seen how these calculations allow us to adjust weights and biases to minimize the cost function, ultimately improving our network's performance.

But again, all the content in these notes is derived from the excellent 3Blue1Brown YouTube series on deep learning, which provides an intuitive and visual approach to understanding these complex concepts.
