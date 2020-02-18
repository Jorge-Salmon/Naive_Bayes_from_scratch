# **Naive Bayes Classifier coded from scratch (without using Scikit-learn)**
A fundamental part of learning ML is understanding the theory behind every used statistical model. Otherwise, Scikit-learn is nothing but a magical black box to the user. The Naive Bayes algorithm is quite simple and fun to start with. It will be explained and used in the Titanic Survival prediction problem, using Python. 

**Note:** My goal is to show the basics of the algorithm and how I implemented it, so I did not put much effort in feature engineering to obtain the highest score possible.

## Theory: Bayes Theorem
It describes the probability of an event, based on prior knowledge of conditions that might be related to the event. Using Bayes theorem, we can find the probability of y happening, given that x has occurred. Here, x  is the evidence, y is the prior knowledge and  P(x|y) is the likelihood. The assumption made here is that the features are independent.

It's equation is as follows:

<img src="https://render.githubusercontent.com/render/math?math=P(y|x)= \dfrac{P(x|y)P(y)}{P(x)}">

where 

* <img src="https://render.githubusercontent.com/render/math?math=y,x"> = Events
* <img src="https://render.githubusercontent.com/render/math?math=P(y|x)"> = Probability of y given x
* <img src="https://render.githubusercontent.com/render/math?math=P(x|y)"> = Probability of x given y
* <img src="https://render.githubusercontent.com/render/math?math=P(x), P(y)"> = Independent probabilities of y and x

Naive Bayes Methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of features given the value of the class variable.

In our case, the variable y is the class variable (Survival 0 or 1), which represents if a passenger will survive or not given the conditions. The variable X represent the features. X is given as <img src="https://render.githubusercontent.com/render/math?math=X=(x_{1}, x_{2}, ..., x_{n})"> and its components can be mapped to Age, Class, Sex, etc. By substituting for X into the Bayes Rule and expanding using the chain rule we get
 
<img src="https://render.githubusercontent.com/render/math?math=P(y|x_{1}, x_{2}, ..., x_{n})= \dfrac{P(x_{1}|y)P(x_{2}|y)...P(x_{n}|y)P(y)}{P(x_{1})P(x_{2})...P(x_{n})}">

Now, you can calculate the values for each probability by looking at the dataset and substitute them into the equation. In our case, the class variable y  has two outcomes: 0 or 1. So the survival probabilities for each case '*Survival*' or '*No survival*' (1 or 0, respectively) needs to be calculated for each passenger. The one having the highest probability is then the final outcome. That is, if  P(1) > P(0), the person will survive.

Click here to see the code: https://github.com/Jorge-Salmon/Naive_Bayes_from_scratch/blob/master/Naive%20Bayes%20Classifier.ipynb 

