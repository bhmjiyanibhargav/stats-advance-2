#!/usr/bin/env python
# coding: utf-8

# # question 01
What is the Probability density function?
In probability theory, a probability density function (PDF) is a mathematical function that describes the relative likelihood for a random variable to take on a given value.

The PDF is used to describe the probability distribution of a continuous random variable. Unlike a discrete probability distribution, where the probabilities are assigned to specific values, the PDF describes the probability of the random variable taking on any value within a certain range.

The PDF is defined such that the integral of the function over its entire domain is equal to 1. This means that the total area under the PDF curve is equal to 1, representing the total probability of all possible outcomes.

The PDF can be used to calculate the probability of a random variable taking on a value within a certain range by integrating the PDF over that range. This probability is given by the area under the PDF curve within that range.

The PDF is an important concept in statistics and probability theory, as it allows us to model and understand the probability distribution of continuous random variables. Many commonly used probability distributions, such as the normal distribution, are defined using a PDF.
# # question 02
Q2. What are the types of Probability distribution?
There are several types of probability distributions, including:

Discrete probability distributions: This type of distribution describes the probability of the occurrence of discrete outcomes, such as the number of heads in a series of coin tosses. Examples of discrete probability distributions include the binomial distribution, Poisson distribution, and geometric distribution.

Continuous probability distributions: This type of distribution describes the probability of the occurrence of continuous outcomes, such as the weight or height of a person. Examples of continuous probability distributions include the normal distribution, exponential distribution, and beta distribution.

Uniform probability distribution: This distribution is a continuous distribution where all values within a given range are equally likely to occur. The probability density function is a constant within the range and zero outside of it.

Bernoulli distribution: This is a special case of the binomial distribution where there is only one trial. The Bernoulli distribution describes the probability of a binary outcome, such as a coin flip landing heads or tails.

Multinomial distribution: This is an extension of the binomial distribution where there are more than two possible outcomes. The multinomial distribution describes the probability of each outcome occurring in a series of trials.

Exponential distribution: This distribution is commonly used to model the time between events in a Poisson process. It is a continuous distribution with a decreasing probability density function.

Poisson distribution: This is a discrete probability distribution that describes the probability of a certain number of events occurring in a fixed interval of time or space. It is commonly used to model rare events, such as accidents or failures.

These are some of the most commonly used types of probability distributions. There are many other probability distributions that are used to model different types of data and situations in statistics and probability theory.
# # question 03
Write a Python function to calculate the probability density function of a normal distribution with
given mean and standard deviation at a given point.
# In[1]:


import math

def normal_pdf(x, mu, sigma):
    """
    Calculates the PDF of a normal distribution with mean mu and standard deviation sigma
    at a given point x.
    """
    return math.exp(-(x - mu)**2 / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))
pdf = normal_pdf(3, 0, 1) # calculates the PDF of a standard normal distribution at x=3
print(pdf) # prints the calculated PDF value


# # question 04
Q4. What are the properties of Binomial distribution? Give two examples of events where binomial
distribution can be applied.
The binomial distribution is a probability distribution that describes the number of successes in a fixed number of independent trials, where each trial has the same probability of success. Some properties of the binomial distribution are:

The probability of success in each trial is denoted by p, and the probability of failure is denoted by q = 1 - p.

The binomial distribution is discrete and defined for a finite number of trials n, with possible values of the number of successes ranging from 0 to n.

The mean or expected value of the binomial distribution is mu = np, and the variance is sigma^2 = npq.

The shape of the binomial distribution is determined by the values of n and p. As n increases and/or p gets closer to 0.5, the distribution becomes more symmetric.

The binomial distribution can be approximated by the normal distribution when n is large and np and nq are both greater than or equal to 10.

Examples of events where binomial distribution can be applied include:

A manufacturer of light bulbs wants to know the probability of producing 8 or more defective bulbs in a sample of 100. Each bulb has a 5% probability of being defective, so this situation can be modeled using a binomial distribution with n = 100 and p = 0.05.

A basketball player wants to know the probability of making at least 3 out of 5 free throws. If the player's free throw percentage is 70%, this situation can be modeled using a binomial distribution with n = 5 and p = 0.7.
# # question 05
5. Generate a random sample of size 1000 from a binomial distribution with probability of success 0.4
and plot a histogram of the results using matplotlib.
# In[2]:


import numpy as np
import matplotlib.pyplot as plt

# Generate a random sample of size 1000 from a binomial distribution
n = 1000
p = 0.4
sample = np.random.binomial(n, p, size=1000)

# Plot a histogram of the sample using matplotlib
plt.hist(sample, bins=20)
plt.title("Histogram of Binomial Distribution")
plt.xlabel("Number of Successes")
plt.ylabel("Frequency")
plt.show()


# # question 06
Q6. Write a Python function to calculate the cumulative distribution function of a Poisson distribution
with given mean at a given point.
# In[4]:


from scipy.stats import poisson

def poisson_cdf(mean, point):
    """Calculates the CDF of a Poisson distribution with the given mean at a given point."""
    cdf = poisson.cdf(point, mean)
    return cdf
cdf = poisson_cdf(3.5, 2)
print(cdf)


# # question 07
How Binomial distribution different from Poisson distribution?
Binomial and Poisson distributions are both discrete probability distributions that are commonly used in statistics and probability theory. However, they are used in different situations and have different characteristics.

The main difference between Binomial and Poisson distributions is that the Binomial distribution is used to model the number of successes in a fixed number of trials, while the Poisson distribution is used to model the number of events that occur in a fixed interval of time or space.

The Binomial distribution has two parameters: the number of trials (n) and the probability of success (p). It gives the probability of obtaining a specific number of successes in n independent trials, where each trial has a probability of success of p. The Binomial distribution is used when the trials are independent and the probability of success is constant throughout the trials. The mean of the Binomial distribution is np, and its variance is np(1-p).

On the other hand, the Poisson distribution has one parameter: the mean (mu). It gives the probability of obtaining a specific number of events in a fixed interval of time or space, where the events occur randomly and independently of each other. The Poisson distribution is used when the number of events is rare and the mean rate of occurrence is constant over time or space. The mean and variance of the Poisson distribution are both equal to mu.

In summary, the Binomial distribution is used for a fixed number of independent trials with a constant probability of success, while the Poisson distribution is used for a rare event with a constant mean rate of occurrence over time or space.
# # question 08
Generate a random sample of size 1000 from a Poisson distribution with mean 5 and calculate the
sample mean and variance.
# In[5]:


import numpy as np

# Set the mean of the Poisson distribution
mean = 5

# Generate a random sample of size 1000 from the Poisson distribution
sample = np.random.poisson(mean, size=1000)

# Calculate the sample mean and variance
sample_mean = np.mean(sample)
sample_var = np.var(sample)

print("Sample Mean: {:.2f}".format(sample_mean))
print("Sample Variance: {:.2f}".format(sample_var))


# # question 09
How mean and variance are related in Binomial distribution and Poisson distribution?
In a Binomial distribution, the mean is equal to the product of the number of trials n and the probability of success p. That is, mean = n*p. The variance of the Binomial distribution is equal to n*p*(1-p).

On the other hand, in a Poisson distribution, the mean is equal to the parameter lambda (which is also equal to the variance). That is, mean = lambda.

So, the relationship between the mean and variance is different for Binomial and Poisson distributions. In the Binomial distribution, the variance depends on both the number of trials and the probability of success, whereas in the Poisson distribution, the variance depends only on the mean (which is also equal to the variance).

In summary, for Binomial distribution, as the probability of success p gets closer to 0 or 1, the variance gets smaller and the distribution becomes more concentrated around the mean. For Poisson distribution, as the mean gets larger, the variance also gets larger, and the distribution becomes more spread out around the mean.
# # question 10
Q10. In normal distribution with respect to mean position, where does the least frequent data appear?
In a normal distribution, the least frequent data appears at the tails of the distribution, which are the values that are farthest away from the mean.

The normal distribution is symmetric around the mean, so the least frequent data appears at the same distance from the mean on both sides of the distribution. Specifically, the least frequent data appears at the values that are more than 2 standard deviations away from the mean, which occur in the tails of the distribution.

For example, in a normal distribution with a mean of 0 and a standard deviation of 1, the least frequent data would appear at the values of -2 and +2. These values are far away from the mean, and so they occur less frequently in the distribution.
# In[ ]:




