---
created: 2024-04-08T13:02
updated: 2024-09-24T22:48
---
Some definition: 
- Nonstationary: In the context of Reinforcement Learning (RL), "stationary" refers to an environment or a process where the probabilities of rewards and state transitions do not change over time. In other words, the rules of the environment and the outcomes of actions are consistent throughout the learning process. In such cases it makes more sense to give more weight to recent rewards. To do that, we add weight to the new reward.  

Background: 
- Basic of statisitcs:
	- **Variance** measures the average degree to which each point differs from the mean, in other words, it's a numerical value that describes the variability of the observations.
	- **Standard Deviation** is a measure of the amount of variation of a random variable expected about its mean. A low standard deviation indicates that the values tend to be close to the mean, while a high standard deviation indicates that the values are spread out over a wider range. 
	- **The law of large number** The Law of Large Numbers (LLN) is a fundamental theorem in probability and statistics that describes the result of performing the same experiment a large number of times. It states that as the number of trials increases, the average of the results obtained from the experiments will converge to the expected value.

# Course 1: 
# 1- The K-Armed Bandit Problem 
## Sequential Decision Making with Evaluation Feedback:
Imagine the next problem where the doctor has to give medicine to a patient and based on which one he choose, he will receive a reward (cure the patient). 

![[Capture.png]]

For the doctor to decide which action is best, we must define the value of taking each action. We call these values the **action values** or the **action value function**. 

The value is the expected reward :
$$ 
q_*(a) \doteq \mathbb{E}[R_t | A_t = a] \quad \forall a \in \{1, \ldots, k\} \\
= \sum_r p(r|a) r
$$
The function define the expected reward when taking the action a.

The goal is to maximize the expected reward: $$argmax_a q_*(a)$$

The next image shows how the action values are calculated as an average. q∗ is the mean of the distributions for each action.

![[Capture 2.png]]

## Learning Action Values 
The value of an action is the expected reward when the action is taken, defined by the following equation: 
$$
q_*(a) \doteq \mathbb{E}[R_t | A_t = a]
$$


$q_*(a)$ is not known, so we estimate it. The estimated value for action A is the sum of rewards observed when taking action A divided by the total number of times action A has been taken.

## Sample-Average Method
The estimate of  $q_∗(a)$ is denoted as $Q_t(a$) and can be written as: 

$$
Q_t(a) \doteq \frac{\text{sum of rewards when } a \text{ taken prior to } t}{\text{number of times } a \text{ taken prior to } t} = \frac{\sum_{i=1}^{t-1} R_i}{t-1}
$$

An example of Sample-Average methods is a Clinical Trial. A reward of 1 is obtained if the treatment succeeds otherwise 0.

![[Capture 5.png]]

Different attempts are made during the trial as shown in the image below. As the experiment advances, the estimates of the Action Values became more accurate Fig. 4.

In reality, our doctor would not randomly assign treatments to their patients. Instead, they would probably assign the treatment that they currently think is the best. We call this method of choosing actions **greedy**. The greedy action is the action that currently has the largest estimated value. Selecting the greedy action means the agent is **exploiting** its current knowledge. It is trying to get the most reward it can right now. We can compute the greedy action by taking the $argmax_a$ of our estimated values.

![[Capture 6.png]]

An example of how is the action selection process and what corresponds to a greedy action is shown below Fig. 5.


![[Capture 8.png]]

The agent can not choose to both explore and exploit at the same time. This is one of the fundamental problems in reinforced learning. **The exploration-exploitation dilemma.** 

## Estimating Action Value Incrementally
How we can turn the sample averages of observed  rewards in a computationally efficient manner. The obvious implementation would be to maintain a record of all the rewards and then perform this computation whenever the estimated value was needed. However, if this is done, then the memory and computational requirements would grow over time as more rewards are seen. Each additional reward would require additional memory to store it and additional computation to compute the sum in the numerator. The incremental update rule is derived as follows: 
												
															
															$$Q_n = \frac{1}{n-1}\sum_{i=1}^{n-1} R_i$$
		 
				$$Q_{n+1} = \frac{1}{n}\sum_{i=1}^{n} R_i = \frac{1}{n} \left( R_n + \sum_{i=1}^{n-1} R_i \right) = \frac{1}{n} \left( R_n + (n - 1) \cdot \frac{1}{n-1} \sum_{i=1}^{n-1} R_i \right) = Q_n + \frac{1}{n} [R_n - Q_n]$$
		  
		

The error in the estimate is the difference between the old estimate Qn and the new target Rn. Taking a step towards that new target will create a new estimate that reduces our error. Here, the new reward is our target. The size of the step is determined by our step size parameter and the error of our old estimate.

## Incremental update rule
												$$NewEstimate ←− OldEstimate + Stepsize [Target − OldEstimate]$$
							
																	$$Q_n+1 = Qn + α_n [R_n − Q_n]$$


The step size α can be a function of n that produces a number from zero to one. $α_n → [0, 1]$ In the specific case of the sample average, the step size is equal to one over n $α_n = 1/n$ This is an example of a non-stationary bandit problem, A simple pseudo-code for this algorithm can be seen in Fig. 6 where $1/N(a)$ can be replaces by $α$ and the function bandits(a) is assumed to take an action and return a corresponding reward. These problems are like the bandit problems we’ve discussed before, except the distribution of rewards changes with time. The doctor is unaware of this change but would like to adapt to it.

![[Capture 10.png]]

One option is to use a fixed step size. If αn is constant like 0.1, then the most recent rewards affect the estimate more than older rewards Fig. 7.

![[Capture 11.png]]

## What is the trade-off
Exploration improves the knowledge for the long-term benefit. Exploitation on the other hand, exploits the agent’s current estimated values. It chooses the greedy action to try to get the most reward. But by being greedy with respect to estimated values, may not actually get the most reward. 

## Epsilon-Greedy Action Selection
The e-greedy selecion process is given as follows 
								$$A_t \leftarrow \begin{cases} argmax_a Q_t(a), & \text{with probability } 1 - \epsilon \\ a \sim \text{Uniform}(\{a_1 \ldots a_k\}), & \text{with probability } \epsilon \end{cases}$$
								
## Optimistic Initial Values
Previously the initial estimated values were assumed to be 0, which is not necessarily optimistic. Now, our doctor optimistically assumes that each treatment is highly effective before running the trial. To make sure we’re definitely overestimating, let’s make the initial value for each action 2. Let’s assume the doctor always chooses the greedy action. 

The optimistic values incentive exploration during the initial steps of the experiment, making $Q$ closer to $q∗$.
Some of the limitations of optimistic initial values are : • Optimistic initial values only drive early exploration • They are not well-suited for non-stationary problems • We may not know what the optimistic initial value should be. 

## Upper Confidence Bound Action Selection
If we had a notion of uncertainty in our value estimates, we could potentially select actions in a more intelligent way. What does it mean to have uncertainty in the estimates? $Q(a)$ here represents our current estimate for action $a$. These brackets represent a confidence interval around q∗(a). They say we are confident that the value of action a lies somewhere in this region Fig. 12. For instance, we believe it may be here or there.  As shown by the Fig. 12, the left bracket is called the lower bound, and the right is the upper bound. The region in between is the confidence interval which represents our uncertainty. If this region is very small, we are very certain that the value of action a is near our estimated value. If the region is large, we are uncertain that the value of action A is near or estimated value.

![[Capture 12.png]]

The next equation defines the action selection process based on the  (UBC) : 
									$$A_t = \underset{a}{\text{argmax}} \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]$$
- $A_t$​ represents the action selected by the agent at time step $t$. The action is chosen to maximize the expression within the brackets.
- $Qt​(a)$ is the estimated value (or quality) of action  at time step $t$. It represents the agent's current estimate of the expected reward for action $a$, based on past experience.
- The square root term $ln(t)/N_t​(a)$​​ represents the uncertainty or variance in the estimate of action $a$. It ensures that actions with fewer trials are given a chance to be chosen (exploration).
    - $ln(t)$ is the natural logarithm of the current time step, which grows slowly as time goes on.
    - $N_t​(a)$ is the number of times that action $a$ has been selected up to time step $t$.
- The constant $c$ is a tunable parameter that determines the degree of exploration. A larger value for $c$ encourages more exploration, while a smaller value encourages exploitation of the already known actions.		


# 2- Markov Decision Processes

## Markov Decision Processes
The k-Armed Bandit problem does not account for the fact that different situations call for different actions. It’s also limited in another way. A bandit rabbit would only be concerned about immediate reward and so it would go for the carrot. But a better decision can be made by considering the long-term impact of our decisions. Now, let’s look at how the situation changes as the rabbit takes actions. We will call these situations states. In each state the rabbits selects an action. For instance, the rabbit can choose to move right. Based on this action the world changes into a new state and produces a reward.

In this case, the rabbit eats the carrot and receives a reward of plus 10. However, the rabbit is now next to the tiger. Let’s say the rabbit chooses the left action. The world changes into a new state and the tiger eats the rabbit and the rabbit receives a reward of minus 100. The next diagram shows the iteration mentioned before and the relation agent - environment

From a set of State-Action the agent receive a Reward and ended in state St+1. The transition dynamics function p, formalizes this notion:

														$$p(s', r|s, a)$$
														
											$$p: S \times R \times S \times A \rightarrow [0, 1]$$
											
							$$\sum_{s' \in S'} \sum_{r \in R} p(s', r|s, a) = 1, \forall s \in S, a \in A(s)$$

Now, to explain the given equations:

- $p(s′,r∣s,a)$ is the probability of transitioning to state $s′$ and receiving $a$ reward $r$ when taking action a in state $s$.
- The notation $p:S×R×S×A→[0,1]$ defines the function $p$ as a function that takes four arguments (a state from set $S$, a reward from set $R$, another state from set $S$, and an action from set $A$) and maps these to a real number in the interval $[0,1]$ which is the range of probabilities.
- The normalization property $∑s′∈S′​∑r∈R​p(s′,r∣s,a)=1$ states that if you sum the probabilities of all possible next states $s'$ and all possible rewards $r$ for a given state $s$ and action $a$, the result must be 1. 
- Note that future state and reward only depends on the current state and action. This is called the Markov property. It means that the present state is sufficient and remembering earlier states would not improve predictions about the future

## Goal of  Reinforcement Learning
In reinforcement learning, the agent’s objective is to **maximize future reward**. We will formalize this notion. Perhaps we can just maximize the immediate reward as we did in bandits. Unfortunately this won’t work in an MDP. An action on this time step might yield large reward because the agent of transition into a state that yields low reward. So what looked good in the short-term, might not be the best in the long-term. The return at time step $t$, is simply the sum of rewards obtained after time step t. We denote the return with the letter $G$. The return $G$ is a random variable because the dynamics of the MDP can be stochastic.

														$$Gt .= Rt+1 + Rt+2 + Rt+3 + . .$$
														
We maximize the expected return. For this to be well-defined, the sum of rewards must be finite. Specifically, let say there is a final time step called T where the agent environment interaction ends.

In the simplest case, the interaction naturally breaks into chunks called episodes. Each episode begins independently of how the previous one ended. At termination, the agent is reset to a start state. Every episode has a final state which we call the terminal state. We call these tasks episodic tasks

The reward signal is your way of communicating to the agent what you want achieved, not how you want it achieved.

## Continuing Tasks
There is a problem when implementing the Return representation of episodic tasks into continuing tasks. Episodic Tasks: Interaction breaks naturally into episodes; Each episode ends in a terminal state; Episodes are independent. Continuing Tasks  is  Interaction goes on actually and doesn't had terminal state. 

The return formulation can then be modified to include discounting. The effect of discounting on the return is simple, immediate rewards contribute more to the sum. Rewards far into the future contribute less because they are multiplied by Gamma raised to successively larger powers of k. 

								         $$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots + \gamma^{k-1}R_{t+k} + \ldots$$
												  
										            	     $$= \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$
## Effect of  $\gamma$  on agent behavior
The return at time $t$,  $G_t$ , is defined as:  
$$  G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots + \gamma^{k-1} R_{t+k} + \ldots $$ 
with $\gamma = 0$: 
							     $$ G_t = R_{t+1} + 0 \cdot R_{t+2} + 0^2 \cdot R_{t+3} + \ldots + 0^{k-1} \cdot R_{t+k} + \ldots $$
									     
																$$ G_t = R_{t+1} $$
														
With $gamma = 0$ it is evident that the agent only cares about the immediate reward. This is called a **Short-sighted agent**. 
With $\gamma = 1$ the agent takes future rewards into account more strongly. This is called a **Far-sighted agent**. 

## Recursive nature of returns
The return can also be expressed recursively:  
									$$ G_t = R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + \gamma^2 R_{t+4} + \ldots) $$ 
Recall that:  
										$$ R_{t+2} + \gamma R_{t+3} + \gamma^2 R_{t+4} + \ldots =  G_{t+1} $$
Thus: 
														$$ G_t = R_{t+1} + \gamma G_{t+1} $$ 

**Remember** that by definition $G_{t+1}$  is equal to zero at the end of the final episode.

# 3- Value Function and Bellman Equations

## Specifying Policies
Policies are how and agent select an action. 
Formally, a policy is a mapping from states to probabilities of selecting each possible action. If the agent is following policy $\pi$  at time $t$, then $\pi(a|s)$ is the probability that $A_t = a$ if $S_t = s$. Like p, $\pi$ is an ordinary function; the “|” in the middle of $\pi(a|s)$ merely reminds us that it defines a probability distribution over $a \in A(s)$ for each $s \in S$.

 In the simplest case, a policy maps each state to a single action. This kind of policy is called the deterministic policy. We will use the fancy Greek letter $\pi$ to denote a policy. $\pi(S)$ represents the action selected in state $S$ by the policy $\pi$. In this example, $\pi$ selects the action A1 in state S0 and action A0 in states S1 and S2. We can visualize a deterministic policy with a table. Each row describes the action chosen by $\pi$ in each state. Notice that the agent can select the same action in multiple states, and some actions might not be selected in any state. The arrows describe one possible policy, which moves the agent towards its house. Each arrow tells the agent which direction to move in each state.

In general, a policy assigns probabilities to each action in each state. We use the notation $\pi(a|s)$, to represent the probability of selecting action A in a state S. A stochastic policy is one where multiple actions may be selected with non-zero probability. Here we show the distribution over actions for state S0 according to $\pi$. Remember that $\pi$ specifies a separate distribution over actions for each state. So we have to follow some basic rules. The sum over all action probabilities must be one for each state, and each action probability must be non-negative.


In this MDP, we can define a policy that chooses to go either left or right with equal probability. We might also want to define a policy that chooses the opposite of what it did last, alternating between left and right actions. However, that would not be a valid policy because this is conditional on the last action. That means the action depends on something other than the state. It is better to think of this as a requirement on the state, not a limitation on the agent. In MDPs, we assume that the state includes all the information required for decision-making. 

**An agent’s behavior is specified by a policy that maps the state to a probability distribution over actions, and the policy can depend only on the current state, not on other things like time or previous states.**

## Value Function
The value function of a state $S$ under a policy $\pi$, denoted $v_\pi(s)$, is the expected return when starting $s$ and following $\pi$. We can define formally: 
$$ v_\pi(s) = \mathbb{E}_\pi\left[G_t \mid S_t = s\right] = \mathbb{E}_\pi\left[\sum_{k=0}^\infty \gamma^k R_{t+k+1} \mid S_t = s\right], \ for\ all \ s \in S $$
It answers the question: "What is the expected reward if I start in this state and follow a particular policy? . 
Value function enable to judge the quality of different policies. 

## Action Value Function
The action value function, or Q-function, denoted as $Q_\pi(s,a)$, estimates the expected return from taking an action $a$ in a state $s$ and then following a policy $π$. It helps determine the expected effectiveness of each possible action from each state. This function answers the question: "What is the expected reward if I start in this state, take this action, and then follow a particular policy?". it's defined as follow: 

$$q_\pi(s, a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]$$



## Bellman equation 
*The bellman equation relate the value of current state with the value of future state without waiting to observe all the future rewards (value of a state and it's possible successor).*
Means that my value function (or action value function) is a function of my value function at the next state. 
To define this equation, we start by recall the value function: 
$$
v_\pi = \mathbb{E}_\pi [G_t| S_t = s] 
$$
$$
= \mathbb{E} [R_{t+1}+ \gamma G_{t+1}| S_t=s]
$$

Now we expand the expected return as a sum over possible action and we expend over possible rewards and next states condition on state $S$ and action $a$: 
$$
= \sum_a \pi(a|s) \sum_{s'} \sum_r p(s',r|s,a) [r + \gamma \mathbb{E} [G_{t+1}| S_{t+1} = s']
$$
This line expands the expectation in terms of the probability distributions involved in the process:
- $\sum_a \pi(a|s)$ : the sum of all possible action given the policy. 
-  $\sum_{s'} \sum_r p(s',r|s,a)$: The probability of transitioning to state s′ and receiving reward r given the current state s and action a.
    
- The term inside the expectation:  $[r + \gamma \mathbb{E} [G_{t+1}| S_{t+1} = s']$: Represents the immediate reward plus the discounted expected return from the next state s′.

The value function  is expressed as the expected return starting from state s under policy π. This is broken down into:  The sum of the immediate rewards,  The future rewards discounted by γ- Weighted by the probabilities of taking actions according to the policy and transitioning between states. 
This equality encapsulates the recursive nature of the value function.

At the end we have the following equation: 
$$
= \sum_a \pi(a|s) \sum_{s'} \sum_r p(s',r|s,a) [r + \gamma V(s')]
$$

We do the same with value action function: 
$$
q_\pi(s,a) = \mathbb{E}_\pi [G_t | A_t=a, S_t=s] 
$$
$$
= \sum_{s'} \sum_{r} p(s',r|s,a)[r+\gamma \mathbb{E}_\pi[G_{t+1}|S_{t+1}=s']]
$$
$$
= \sum_{s'} \sum_{r} p(s',r|s,a)[r+\gamma \sum_{a'} \pi(a'|s')\mathbb{E}_\pi[G_{t+1}|S_{t+1}=s',A_{t+1}=a']]
$$
$$
= \sum_{s'} \sum_{r} p(s',r|s,a)[r+\gamma \sum_{a'} \pi(a'|s')q_\pi(s',a')]
$$
These equation provide the relationship between the values of a state or state action pair and the possible next states or next state action pairs. 
## Why bellman equation is so important 

The Bellman equation is important because it helps find the best actions by breaking down complex problems into simpler steps, making it easier to maximize long-term rewards.
The Bellman equation forms the basis of a number of ways to compute, approximate, and learn $v_{\pi}$. 

## Optimal policies 
The point of policies is to select behavior. Given this set of behavior, we then find the total value for each state. Then, pick whichever behavior leads to the highest value possible. Ultimately, the value of each state reflects what will be the total reward in the long run. We will say that policy $\pi_1$ is as good or better than policy $\pi_2$ if, for every state $s$, it is greater than or equal to the total value for every state $s$. $\pi_1 \geq \pi_2$

Thus, an optimal policy will have the highest possible value in every state. **There's always at least one optimal policy**, but there may be more than one.  We’ll use the notation $\pi_*$ to denote any optimal policy. 


Simple MDP Example:
Policies :  - $\pi_1(X) = A_1$ - $\pi_2(X) = A_2$ 

Is $\pi_1$ or $\pi_2$ better? Depends on the discount factor $\gamma$: 
 
- For $\gamma = 0$: - $V_{\pi_1}(X) = 1$ - $V_{\pi_2}(X) = 0$ - $\therefore \pi_1$ is optimal 

- For $\gamma = 0.9$: - $V_{\pi_1}(X) = \sum_{k=0}^{\infty} (0.9)^k = \frac{1}{1-0.9} \approx 5.3$ - $V_{\pi_2}(X) = \sum_{k=0}^{\infty} (0.9)^{2k+1} \cdot 2 = \frac{0.9}{1-0.9^2} \cdot 2 \approx 9.5$ - $\therefore \pi_2$ is optimal
How do we find the optimal policy for harder examples? In total, there are $A^S$ possible policies ($A = \text{actions}$, $S = \text{states}$), and calculating the value function for each one is inefficient. - Use Bellman Optimality Equations to find the optimal policy.

## Optimal value function 

The value function for the optimal policy thus has the greatest value possible in every state. We can express this mathematically, by writing that $v_{\pi_*}(s)$ is equal to the maximum value over all policies. This holds for every state in our state-space. Taking a maximum over policies might not be intuitive. So let's take a moment to break down what it means. Imagine we were to consider every possible policy and compute each of their values for the state $S$. The value of an optimal policy is defined to be the largest of all the computed values. We could repeat this for every state and the value of an optimal policy would always be the largest. All optimal policies have this same optimal state-value function, which we denote by $v_*$. Optimal policies also share the same optimal action-value function, which is again the maximum possible for every state action pair. We denote this shared action value function by $q_*$.

$$ v_* \rightarrow v_{\pi_*}(s) \equiv \mathbb{E}_\pi \left[ G_t \mid S_t = s \right] = \max_\pi v_{\pi}(s) \quad \forall s \in S $$

$$ q_* \rightarrow q_{\pi_*}(s, a) = \max_\pi q_{\pi}(s, a) \quad \forall s \in S \text{ and } a \in A $$

This is simply the Bellman equation we introduced previously for the specific case of an optimal policy. However, because this is an optimal policy, we can rewrite the equation in a special form, which doesn’t reference the policy itself. Remember there always exists an optimal deterministic policy, one that selects an optimal action in every state. Such a deterministic optimal policy will assign Probability 1 for an action that achieves the highest value and Probability 0, for all other actions. We can express this another way by replacing the $\sum_a \pi_*(a \mid s)$ with $\max_a$. Notice that $\pi_*$ no longer appears in the equation. We have derived a relationship that applies directly to $v_*$ itself. We call this special form, the Bellman optimality equation for $v_*$.

Recall that $v_{\pi}(s) = \sum_a \pi(a \mid s) \sum_{s'} \sum_r p(s', r \mid s, a) [r + \gamma v_{\pi}(s')]$

$$ v_*(s) = \sum_a \pi_*(a \mid s) \sum_{s'} \sum_r p(s', r \mid s, a) [r + \gamma v_*(s')] $$

$$ v_*(s) = \max_a \sum_{s'} \sum_r p(s', r \mid s, a) [r + \gamma v_*(s')] \leftarrow \text{Bellman Optimality Equation for } v_* $$

We can make the same replacement in the Bellman equation for the action-value function. Here the optimal policy appears in the inner sum. Once again, we replace the sum over $\pi_*$ with a max over $a$. This gives us the Bellman optimality equation for $q_*$.

Recall that $q_{\pi}(s, a) = \sum_{s'} \sum_r p(s', r \mid s, a) [r + \gamma \sum_{a'} \pi(a' \mid s') q_{\pi}(s', a')]$

$$ q_*(s, a) = \sum_{s'} \sum_r p(s', r \mid s, a) \left[ r + \gamma \max_{a'} q_*(s', a') \right] $$

$$ q_*(s, a) = \sum_{s'} \sum_r p(s', r \mid s, a) \left[ r + \gamma \max_{a'} q_*(s', a') \right] \leftarrow \text{Bellman Optimality Equation for } q_* $$

The Bellman’s optimality equation gives us a similar system of equations for the optimal value. One natural question is, can we solve this system in a similar way to find the optimal state-value function? Unfortunately, the answer is no. Taking the maximum over actions is not a linear operation. So standard techniques from linear algebra for solving linear systems won’t apply.

$$ v_{\pi}(s) = \sum_a \pi(a \mid s) \sum_{s'} \sum_r p(s', r \mid s, a) [r + \gamma v_{\pi}(s')] $$



$$ v_*(s) = \max_a \sum_{s'} \sum_r p(s', r \mid s, a) [r + \gamma v_*(s')] \rightarrow \text{the term max is not linear} $$

## Using optimal value function to get optimal policy 
In general, having $v_*$ makes it relatively easy to work out the optimal policy as long as we also have access to the dynamics function $p$. For any state, we can look at each available action and evaluate the boxed term. There will be some action for which this term obtains a maximum. $v_*$ is equal to the maximum of the boxed term over all actions. $\pi_*$ is the $argmax$, which simply means the particular action which achieves this maximum: $$ v_*(s) = \max_a \sum_{s'} \sum_r p(s', r \mid s, a) \left[ r + \gamma v_*(s') \right] $$ $$ \pi_*(s) = \arg\max_a \sum_{s'} \sum_r p(s', r \mid s, a) \left[ r + \gamma v_*(s') \right] $$

If instead we have access to $q_*$, it’s even easier to come up with the optimal policy. In this case, we do not have to do a one-step look ahead at all. We only have to select any action $a$ that maximizes $q(s, a)$. The action-value function caches the results of a one-step look ahead for each action. In this sense, the problem of finding an optimal action-value function corresponds to the goal of finding an optimal policy. As a reminder, the next optimal equations summarize how to obtain $\pi_*$: $$ \pi_*(s) = \arg\max_a \sum_{s'} \sum_r p(s', r \mid s, a) \left[ r + \gamma v_*(s') \right] $$ $$ \pi_*(s) = \arg\max_a q_*(s, a) $$

# 4 -  Dynamic Programming

**Dynamic programming** refers to a collection of algorithms that can be used to compute optimal policies given a perfect model of the environment as an MDP. 
Classical DP models are limited because they assume a perfect model and are computationally expensive. 
The key idea of DP, and of reinforcement learning generally, is the use of value functions to organize and structure the search for good policies.
We assume finite MDP's:
	 - States, actions, and rewards are finite $(S, A, R)$. 
	 - Dynamics are given by a set of probabilities $p(s', r \mid s, a)$ for all $s \in S$, $a \in A(s)$, $r \in R$, $s' \in S^+$. 
	 - $S^+$ is $S$ plus the terminal state for episodic tasks. 
	 - TL;DR: DP algorithms are obtained by turning Bellman equations into update rules for improving approximations of the desired value functions. 
	 - Bellman equations define iterative algorithms for both **policy evaluation** and **control**.
## Policy Evaluation 
We want to compute the state-value function $V_{\pi}$)for an arbitrary policy $\pi$. This is called **policy evaluation** or the **prediction problem**. 
*Recall*
$$
 V_{\pi}(s) \equiv \mathbb{E}_{\pi} [G_t | S_t = s] 
 $$
 $$
 = \mathbb{E}_{\pi} [R_{t+1} + \gamma G_{t+1} | S_t = s] 
 $$
 $$
 = \mathbb{E}_{\pi} [R_{t+1} + \gamma V_{\pi}(S_{t+1}) | S_t = s] 
$$
$$
 = \sum_a \pi(a | s) \sum_{s', r} p(s', r | s, a) [r + \gamma V_{\pi}(s')] 
$$

$V_{\pi}$ is guaranteed if $\gamma < 1$ or $\pi$ guarantees that all states eventually terminate. 

To find a system of $V_{\pi}(s)$ to solve: 

1. Assign \(V_0\) = arbitrary real number.
2. $$
V_{k+1}(s) = \mathbb{E}_{\pi} [R_{t+1} + \gamma V_k(S_{t+1}) | S_t = s]  = \sum_a \pi(a | s) \sum_{s', r} p(s', r | s, a) [r + \gamma V_k(s')] <- **Update\_rule** 
$$
This is the **iterative policy evaluation**. 

- Iterative Policy Evaluation Algorithm: 
	- **For estimating** $V = V_{\pi}$
	- **Input**: $\pi$, the policy to be evaluated. 
	- **Hyperparameter**: $\theta$, small threshold for determining accuracy of estimation, $\theta > 0$. 
	- Initialize $V(s)$ for all $s \in S^+$ arbitrarily, except $V(\text{terminal}) = 0$.
		- Loop: $$ \begin{align*} 1. & \quad \Delta = 0 \\ 2. & \quad \text{For each } s \in S: \\ & \quad \quad v \leftarrow V(s) \\ & \quad \quad V(s) \leftarrow \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) [r + \gamma V(s')] \\ & \quad \quad \Delta \leftarrow \max(\Delta, |v - V(s)|) \\ 3. & \quad \text{Break when } \Delta < \theta \end{align*} $$
## Policy Improvement (control) 




**Recall**  $$ q_{\pi}(s, a) \equiv \mathbb{E}_{\pi} [R_{t+1} + \gamma V_{\pi} (S_{t+1}) | S_t = s, A_t = a] $$ $$ = \sum_{s', r} p(s', r | s, a) [r + \gamma V_{\pi}(s')] $$Theorem: Let $\pi$ and $\pi'$ be any pair of deterministic policies such that for all $s \in S$:  $$ q_{\pi}(s, \pi'(s)) \geq V_{\pi}(s) \quad \forall s $$  Then $\pi' \geq \pi$.  i.e., It must obtain greater or equal expected return from all states $s \in S \rightarrow V_{\pi'}(s) \geq V_{\pi}(s)$.  
**Proof**:  $$ \begin{aligned} v_{\pi}(s) & \leq q_{\pi}(s, \pi'(s)) \\ & = \mathbb{E}_{\pi} [R_{t+1} + \gamma V_{\pi} (S_{t+1}) | S_t = s, A_t = \pi'(s)] \\ & = \mathbb{E}_{\pi} [R_{t+1} + \gamma V_{\pi} (S_{t+1}) | S_t = s] \\ & = \sum_{s', r} p(s', r | s, \pi'(s)) [r + \gamma V_{\pi}(s')] \\ & = V_{\pi}(s) \\ \end{aligned} $$  **Policy improvement** is the process of making a new policy that improves on an original one by making it *greedy* with respect to the value function of the original policy:$$ \pi'(s) \equiv \arg\max_a q_{\pi}(s, a) $$  $$ = \arg\max_a \mathbb{E} [R_{t+1} + \gamma V_{\pi} (S_{t+1}) | S_t = s, A_t = a] $$  $$ = \arg\max_a \sum_{s', r} p(s', r | s, a) [r + \gamma V_{\pi}(s')] $$  If $\pi' = \pi$:  Then $V_{\pi} = V_{\pi'}$ for all $s \in $, and it follows that:  $$ V_{\pi}(s) = \max_a \mathbb{E} [R_{t+1} + \gamma V_{\pi} (S_{t+1}) | S_t = s, A_t = a] $$  $$ = \max_a \sum_{s', r} p(s', r | s, a) [r + \gamma V_{\pi}(s')] $$  Which is the same as the Bellman Optimality Equation:  $$ V_*(s) = \max_a \sum_{s', r} p(s', r | s, a) [r + \gamma V_*(s')] $$  Thus, both $\pi$ and $\pi'$ are optimal policies.  How do we actually use DP to find $\pi'$?  We have iterative policy evaluation for $V_{\pi}$, but how do we incorporate changing $\pi$? -> **Policy Iteration**


## Policy Iteration 

$$ \pi_0 \xrightarrow{E} V_{\pi_0} \xrightarrow{I} \pi_1 \xrightarrow{E} V_{\pi_1} \xrightarrow{I} \ldots \xrightarrow{I} \pi_* \xrightarrow{E} V_* $$ - **E**: policy evaluation and  **I**: policy improvement 
 Since a finite MDP has a finite number of policies, this process must converge to an optimal policy in a finite number of iterations. 
 Policy Iteration (using iterative policy evaluation) for estimating $\pi$ $\approx$ $\pi_*$ : 
 1. **Initialization**: - $V(s) \in \mathbb{R}$ and $\pi(s) \in A(s)$ arbitrarily for all $s \in S$ 
 2. **Policy Evaluation**: 
	 - Loop: $$ \begin{aligned} & \Delta \leftarrow 0 \\ & \text{Loop for each } s \in S: \\ & \quad v \leftarrow V(s) \\ & \quad V(s) \leftarrow \sum_{s', r} p(s', r \mid s, \pi(s)) [r + \gamma V(s')] \\ & \quad \Delta \leftarrow \max(\Delta, |v - V(s)|) \\ \end{aligned} $$ - until $\Delta < \theta$ (a small positive number determining the accuracy of estimation) 
   3 . **Policy Improvement**: 
   - $\text{policy-stable} \leftarrow \text{true}$ 
   - For each $s \in S$: 
$$\text{old-action} \leftarrow \pi(s)   $$$$ \pi(s) \leftarrow \arg\max_a \sum_{s', r} p(s', r \mid s, a) [r + \gamma V(s')] $$ - If  $\text{old-action} \neq \pi(s)$, then $\text{policy-stable} \leftarrow \text{false}$ 
- If $\text{policy-stable}$, then stop and return $V \approx v_*$ and  $\pi \approx \pi_*$; else go to 2.  
## Flexibility of the Policy Iteration Framework 
![[2024-08-02_17-26.png]]
- Each policy improvement step makes policy a little more greedy, but not totally greedy.
  - We should still get $\pi_*$

**Generalized policy iteration** refers to all the different combinations of policy evaluation and improvement. One algorithm/combination is **value iteration**.
## Value iteration
**Same as policy iteration but now we update using action that maximizes current value estimate**
Policy evaluation is stopped after one sweep. $V_k$ is still shown to converge to $V_*$ under the same conditions. $$ V_{k+1}(s) \equiv \mathbb{E}_{\pi} [R_{t+1} + \gamma V_k (S_{t+1}) | S_t = s] $$ $$ = \max_a \sum_{s', r} p(s', r \mid s, a) [r + \gamma V_k(s')] $$*Policy Evaluation* $$ V_{k+1}(s) \equiv \mathbb{E}_{\pi} [R_{t+1} + \gamma V_k (S_{t+1}) | S_t = s] $$ $$ = \sum_a \pi(a | s) \sum_{s', r} p(s', r \mid s, a) [r + \gamma V_k(s')] $$  Value Iteration, for estimating $\pi \approx \pi_*$ 
Algorithm parameter: a small threshold $\theta > 0$ determining accuracy of estimation Initialize $V(s)$, for all $s \in S^+$, arbitrarily except that $V(\text{terminal}) = 0$. 
Loop: $$ \begin{aligned} & \Delta \leftarrow 0 \\ & \text{Loop for each } s \in S: \\ & \quad v \leftarrow V(s) \\ & \quad V(s) \leftarrow \max_a \sum_{s', r} p(s', r \mid s, a) [r + \gamma V(s')] \\ & \quad \Delta \leftarrow \max(\Delta, |v - V(s)|) \\ \end{aligned} $$ - until $\Delta < \theta$ 

-Output a deterministic policy, $\pi \approx \pi_*$, such that: $$ \pi(s) = \arg\max_a \sum_{s', r} p(s', r \mid s, a) [r + \gamma V(s')] $$ 
# Course 2: 

# 1- Monte Carlo methods for prediction and control

PREDICTION = Policy Evaluation, learn the value function that estimate the expected reward starting at state S. 
CONTROL = Policy Improvement, learn the optimal policy.

## What is Monte Carlo 
The term Monte Carlo is often used more broadly for any estimation method that relies on repeated random sampling. In RL Monte Carlo methods allow us to estimate values directly from experience, from sequences of *states, actions and rewards*. Learning from experience is striking because the agent can accurately estimate a value function without prior knowledge of the environment dynamics. To use a pure Dynamic Programming approach, the agent needs to know the environments transition probabilities. In some problems we do not know the environment transition probabilities and  the computation can be error-prone and tedious. 

In  RL we want to learn a value function. Value functions represent expected returns. So a Monte Carlo method for learning a value function would first observe multiple returns from the same state. Then, it average those observed returns to estimate the expected return from that state. As the number of samples increases, the average tends to get closer and closer to the expected return. These returns can only be observed at the end of an episode. So we will focus on Monte Carlo methods for episodic tasks. 

The pseudo-code of the Monte Carlo prediction can be seen below: 

Input: a policy $\pi$ to be evaluated 
**Initialize:**
- $V(s) \in \mathbb{R}$, arbitrarily, for all $s \in \mathcal{S}$
- Returns(s) ← an empty list, for all $s \in \mathcal{S}$

**Loop forever (for each episode):**
1. Generate an episode following $\pi$: $S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_{T-1}, A_{T-1}, R_T$
2. $G \leftarrow 0$
3. Loop for each step of episode, $t = T-1, T-2, \ldots, 0$:
    - $G \leftarrow \gamma G + R_{t+1}$
    - Append $G$ to Returns($S_t$)
    - $V(S_t) \leftarrow$ average(Returns($S_t$))


Some implication of the Monte Carlo learning: 
- Monte Carlo learn directly from experience no need to save the model of the environment. 
- Monte Carlo can estimate the value of each state independently of the value of the other state not like Dynamic programming (DP).
- The computation to update the value of each state doesn't depend of the size of the MDP but depend on the larger of the episode.  

## Monte Carlo prediction 
Monte Carlo approach for learning the state-value function for a given policy.
**Goal**: Estimate $V_{\pi}(s)$ given a set of episodes following $\pi$ and passing through $s$. 
Each occurrence of state $s$ in an episode is called a **visit** to $s$.  $s$ may be visited multiple times in the same episode. The first visit to $s$ in an episode is called the **first visit**. 

First-Visit MC Prediction Estimates $V_{\pi}(s)$ as the average of returns following **first visits** to $s$.  Since $s$ can be visited multiple times in an episode, we only take the average of returns after the first visit. 
Every-Visit MC Method Estimates $V_{\pi}(s)$ as the average of returns following **all visits** to $s$. 

Algorithm 1: First-Visit MC Prediction 
**Input**: policy $\pi$, positive integer num_episodes 
**Output**: value function $V$ ≃ $v_{\pi}$, if num\_episodes is large enough 
Initialize $N(s) = 0$ for all $s \in S$ 
Initialize $Return(s) = 0$ for all $s \in S$ 
for episode $e = 1$ to $e \leftarrow \text{num\_episodes}$ 
	do Generate, using $\pi$, an episode $S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_{T-1}, A_{T-1}, R_T$ 
	$G \leftarrow 0$ 
		for time step $t = T - 1$ to $t = 0$ (of the episode $e$) 
			do $G \leftarrow G + R_{t+1}$ 
			if state $S_t$ is not in the sequence $S_0, S_1, \ldots, S_{t-1}$ 
				then $Return(S_t) \leftarrow Return(S_t) + G$ 
				$N(S_t) \leftarrow N(S_t) + 1$ 
			end 
		end 
end 
$V(s) \leftarrow \frac{Return(s)}{N(s)}$ for all $s \in S$ 
return $V$


Algorithm 2 : Every-visit MC Prediction 
**Input**: policy $\pi$, positive integer num_episodes 
**Output**: value function $V$ ≃ $v_{\pi}$, if num\_episodes is large enough 
Initialize $N(s) = 0$ for all $s \in S$ 
Initialize $Return(s) = 0$ for all $s \in S$ 
for episode $e = 1$ to $e \leftarrow \text{num\_episodes}$ 
	do Generate, using $\pi$, an episode $S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_{T-1}, A_{T-1}, R_T$ 
	$G \leftarrow 0$ 
		for time step $t = T - 1$ to $t = 0$ (of the episode $e$) 
			do $G \leftarrow G + R_{t+1}$ 
			$Return(S_t) \leftarrow Return(S_t) + G$ 
			$N(S_t) \leftarrow N(S_t) + 1$ 
			end 
		end 
end 
$V(s) \leftarrow \frac{Return(s)}{N(s)}$ for all $s \in S$ 
return $V$


**Implications of Monte Carlo Learning**:
- No need to keep a large model of the environment (because it learns from experience) 
- The value of a state can be estimated independently of the other states (Not like in Dynamic Programming) 
- The computation needed to update the value of each state doesn’t depend on the size of the MDP (only the length of the episode and the number of states)


## Monte Carlo Estimation of Action Values 
Without a model, we need to estimate action values instead of state values to determine a policy. 
Goal is to estimate $q_{\pi}$ with Monte Carlo methods. 
A state-action pair $(s, a)$ is **visited** if state $s$ is visited and action $a$ is taken in it. 
The problem with using every/first-visit Monte Carlo methods is that many state-action pairs may not be visited. If $\pi$ is deterministic, following $\pi$ will only observe returns for one action per state. 
We need to estimate the value of all the actions from each state to compare alternatives.  This is the problem of **maintaining exploration**.
*Exploring Starts*: One way to solve this problem is to make episodes start in a state-action pair, with each pair having a nonzero probability of being chosen.  As the number of episodes $\rightarrow \infty$, all state-action pairs will be visited.

## Using Monte Carlo for control 
*recall*, Control refers to the process of learning an optimal policy that maximize the cumulative reward an agent can achieve in an environment. 

Same Idea than Dynamic programming, we have policy evaluation and policy improvement. 

- Same pattern outlined in DP chapter: $$ \pi_0 \xrightarrow{E} q_{\pi_0} \xrightarrow{I} \pi_1 \xrightarrow{E} q_{\pi_1} \xrightarrow{I} \ldots \xrightarrow{I} \pi_* \xrightarrow{E} q_{\pi_*} $$

 $E$: Evaluation - $I$: Improvement - $\pi_0$: Random policy - Since we use action-value functions, we don't need the model to construct the greedy policy. $$ \pi(s) = \arg\max_a q(s, a) $$ 
 
 Monte Carlo ES (Exploring Starts), for estimating $\pi \approx \pi_*$ algorithm is as follow: 
 **Initialize:** 
		 -  $\pi(s) \in \mathcal{A}(s)$ (arbitrarily), for all $s \in \mathcal{S}$
		 -  $Q(s, a) \in \mathbb{R}$ (arbitrarily), for all $s \in \mathcal{S}$, $a \in \mathcal{A}(s)$ 
		 -  Returns(s, a) ← empty list, for all $s \in \mathcal{S}$, $a \in \mathcal{A}(s)$ 
 **Loop forever (for each episode):** 
	 1. Choose $S_0 \in \mathcal{S}, A_0 \in \mathcal{A}(S_0)$ randomly such that all pairs have probability $> 0$ 
	 2. Generate an episode from $S_0, A_0$, following $\pi$: $S_0, A_0, R_1, \ldots, S_{T-1}, A_{T-1}, R_T$ 
	 3. $G \leftarrow 0$ 
 **Loop for each step of episode,** $t = T-1, T-2, \ldots, 0$: 
 - $G \leftarrow \gamma G + R_{t+1}$ 
 - Unless the pair $S_t, A_t$ appears in $S_0, A_0, S_1, A_1, \ldots, S_{t-1}, A_{t-1}$: 
	 - Append $G$ to Returns(S_t, A_t) 
	 - $Q(S_t, A_t) \leftarrow$ average(Returns(S_t, A_t))
	- $\pi(S_t) \leftarrow \arg\max_a Q(S_t, a)$


## Monte Carlo Control without Exploring Starts
In many examples, it may be difficult to sample a random state-action pair. So how can we learn all the action-values without exploring starts? 
Recall: $\epsilon$-Greedy actions have a chance to select a random action. 
$\epsilon$-Greedy policies are a subset of $\epsilon$-Soft policies. 
**$\epsilon$-Greedy vs. $\epsilon$-Soft Policies**

$\epsilon$-Soft policies take each action with probability at least $\frac{\epsilon}{\#\text{actions}} = \frac{\epsilon}{|A(s)|}$. 
Forces the agent to continuously explore. Exploring starts is no longer needed. 
$\epsilon$-Soft policies can only be used to find the optimal $\epsilon$-Soft policy, not the actual optimal policy. - This is because they are not deterministic. 
$Q$-learning is used to find the optimal policy (covered later).

Algorithm: On-policy First-Visit Monte Carlo Control (for $\epsilon$-soft policies), estimates $\pi \approx \pi_*$ 
**Algorithm parameter**: small $\epsilon > 0$ 
**Initialize**: 
- $\pi \leftarrow$ an arbitrary $\epsilon$-soft policy 
- $Q(s, a) \in \mathbb{R}$ (arbitrarily), for all $s \in S$, $a \in A(s)$ 
- $Returns(s, a) \leftarrow$ empty list, for all $s \in S$, $a \in A(s)$ 
**Repeat forever (for each episode):**
1. Generate an episode following $\pi$: $S_0, A_0, R_1, \dots, S_{T-1}, A_{T-1}, R_T$ 
2. $G \leftarrow 0$ 
3. Loop for each step of episode, $t = T-1, T-2, \dots, 0$: - $G \leftarrow \gamma G + R_{t+1}$ 
	- Unless the pair $S_t, A_t$ appears in $S_0, A_0, S_1, A_1, \dots, S_{t-1}, A_{t-1}$: 
		- Append $G$ to $Returns(S_t, A_t)$ 
		- $Q(S_t, A_t) \leftarrow$ average($Returns(S_t, A_t)$) 
		- $A^* \leftarrow \underset{a}{\arg\max} \, Q(S_t, a)$ (with ties broken arbitrarily)
		- For all $a \in A(S_t)$:$$ \pi(a | S_t) \leftarrow \begin{cases} 1 - \epsilon + \frac{\epsilon}{|A(S_t)|} & \text{if } a = A^* \\ \frac{\epsilon}{|A(S_t)|} & \text{if } a \neq A^* \end{cases} $$

## policy prediction vie Importance sampling
Because with $\epsilon$-soft we learn the optimal $\epsilon$-soft policy not the optimal policy, we introduce the off-policy method. 

**On-policy Methods**: On-policy methods attempt to evaluate or improve the policy that is used to make decisions. - *e.g.*, $\epsilon$-soft policies, exploring starts  
**Off-policy Methods**: Off-policy methods evaluate or improve a policy **different** from the one used to generate the data. - *e.g.*, learning the optimal policy while following a completely random one.

How can agents learn the optimal policy while behaving according to a non-optimal, exploratory policy? - Just use two policies: 
**Target Policy** - The one that is learned and becomes the optimal policy. 
**Behavior Policy** - The exploratory one used to generate behavior. 

The learning is from data "off" the target policy, hence **off-policy learning**. 
Notation: 
	- **Target Policy**: $\pi(a|s)$ 
	- **Behavior Policy**: $b(a|s)$ 
	- The behavior policy must cover the target policy. - i.e., If $\pi(a|s) > 0$ for some $(a, s)$, then $b(a|s) > 0$. 
	- **On-policy**: $\pi(a|s) = b(a|s)$

Off-policy learning offers key benefits:

- **Data Efficiency**: It uses data from different policies, maximizing data utilization.
- **Exploration-Exploitation**: It separates the exploration-focused behavior policy from the exploitation-focused target policy.
- **Temporal Decoupling**: It allows updating the target policy independently of data collection, enhancing stability.
- **Generalization**: It enables learning value functions that generalize beyond the current policy.
- **Offline/Batch Learning**: It supports learning from pre-collected datasets without needing real-time interaction.
## Importance Sampling
- Important sampling allow us to do off-policy learning, learning about one policy while following another.
- **Why is Importance Sampling Needed in Reinforcement Learning?** In off-policy learning, you want to learn the value of a target policy $\pi$, but the data (state-action-reward sequences) is generated by a different behavior policy $b$. The key challenge here is that the distribution of state-action pairs under $b$ can be very different from the distribution under $\pi$. To correctly update the value estimates for $\pi$, you need to adjust for this difference, which is where importance sampling comes in.
- Since the distribution of actions under the behavior policy may differ from the target policy, importance sampling adjusts the estimates of expected returns by weighting them according to the likelihood ratio between the target policy and the behavior policy.

We have some random variable $x$ that’s being sampled from a probability distribution $b$. We want to estimate the expected value of $x$ but with respect to the target distribution $\pi$. Because $x$ is drawn from $b$, we cannot simply use the sample average to compute the expectation under $\pi$. This sample average will give us the expected value under $b$ instead.

**Sample**: $x \approx b$

**Estimate**: $\mathbb{E}_\pi[X]$

Let’s start with the definition of the expected value. We sum over all possible outcomes $x$ multiplied by its probability according to $\pi$. Next, we can multiply by $b(x)$ divided by $b(x)$ because this term is equal to 1. $b(x)$ is the probability of observed outcome $x$ under $b$. Shifting around the numerator, we end up with a ratio $\frac{\pi(x)}{b(x)}$. This ratio is very important to us and is called the importance sampling ratio $\rho(x)$.
$$
\mathbb{E}_\pi[X] \doteq \sum_{x \in X} \pi(x) = \sum_{x \in X} \pi(x) \frac{b(x)}{b(x)}
$$

$$
\mathbb{E}_\pi[X] = \sum_{x \in X} x \frac{\pi(x)}{b(x)} b(x) = \sum_{x \in X} x \rho(x) b(x)
$$

If we treat $x \ast \rho(x)$ as a new random variable, times the probability of observing $b(x)$, we can then rewrite this sum as an expectation under $b$. Notice that our expectation is now under $b$ instead of being under $\pi$.

$$
\mathbb{E}_\pi[X] = \sum_{x \in X} x \rho(x) b(x) = \mathbb{E}_b \left[ X \rho(x) \right]
$$

We know how to use importance sampling to correct the expectation, but how do we use it to estimate the expectation from data? It’s actually very simple. We just need to compute a weighted sample average with the importance sampling ratio as the weightings. Note that these samples $X_i$ are drawn from $b$, not $\pi$.

Recall: $\mathbb{E}[X] \approx \frac{1}{n} \sum_{i=n}^n x_i$

$$
\mathbb{E}_b \left[ \rho(x) \right] = \sum_{x \in X} x \rho(x) b(x) \approx \frac{1}{n} \sum_{i=n}^n x_i \rho(x_i)
$$

$x_i \sim b$

$$
\mathbb{E}_\pi[X] \approx \frac{1}{n} \sum_{i=n}^n x_i \rho(x_i)
$$

## Off-Policy Monte Carlo Prediction 
If we simply average the returns, we saw from state s under behavior b, we will not get the right answer. We have to correct each return in the average. This is just what important sampling is for. All we have to do is figure out the value of ρ for each of the sampled returns.
Given the following equation:

$$
V_{\pi}(s) = \mathbb{E}_{\pi} [G_t | S_t = s] \approx \text{average}\ (\rho_0 \text{Returns}[0], \rho_1 \text{Returns}[1], \rho_2 \text{Returns}[2])
$$

where $\rho$ is the probability of the trajectory under $\pi$ divided by the probability of the trajectory under $b$. This $\rho$ corrects the distribution over entire trajectories, and so corrects the distribution over returns.

$$
\rho = \frac{\mathbb{P}(\text{trajectory under } \pi)}{\mathbb{P}(\text{trajectory under } b)}
$$

$$
V_{\pi}(s) = \mathbb{E}_b [\rho G_t | S_t = s]
$$

Let's consider the probability distribution over trajectories. We read this probability as: given that the agent is in some state as $t$, what is the probability that it takes action $A_t$, then ends up in state $S_{t+1}$, then it takes action $A_{t+1}$ and ends up in $S_{t+2}$, and so on, until termination at time $T$. All of the actions are sampled according to behavior $b$.

$$
P(A_t, S_{t+1}, A_{t+1}, \ldots, S_T | S_t, A_t:T)
$$

Because of the Markov property, we can break this probability distribution into smaller chunks. The first chunk is the probability that the agents take action $A_t$ in state $S_t$ times the probability that the environment transitions into state $S_{t+1}$.

$$
b(A_t | S_t) p(S_{t+1} | S_t, A_t) b(A_{t+1} | S_{t+1}) p(S_{t+2} | S_{t+1}, A_{t+1}) \ldots p(S_T | S_{T-1}, A_{T-1})
$$

We can rewrite this list of product probabilities using the product notation. Now, we've defined the probability of a trajectory under $b$. Remember where we are going. We would like to define $\rho$ using the probability of the trajectory under $\pi$ and the probability of the trajectory under $b$.

$$
\prod_{k=t}^{T-1} b(A_k | S_k) p(S_{k+1} | S_k, A_k)
$$

We would like to define $\rho$ using the probability of the trajectory under $\pi$ and the probability of the trajectory under $b$. Let's plug these probabilities into our definition of $\rho$. As we saw before, we can take these probabilities and multiply them by the importance sampling ratio.

$$
\mathbb{P}(\text{Trajectory under } b) = \prod_{k=t}^{T-1} b(A_k | S_k) p(S_{k+1} | S_k, A_k)
$$
The equations describe the process for calculating the importance sampling ratio $\rho_{t:T-1}$:

$$
\rho_{t:T-1} = \frac{\mathbb{P}(\text{trajectory under } \pi)}{\mathbb{P}(\text{trajectory under } b)}
$$

$$
\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k | S_k) p(S_{k+1} | S_k, A_k)}{b(A_k | S_k) p(S_{k+1} | S_k, A_k)}
$$

Simplifying, this becomes:

$$
\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k | S_k)}{b(A_k | S_k)}
$$

The agent observes many returns, each according to the behavior policy $b$. We can estimate $V_{\pi}$ using these returns by correcting each return with $\rho$:

$$
\mathbb{E}_b [\rho_{t:T-1} G_t | S_t = s] = V_{\pi}(s)
$$


**On-policy Monte Carlo Algorithm**:
- **Input**: A policy $\pi$ to be evaluated.
- **Initialize**:
  - $V(s) \in \mathbb{R}$, arbitrarily, for all $s \in S$
  - $Returns(s)$ – an empty list, for all $s \in S$
- **Loop forever (for each episode)**:
  - Generate an episode following $\pi$: $S_0, A_0, R_1, S_1, \ldots, S_{T-1}, A_{T-1}, R_T$
  - $G = 0$
  - **Loop for each step of the episode,** $t = T-1, T-2, \ldots, 0$
    - $G = \gamma G + R_{t+1}$
    - Append $G$ to $Returns(S_t)$
    - $V(S_t) = \text{average}(Returns(S_t))$

Now the algorithm for off-policy Monte Carlo. The return is corrected by a new term $W$, which is the accumulated product of important sampling ratios $\prod_{t=1}^{T-1} \frac{\pi(A_t | S_t)}{b(A_t | S_t)}$ on each time step of the episode.

**Off-policy every-visit MC prediction, for estimating $V \approx V_{\pi}$**:
- **Input**: A policy $\pi$ to be evaluated.
- **Initialize**:
  - $V(s) \in \mathbb{R}$, arbitrarily, for all $s \in S$
  - $Returns(s)$ – an empty list, for all $s \in S$
- **Loop forever (for each episode)**:
  - Generate an episode following $b$: $S_0, A_0, R_1, S_1, \ldots, S_{T-1}, A_{T-1}, R_T$
  - $G = 0$, $W = 1$
  - **Loop for each step of the episode,** $t = T-1, T-2, \ldots, 0$
    - $G = \gamma G + R_{t+1}$
    - Append $G$ to $Returns(S_t)$
    - $V(S_t) = \text{average}(Returns(S_t))$
    - $W = W \frac{\pi(A_t | S_t)}{b(A_t | S_t)}$

We can compute $\rho$ from $t$ to $T-1$ incrementally. To see why, let's write out the product at each time step. Recall that the Monte Carlo algorithm loops over time steps backwards. So on the first step of the algorithm, $W$ is set to $\rho$ on the last time step.

On the next time step, $W$ is the second last $\pi$ times the last $\rho$, and so on. Each time step adds one additional term to the product and reuses all previous terms. We can compute this recursively without having to store all past values of $\rho$.

# Temporal Difference Learning
TD is better than MC in 1) updates at every time when feedback from environment and no need to wait till end of episode 2) uses bootstrapping instead of actual rewards at end of episode.
To be remember, the term "bootstrap" refers to the method of updating an estimate based on other estimates rather than waiting for a final outcome. 
## What is Temporal Difference (TD) learning?

Temporal-Difference Learning combines Monte Carlo and dynamic programming ideas. Like Monte Carlo methods, TD methods learn directly from experience without a model of the environment. Unlike dynamic programming, TD updates estimates based in part on other estimates without waiting for a final outcome (bootstrapping).

## TD Prediction
- Every-visit Monte Carlo:
  $$
  V(S_t) \leftarrow V(S_t) + \alpha (G_t - V(S_t))
  $$
  Must wait for episode to end to get $G_t$.
- TD Method:
  $$
  V(S_t) \leftarrow V(S_t) + \alpha (R_{t+1} + \gamma V(S_{t+1}) - V(S_t))
  $$
  TD methods can determine the increment to $V(S_t)$ after each step. This is a special case of TD($\lambda$) called TD(0) or one-step TD.

####  TD(0) Algorithm Pseudocode
- **Tabular TD(0) for estimating $\pi$**:
  - Input: the policy $\pi$ to be evaluated
  - Algorithm parameter: step size $\alpha \in (0,1)$
  - Initialize $V(s)$, for all $s \in S^+$, arbitrarily except that $V(\text{terminal}) = 0$
  - Loop for each episode:
    - Initialize $S$
    - Loop for each step of episode:
      - $A \leftarrow$ action given by $\pi$ for $S$
      - Take action $A$, observe $R, S'$
      - $V(S) \leftarrow V(S) + \alpha (R + \gamma V(S') - V(S))$
      - $S \leftarrow S'$
    - until $S$ is terminal

This is a bootstrapping method, where:
$$
V(S_t) = E[G_t | S_t = s] = E[R_{t+1} + \gamma G_{t+1} | S_t = s] = E[R_{t+1} + \gamma V(S_{t+1}) | S_t = s]
$$

TD Error:
$$
\delta = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
$$

From:
$$
V(S) = V(S) + \alpha (R + V(S') - V(S))
$$
DP vs MC vs TD: 


![[Pasted image 20240909214511.png]]


## TD vs Monte Carlo Methods
### Example: Random Walk

- Value of each state is the probability of terminating on the right if you are at that state.
- Reward is 1 for terminating on the right, else 0.
- Always start at C.
- Random chance of moving left/right.
- Policy π(s) = 1/2 ∀ s
- Discount factor γ = 1
### State Values
| State | A     | B     | C   | D     | E     |
| ----- | ----- | ----- | --- | ----- | ----- |
| Value | 0.167 | 0.333 | 0.5 | 0.667 | 0.833 |
### After a Few Episodes

#### Target/Exact Values
| State | A     | B     | C   | D     | E     |
| ----- | ----- | ----- | --- | ----- | ----- |
| Value | 0.167 | 0.333 | 0.5 | 0.667 | 0.833 |
#### Updates using TD Learning
| State | A     | B     | C     | D     | E     |
|-------|-------|-------|-------|-------|-------|
| Value | 0.008 | 0.395 | 0.659 | 0.688 | 0.914 |
#### Updates using Monte Carlo
| State | A     | B     | C     | D     | E     |
|-------|-------|-------|-------|-------|-------|
| Value | 0.063 | 0.578 | 0.793 | 0.922 | 0.922 |
**Observation: TD is doing better.**

## TD Performance Over Number of Episodes

- **Observation:** The estimated values would approach closer to the true values with a smaller learning rate α.

 ![[2024-09-22_18-27.png]]
### Graph Description
- The graph shows estimated values for different numbers of episodes: 0, 1, 10, and 100.
- True values are shown as a black baseline.


### RMS Error Averaged Over States

![[2024-09-22_18-28.png]]

- **MC** stands for Monte Carlo methods.
- **TD** stands for Temporal Difference methods.
- Various learning rates (α) are used, ranging from 0.01 to 0.15.

### Notes
- TD converges to a lower final error for this problem compared to Monte Carlo methods.
- TD is suitable for continuing tasks as they update incrementally via bootstrapping, unlike Monte Carlo methods.
- Temporal Difference learning adapts more quickly and is more stable under a variety of conditions, making it effective for problems where the environment changes or continues indefinitely.


# Temporal Difference Learning For Control 
## Sarsa : on-policy TD control 
Now we want to use TD for control task 
- Still follows the pattern of generalized policy iteration (GPI), similar to Monte Carlo (MC) methods. 
- Uses an on-policy TD control method.

**Method**:  Estimate action-value functions now (like with MC) instead of state-action functions like previously.  

Update : 
$$
Q(S_t, A_t) = Q(S_t, A_t) + α[R_{t+1} + γQ(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
$$

- **Update is done after every transition** from a non terminal state $S_t$.
- This rule uses a quintuplet of events that make up a transition: $(S_t,A_t,R_{t+1},S_{t+1},A_{t+1})$.

##### Sarsa (on-policy TD control) algorithm 
**Algorithm parameters:** 
- step size $\alpha \in [0, 1]$
- small $\epsilon > 0$
**Initialize** $Q(s, a)$, for all $s \in S^+$, $a \in A(s)$, arbitrarily except that $Q(\text{terminal}, \cdot) = 0$
**Loop for each episode:**
- **Initialize** $S$
- **Choose** $A$ from $S$ using policy derived from $Q$ (e.g., $\epsilon$-greedy)
  **Loop for each step of episode:**
  - **Take action** $A$, **observe** $R, S'$
  - **Choose** $A'$ from $S'$ using policy derived from $Q$ (e.g., $\epsilon$-greedy)
  - **Update** $Q$ value:
    $Q(S, A) \leftarrow Q(S, A) + \alpha [R + \gamma Q(S', A') - Q(S, A)]$
  - **Assign** $S \leftarrow S'; A \leftarrow A'$;
  **until** $S$ is terminal


When updating your Q-value for state s and action a, you want to account for the fact that, in the next state s′, you will always pick the best possible action to maximize your future reward. The term $max_a Q(s', a)$ represents the **value of the best action** in state s′—the action that leads to the highest future reward. It's not the case for SARSA, you will pick an action a from s' according to a policy from Q. 

## Q-Learning: Off-Policy TD Control

$$Q(s_t, A_t) = Q(s_t, A_t) + \alpha [R_{t+1} + \gamma \max_{A} Q(s_{t+1}, a) - Q(s_t, A_t)]$$
Q-learning directly approximates the optimal action-value function $Q^*$, independent of the policy followed. It differs from Sarsa because it doesn't use $A_{t+1}$ in the update rule.

##### Q-Learning algorithm 
**Q-learning (off-policy TD control) for estimating $\pi \approx \pi_*$**
**Algorithm parameters:**
- step size $\alpha \in [0, 1]$
- small $\epsilon > 0$
**Initialize** $Q(s, a)$, for all $s \in S^+$, $a \in A(s)$, arbitrarily except that $Q(\text{terminal}, \cdot) = 0$
**Loop for each episode:**
- **Initialize** $S$
- **Choose** $A$ from $S$ using policy derived from $Q$ (e.g., $\epsilon$-greedy)
  **Loop for each step of episode:**
	  - **Take action** $A$, **observe** $R, S'$
	  - **Choose** $A'$ from $S'$ using policy derived from $Q$ (e.g., $\epsilon$-greedy)
	  - **Update** $Q$ value:
	    $Q(S, A) \leftarrow Q(S, A) + \alpha [R + \gamma \max_{A'} Q(S', A') - Q(S, A)]$
	  - **Assign** $S \leftarrow S'; A \leftarrow A'$;
  **until** $S$ is terminal

##### Why Doesn't Q-Learning Use $A_{t+1}$?

**Sarsa:**
$Q(S,A) \leftarrow Q(S,A) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S,A)]$
- Represents the Bellman equation for action values under the policy being followed.

**Q-learning:**
$Q(S,A) \leftarrow Q(S,A) + \alpha [R_{t+1} + \gamma \max_{A'} Q(S_{t+1}, A') - Q(S,A)]$
- Represents the Bellman optimality equation for action values.

*Remember that the Bellman Optimality Equation allows Q-Learning to directly learn $Q^*$ without switching between policy improvement and evaluation.*


## Why Q-learning isn't using important sampling ? 
Q-learning is off-policy because it updates its action-value estimates using the maximum reward that is attainable from the next state under any policy, not necessarily the policy being followed to generate the data. This decouples the policy used to generate the behavior (behavior policy) from the policy being evaluated and improved (estimation policy).
Q-learning avoids the need for importance sampling by directly approximating the optimal policy. Since it always considers the maximum action value for updates regardless of the actual action taken, it inherently accounts for the best possible future actions.
## Expected Sarsa

Expected Sarsa is similar to Sarsa but uses a deterministic algorithm instead of an expectation-based one. The update equation is as follows:

$$Q(s_t, A_t) = Q(s_t, A_t) + \alpha \left[ R_{t+1} + \gamma \mathbb{E} \left[ Q(S_{t+1}, A_{t+1}) \mid S_{t+1} \right] - Q(s_t, A_t) \right]$$

The expectation is computed over all possible actions from the next state, weighted by the policy's probability of selecting each action:

$$Q(s_t, A_t) = Q(s_t, A_t) + \alpha \left[ R_{t+1} + \gamma \sum_a \pi(a \mid S_{t+1}) Q(S_{t+1}, a) - Q(s_t, A_t) \right]$$

Where:
- $\alpha$ is the step size (learning rate),
- $\gamma$ is the discount factor,
- $\mathbb{E}$ denotes the expected value, and
- $\pi(a \mid S_{t+1})$ is the probability of taking action $a$ in state $S_{t+1}$ under the current policy.

**Key Points:**
- Expected Sarsa is more computationally intensive than Sarsa because it computes an expectation over all possible next actions.
- However, it reduces the variance introduced by the random selection of $A_{t+1}$ in the standard Sarsa algorithm, potentially leading to more stable and reliable learning.

