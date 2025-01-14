# 6.

If we only consider the original constraint (i.e. the constraints for example $1$ to $N$), the lagrange function with lagrange multipliers $\alpha_n$ and $\beta_n$ is:

$$
\mathcal{L}(b, \mathbf{w}, \mathbf{\xi}, \mathbf{\alpha}, \mathbf{\beta}) = \frac{1}{2} \mathbf{w}^T\mathbf{w} + C \sum_{n=1}^N \xi_n + \sum_{n=1}^N \alpha_n \left(1 - \xi_n - y_n(b + \mathbf{w}^T \mathbf{\Phi(x_n)}) \right) + \sum_{n=1}^N \beta_n (-\xi_n)
$$

For the anchor pseudo-example, we have the constraint:

$$
y_0(\mathbf{w}^T\mathbf{\Phi(x_0)} + b) \ge 1 
$$

Since we have $\mathbf{x}_0 = \mathbf{0}$ and $y_0 = -1$, the constraint becomes:

$$
-b \ge 1 \Rightarrow b \le - 1
$$

Convert into canonical form:

$$
b + 1 \le 0
$$

So we can add the term $\gamma_0(b + 1)$, where $\gamma_0$ is the corresponding lagrange multiplier.

Thus we have the new lagrange function:

$$
\mathcal{L}(b, \mathbf{w}, \mathbf{\xi}, \mathbf{\alpha}, \mathbf{\beta}, \gamma_0) = \frac{1}{2} \mathbf{w}^T\mathbf{w} + C \sum_{n=1}^N \xi_n + \sum_{n=1}^N \alpha_n \left(1 - \xi_n - y_n(b + \mathbf{w}^T \mathbf{\Phi(x_n)}) \right) + \sum_{n=1}^N \beta_n (-\xi_n) + \gamma_0(b + 1)
$$

For the lagrange dual, we need to solve:

$$
\max_{\alpha, \beta, \gamma_0 \ge 0} \min_{b, \mathbf{w}, \mathbf{\xi}} \mathcal{L}(b, \mathbf{w}, \mathbf{\xi}, \mathbf{\alpha}, \mathbf{\beta}, \gamma_0)
$$

Taking the partial derivative of each variable, we get:

$$
\frac{\partial \mathcal{L}}{\partial b} = 0 \Rightarrow -\sum_{n=1}^N \alpha_n y_n + \gamma_0 = 0
\tag{1}
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 0 \Rightarrow \mathbf{w} = \sum_{n=1}^N \alpha_n y_n \mathbf{\Phi(x_n)} 
\tag{2}
$$

$$
\frac{\partial \mathcal{L}}{\partial \xi_n} = 0 \Rightarrow C - \alpha_n - \beta_n = 0 
\tag{3}
$$

Since $\alpha_n \ge 0$ and $\beta_n \ge 0$, we have $0 \le \alpha_n \le C$.

From equation $(2)$, we can substitute $\mathbf{w}$ into the Lagrangian and obtain the dual:

$$
\frac{1}{2} \Bigg\| \sum_{n=1}^N \alpha_n y_n \Phi(\mathbf{x}_n) \Bigg\|^2+ C \sum_{n=1}^N \xi_n + \sum_{n=1}^N \alpha_n \left(1 - \xi_n - y_n(b + \mathbf{w}^T \mathbf{\Phi(x_n)}) \right) + \sum_{n=1}^N \beta_n (-\xi_n) + \gamma_0(b + 1)
$$

Converting $||\mathbf{w}||^2$ to the proper form, can write:

$$
||\mathbf{w}||^2 = \sum_{n=1}^N \sum_{m=1}^N \alpha_n \alpha_m y_n y_m \mathbf{\Phi(x_n)}^T \mathbf{\Phi(x_m)} 
$$

Plug in back to the dual, we get:

$$
\frac{1}{2}\sum_{n=1}^N \sum_{m=1}^N \alpha_n \alpha_m y_n y_m K(x_n, x_m) + C \sum_{n=1}^N \xi_n + \sum_{n=1}^N \alpha_n \left(1 - \xi_n - y_n(b + \mathbf{w}^T \mathbf{\Phi(x_n)}) \right) + \sum_{n=1}^N \beta_n (-\xi_n) + \gamma_0(b + 1)
$$

Expand the terms and we get:

$$
\begin{aligned}
&\frac{1}{2}\sum_{n=1}^N \sum_{m=1}^N \alpha_n \alpha_m y_n y_m K(x_n, x_m) + C \sum_{n=1}^N \xi_n \\
& + \sum_{n=1}^N \alpha_n - \sum_{n=1}^N \alpha_n \xi_n - \sum_{n=1}^N \alpha_n y_n b - \sum_{n=1}^N \alpha_n y_n \mathbf{w}^T \mathbf{\Phi(x_n)} \\
&- \sum_{n=1}^N \beta_n \xi_n + \gamma_0 b + \gamma_0
\end{aligned}
$$

From equation $(1)$, we have:

$$
- \sum_{n=1}^N \alpha_n y_n + \gamma_0 = 0 \Rightarrow (- \sum_{n=1}^N \alpha_n y_n + \gamma_0)b = 0
$$

And from equation $(3)$, we have:

$$
(C - \alpha_n - \beta_n)\sum_{n=1}^N \xi_n= 0 
$$

The Largrangian is simplified to:

$$
\frac{1}{2}\sum_{n=1}^N \sum_{m=1}^N \alpha_n \alpha_m y_n y_m K(x_n, x_m)   + \sum_{n=1}^N \alpha_n  - \sum_{n=1}^N \alpha_n y_n \mathbf{w}^T \mathbf{\Phi(x_n)}  + \gamma_0
\tag{4}
$$

Observe that the term that contain $\mathbf{w}$ can also be expanded by $(2)$ as:

$$
\begin{split}
- \sum_{n=1}^N \alpha_n y_n \mathbf{w}^T \mathbf{\Phi(x_n)}  
&= - \sum_{n=1}^N \alpha_n y_n \left( \sum_{m=1}^N \alpha_m y_m \mathbf{\Phi(x_m)} \right)^T \mathbf{\Phi(x_n)} \\
&= - \sum_{n=1}^N \sum_{m=1}^N \alpha_n \alpha_m y_n y_m \mathbf{\Phi(x_m)}^T \mathbf{\Phi(x_n)}  \\
&= - \sum_{n=1}^N \sum_{m=1}^N \alpha_n \alpha_m y_n y_m K(x_n, x_m)
\end{split}
$$

Again, plug into the previous simplified Lagrangian $(4)$, we get:

$$
-\frac{1}{2}\sum_{n=1}^N \sum_{m=1}^N \alpha_n \alpha_m y_n y_m K(x_n, x_m) + \sum_{n=1}^N \alpha_n  + \gamma_0 
$$


From the problem description, we knew that $y_n = +1 \quad \forall n, \ n \ne 0$, so we have:

$$
-\frac{1}{2}\sum_{n=1}^N \sum_{m=1}^N \alpha_n \alpha_m K(x_n, x_m) + \sum_{n=1}^N \alpha_n  + \gamma_0 
$$

In order to use the QP solver, we first need to convert the above maximization problem into a minimization problem:

$$
\min_{\mathbf{\alpha}, \gamma_0 \ge 0} \frac{1}{2}\sum_{n=1}^N \sum_{m=1}^N \alpha_n \alpha_m K(x_n, x_m) - \sum_{n=1}^N \alpha_n  - \gamma_0 
\tag{*}
$$

With constraints from $(1), (3)$:

$$
\sum_{n=1}^N \alpha_n - \gamma_0 = 0 
\tag{5}
$$

$$
C - \alpha_n - \beta_n = 0 
\tag{6}
$$

To convert into the QP form, we define:

$$
\begin{split}
\mathbf{\alpha} &= [\alpha_1 \ \alpha_2 \ \cdots \ \alpha_N]^T \\
\mathbf{u} &= [\gamma_0 \ \alpha_1 \ \alpha_2 \ \cdots \ \alpha_N]^T \\
\mathbf{Q} &= [K(x_n, x_m)]_{N \times N} \quad \text{(Gram matrix)}
\end{split}
$$

Then the original problem $(*)$ can be written as:

$$
\min_{\mathbf{x} \ge 0} \frac{1}{2} \mathbf{u}^T \mathbf{Q} \mathbf{u} - \mathbf{1}^T \mathbf{u}
$$

So:

$$
Q =  [K(x_n, x_m)]_{N \times N} \quad \text{and} \quad \mathbf{p} = \mathbf{-1}
$$

subject to:

$$
\begin{split}
\sum_{n=1}^N \alpha_n = \gamma_0 \qquad \text{by } (5) \\
\alpha_n = C - \beta_n \qquad \text{by } (6)
\end{split}
$$

Thus for each row in $A$, 

$$
\mathbf{a}_n^T = y_n [1 \ \mathbf{\Phi(x_n)^T}] = 
\begin{cases}
[1 \ \mathbf{\Phi(x_n)^T}] &  \forall n \ne 0 \\
[1 \ \mathbf{0^T}] &  n = 0
\end{cases}
$$

Finally, we have each element in $\mathbf{c}$ as $c_n = 0$
