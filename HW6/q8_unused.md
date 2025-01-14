
# 8. (Unused)

Using the Taylor expansion of $\cos(x)$, which is:

$$
\cos(x) = \sum_{n=0}^\infty (-1)^n \frac{x^{2n}}{(2n)!}
$$

We set $x = 2 \cos(x_i - x_j)$, then we'll get:

$$
\mathrm{exp}(2 \cos(x_i - x_j)) = \sum_{n=0}^\infty (-1)^n \frac{(2 \cos(x_i - x_j))^{2n}}{(2n)!}
$$

To further simplify the expression, we use the property:

$$
\cos(x_i - x_j) = \cos(x_i) \cos(x_j) + \sin(x_i) \sin(x_j)
$$

Plugging in the previous expression, we have:

$$
\begin{split}
\mathrm{exp}(2 \cos(x_i - x_j)) 
&= \sum_{n=0}^\infty (-1)^n \frac{2^{2n}\left[\cos(x_i) \cos(x_j) + \sin(x_i) \sin(x_j)\right]^{2n}}{(2n)!} \\
&= \sum_{n=0}^\infty (-1)^n \frac{\left\{\left[2 \cos(x_i) \cos(x_j) + 2 \sin(x_i) \sin(x_j)\right]^2\right\}^n}{(2n)!} \\
\end{split}
$$

To expand the inner term $\left[2 \cos(x_i) \cos(x_j) + 2 \sin(x_i) \sin(x_j)\right]^2$, we have:

$$
\begin{split}
\left[2 \cos(x_i) \cos(x_j) + 2 \sin(x_i) \sin(x_j)\right]^2
&= 4 \cos^2(x_i) \cos^2(x_j) + 8 \cos(x_i) \cos(x_j) \sin(x_i) \sin(x_j) + 4 \sin^2(x_i) \sin^2(x_j) \\
&= 4 \left(\frac{1+ \cos(2x_i)}{2} \frac{1 + \cos(2x_j)}{2}\right) +4 \left(\frac{1 - \cos(2x_i)}{2} \frac{1 - \cos(2x_j)}{2}\right) + 8 \cos(x_i) \cos(x_j) \sin(x_i) \sin(x_j) \\
&= 2 \left(1 + \cos(2x_i)\right) \left(1 + \cos(2x_j)\right) + 2 \left(1 - \cos(2x_i)\right) \left(1 - \cos(2x_j)\right) + 8 \cos(x_i) \cos(x_j) \sin(x_i) \sin(x_j) \\
&= \left[2 + 2 \cos(2x_i) + 2 \cos(2x_j) + 2 \cos(2x_i) \cos(2x_j)\right] + \left[2 - 2 \cos(2x_i) - 2 \cos(2x_j) + 2 \cos(2x_i) \cos(2x_j)\right] + 8 \cos(x_i) \cos(x_j) \sin(x_i) \sin(x_j) \\
&= 4 + 4 \cos(2x_i) \cos(2x_j) + 8 \cos(x_i) \cos(x_j) \sin(x_i) \sin(x_j) \\
\end{split}
$$

Then by the property:

$$
\sin(2x) = 2 \sin(x) \cos(x)
$$

We have:

$$
\begin{split}
8 \cos(x_i) \cos(x_j) \sin(x_i) \sin(x_j) 
&= 4 \left[2\sin(x_i) \cos(x_i) \times 2 \sin(x_j) \cos(x_j) \right]\\
&= 4 \left[\sin(2x_i) \times \sin(2x_j) \right]\\
\end{split}
$$

Substitute back to the previous expression, we have:

$$
\begin{split}
\left[2 \cos(x_i) \cos(x_j) + 2 \sin(x_i) \sin(x_j)\right]^2 = 4 + 4 \cos(2x_i) \cos(2x_j) + 4 \sin(2x_i) \sin(2x_j)
\end{split}
$$

Similarly, for the terms $4 \cos(2x_i) \cos(2x_j) + 4 \sin(2x_i) \sin(2x_j)$

Now we need to show the following expression is non-negative:

$$
\sum_{n=0}^\infty (-1)^n \frac{\left[4 + 4 \cos(2x_i) \cos(2x_j) + 4 \sin(2x_i) \sin(2x_j)\right]^n}{(2n)!} 
$$

Using the same property $\cos(x_i - x_j) = \cos(x_i) \cos(x_j) + \sin(x_i) \sin(x_j)$, we have:

$$
4 \cos(2x_i) \cos(2x_j) + 4 \sin(2x_i) \sin(2x_j) = 4 \cos(2x_i - 2x_j)
$$

And the previous expression becomes:

$$
\begin{split}
\sum_{n=0}^\infty (-1)^n \frac{\left[4 + 4 \cos(2x_i - 2x_j)\right]^n}{(2n)!} 
&= \sum_{n=0}^\infty (-1)^n \frac{\left[4 \left(1 + \cos(2(x_i - x_j))\right)\right]^n}{(2n)!} \\
&= \sum_{n=0}^\infty (-1)^n \frac{4^n \left[1 + \cos(2(x_i - x_j))\right]^n}{(2n)!} \\
\end{split}
$$

Let $t = 1 + \cos(2(x_i - x_j))$, then we have:

$$
\begin{split}
\sum_{n=0}^\infty (-1)^n \frac{4^n t^n}{(2n)!} 
&= \sum_{n=0}^\infty (-1)^n \frac{4^n t^n}{(2n)!} \\
&= \sum_{n=0}^\infty (-1)^n \frac{\left(2\sqrt{t}\right)^{2n}}{(2n)!} \\
&= \cos(2\sqrt{t})
\end{split}
$$


Since $\cos(x) \in [-1, 1]$, we have:

$$
t = 1 + \cos(2(x_i - x_j)) \in [0, 2]
$$

And 


$$
\cos(2\sqrt{t}) \in [\cos(0), \cos(2\sqrt{2})] 
$$
