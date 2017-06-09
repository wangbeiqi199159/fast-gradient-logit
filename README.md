# Fast Gradient Ridge Logistic Regression

Ridge logistic regression is one of common regularized regressions, which is also called l2 logistic regression. The problem writes as:

[![Screen Shot 2017-06-03 at 12.27.32 PM.png](https://s12.postimg.org/3vvd3ert9/Screen_Shot_2017-06-03_at_12.27.32_PM.png)](https://postimg.org/image/pv1rqm8nd/)

In this file, I use `fast gradienct descent` to solve the optimization problem. Algorithm of fast gradient descent can be described as follows:

[![Screen Shot 2017-06-04 at 12.56.11 PM.png](https://s1.postimg.org/haenfii4v/Screen_Shot_2017-06-04_at_12.56.11_PM.png)](https://postimg.org/image/rx8gkxqa3/)

## Software dependencies and license information
#### Programming language: 

- Python version 3.0 and above 

#### Python packages needed:

- pandas
- NumPy
- sklearn
- scipy.linalg
- matplotlib.pyplot

#### License Information:
The MIT License is a permissive free software license originating at the Massachusetts Institute of Technology (MIT). As a permissive license, it puts only very limited restriction on reuse and has therefore an excellent license compatibility. For detailed description of the contents of license please refer to the file [License](https://github.com/wangbeiqi199159/analyze-of-seattle-airbnb-hosts/blob/master/LICENSE).

## Demo

There are three demo files:

`demo on simulated dataset.ipynb` allows a user to launch the method on a simple simulated dataset,
visualize the training process, and print the performance

`demo on real world dataset.ipynb` allows a user to launch the method on a real-world dataset of your
choice, visualize the training process, and print the performance

`Experimental Comparison to Sklearn.ipynb` allows a user to compare result of this package with scikit-learn.

