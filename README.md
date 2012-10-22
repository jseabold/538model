This is a Python script that replicates some features of Nate Silver's 538 Election Forecasting Model. It was constructed from reading the methodology posts on the [old site](http://www.fivethirtyeight.com/2008/03/frequently-asked-questions-last-revised.html) and the new one at the [New York Times](http://fivethirtyeight.blogs.nytimes.com/). This is my interpretation of these posts. Any and all errors are, of course, mine. Furthermore, this code should be considered as more of an example of how to conduct data analysis in Python using pandas and statsmodels rather than a "real" model. You can consider it a starting point for doing more complex analyses with Python rather than a real forecasting model. Or better yet, consider a fun way to learn some Python data tricks.

The polling data is up to date as of 10/2/2012. It is all publicly available from [Real Clear Politics](http://www.realclearpolitics.com/). For some reason Real Clear Politics stopped allowing directory access to their servers, so if you want to update the polling data, you'll have to update the script to walk the links on their site or do it by hand. This should be trivial, I just don't have the time. Historical polling data was obtained from [Electoral Vote](electoral-vote.com).

The pollster reliability/weighting data is also very out of date, and I did not attempt to replicate the calculation of these. I simply used old weights.

Pull requests are welcome. Suggestions and comments on anything from the programming to the modeling are also welcome.

The third-party packages used are 

* [matplotlib](http://matplotlib.org/)
* [numpy](http://numpy.org/)
* [pandas](http://pandas.pydata.org/)
* [scikit-learn](http://scikit-learn.org/stable/)
* [scipy](http://www.scipy.org/)
* [statsmodels](http://statsmodels.sourceforge.net/)

and, of course, [IPython](http://ipython.org/) is used for the notebooks.
