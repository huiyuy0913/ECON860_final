# ECON860_final
## Part (c)
The code for this part is [c_run_factor_analysis.py](https://github.com/huiyuy0913/ECON860_final/blob/main/c_run_factor_analysis.py). 

At the beginning. I check if there are numbers greater than 5 in the first 40 columns and find
there is no such numbers. Then, I delete all rows contain 0 in the first 40 columns since it will make it easy to analyse the relationship
between personality traits and math ability in the following parts.

Based on eigenvalues, I find it better to choose 7 as the number of personality traits. Then, I do the factor analysis when the number
of factors equals 6, 7, and 8. I find 6 would be the best choice as the number of personality traits since no question goes into the seventh 
factor when I increase factors to 7, nor does the eighth factor. Next, I rerun the factor analysis when the number of personality traits is 6
and rotation is varimax. Finally, I use loadings from the last model to multiply with the data containing people's answers for 40 questions and then generate the [results.csv](https://github.com/huiyuy0913/ECON860_final/blob/main/results.csv).

## Part (d)
The code for this part is [d_unsupervised_learning.py](https://github.com/huiyuy0913/ECON860_final/blob/main/d_unsupervised_learning.py).

I use three models in this part: the KMeans clustering model, the KMedoids clustering model, and the Gaussian mixture model. I calculate silhouette scores and draw the silhouette score graphs for each.

## Part (e)
I think the Gaussian mixture model will give me a better result.

For KMedoids, I didn't choose it since it has an empty cluster when the number of cluster groups is greater than 3. This means the silhouette scores are not accurate for these groups. Besides, its highest silhouette score is lower than the highest silhouette scores of the other two methods. For KMeans, I didn't choose it as its highest silhouette score is lower than the highest silhouette score of the Gaussian mixture model. Besides, since the data are in high dimensions, there are chances for them to have a strange shape which KMeans couldn't solve. It is reasonable to assume the personality traits have single spike distribution as the extreme values are usually rare. So, we can assume the normal distribution and use the Gaussian mixture model here.

## Part (f)
The code for this part is [f_supervised_learning.py](https://github.com/huiyuy0913/ECON860_final/blob/main/f_supervised_learning.py).

I use linear regression and logistic regression in this part. Besides, I also calculate the R2 score for each.

## Part (g)
I think the linear regression will give me a better result. The logistic regression failed to converge even if I increased the maximum iteration to 10,000, so I couldn't get the accurate R2 score for the logistic regression and couldn't make a comparison with the linear regression's R2 score. By the way, the linear regression's R2 score is still greater than the logistic regression's R2 score, even after I increased the maximum iteration to 10,000. It partly reflects that linear regression gives us a better result.

The other reason is that the measurement of math ability is an ordered categorical variable, meaning linear regression is more suitable than logistic regression.


## Part (h)
The code for this part is [h_i_choose_questions.py](https://github.com/huiyuy0913/ECON860_final/blob/main/h_i_choose_questions.py).

The main idea I have to solve this question is that finding questions have a larger impact on math ability. 

First, I make a dataframe "df_total" which contains absolute values of six factor loadings (correspond to six personality traits) in part (c) (columns 0, 1, 2, 3, 4, 5), the corresponding question numbers (columns index_0, index_1, index_2, index_3, index_4, index_5), and the ranking of the absolute values of factor loadings by column (columns rank_0, rank_1, rank_2, rank_3, rank_4, rank_5). 
Then, I calculate the coefficients of the linear regression in part (f) and name it as "slope". Since each personality trait corresponds to one coefficent, I multiply them correspondingly and generate a new dataframe "df_new_total" with columns 0 to 5 (the new interections of personality traits and coefficients), index_0 to index_1 (same as df_total), and rank_0 to rank_5 (ranking of columns 0 to 5 accordingly).
Finally, I choose the top 20 questions that have largest interections among columns 0, 1, 2, 3, 4, 5.
So, the questions I will choose are questions 25, 23, 11, 33, 27, 38, 14, 16, 15, 7, 0, 21, 37, 10, 31, 5, 22, 6, 2, 9. Of course, you should plus 1 to get the correct question numbers. 




## Part (i)
The code for this part is [h_i_choose_questions.py](https://github.com/huiyuy0913/ECON860_final/blob/main/h_i_choose_questions.py).

Suppose I want to assemble a team that requires a variety of many different personality traits. I will choose the top 4 questions that impact the 6 personality traits the most. So, I will observe dataframe "df_total" this time. 

Since I can only choose 20 questions and couldn't divide 20 questions evenly to 6 personality traits, I will select the top 3 questions that have the largest impact on the 6 personality traits and choose the other two top 4th questions among the 6 personality traits. For the other two, I will choose the top 4th questions for the third and the sixth personality traits as the questions' loadings (or impact to corresponding personality traits) are relatively lower.

Finally, I will choose questions 33, 38, 14, 3, 4, 1, 25, 23, 11, 0, 37, 31, 5, 22, 39, 20, 28, 29, 27, 24. Of course, you should plus 1 to get the correct question numbers. 


## Bonus Question
The code for this part is [bonus_question.py](https://github.com/huiyuy0913/ECON860_final/blob/main/bonus_question.py). 

I run two linear regressions in this code. They have the same dependent variable but different independent variables. One treats 6 personality traits as independent variables; the other one treats 40 questions as independent variables. I calculate the R2 score for each of them. I find the R2 score for the model with 40 questions is systematically higher than the model with 6 personality traits. 

So, it is a good idea to use the questionnaire answers themselves to predict the math ability of the individuals. One reason is that the 40 questions include more details about the math ability which the 6 personality traits can omit. 

But, most of the time, the 6 personality traits will give better categorizations than directly looking at every question.
