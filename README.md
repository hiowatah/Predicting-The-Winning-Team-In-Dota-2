# Using Machine Learning to Predict Which Dota 2 Team Will Win

# Project Overview

For this project, I entered a Kaggle competition which provided a dataset containing over 181 thousand matches to train my classification models in order to predict whether the 'Radiant Team' will win. However, the kicker was that only data from the first 5 minutes was provided. This makes it very difficult to accurately predict the winner because games typically last anywhere from 20 to 50 minutes. So the data only provides a very brief snapshot.

Dota 2 is a MOBA (Multiplayer Online Battle Arena) in which 2 teams (Radiant and Dire) of 5 players face off until one team destroys the other's base. Based on this, predicting the winner required a binary classifier model as there are only 2 outcomes, win or loss.

# Hypothesis

Based on my experience with League of Legends, another MOBA, I believe that features related to the heros picked, the number of kills the team has as well as the gold lead should be the strongest indicators of which team will win. 

However, also based on my experience with this genre, I know that the first 5 minutes do not tell the entire story of the match itself and it will be tough to predict the outcome regardless of features. Due to this, I do not expect a great accuracy rating, but my final model should be greater than 50% as some games are inherently easier to predict due to one side getting an early lead. This would make it better to predict than a coin-flip. Logic dictates that if I have ~14% of the match data, I should be able to predict with 14% more accuracy than a coin flip. As you will see, my model exceeded this baseline expectation.


# Initial EDA 

Looking at the dataset, I used my intuition to create new features to help make it more apparent to the model the relationship of certain columns with one another. I engineered new features to put them in the perspective of one team to help the model make more linear relationships with the outcome of the game and the different features.

To get a sense of my predictive power before I began modeling, I graphed some of my features which I hypothesized would be the strongest. As with any game, these features include kills and gold earned. They should have a strong correlation with the number of victories as they tend to be a prerequisite of winning a game. Please see below for a graph comparing the number of kills by the 'Radiant' team and the number of victories in my dataset and another graph comparing the amount of gold accumulated by the 'Radiant' team and the number of victories in my dataset. 

<u><b> Radiant Kills and Number of Victories </b></u>

<p align="center">
  <img src="./Mod_3_Project/dota-2-prediction/Images/Radiant Kills.png" title="Radiant Kills">
</p>

<u><b> Radiant Gold and Number of Victories </b></u>

<p align="center">
  <img src="./Mod_3_Project/dota-2-prediction/Images/Radiant Gold.png" title="Radiant Gold">
</p>

These 2 graphs show that the mean gold and kills are just slightly higher for the 'Radiant' wins plot vs the 'Radiant' losses plot. This shows that these features are only slightly predictive for a Radiant win as the difference isn't high enough to say "if the Radiant team has more gold at the 5 minute mark then they will always win". 

<u><b> Correlation Plot </b></u>

My dataset when I prepared my first set of models contained 45 features and the following correlation plot:

<p align="center">
  <img src="./Mod_3_Project/dota-2-prediction/Images/Correlation Plot.png" title="Correlation Plot">
</p>

While it is hard to see from this image what the features are, the important takeaway is that the majority of the features don't correlate with one another besides the features I believe would be the strongest predictors of the outcome of the game, which include which team got the first kill and the new features I engineered.

Another thing to look for before I started modeling was to check for class imbalance in my data set as that would alter my strategy for modeling.

<u><b> Class Imbalance </b></u>

<p align="center">
  <img src="./Mod_3_Project/dota-2-prediction/Images/Class Imbalance.png" title="Class Imbalance">
</p>

Fortunately, my dataset did not have much imbalance and so I created a KNN classifier, Logistic Regression, Decision Trees, and Random Forests. Unfortunately, my inital models were no better than flipping a coin for each game. This was especially true for my tree based models and so I need I would have to do a lot of hyper parameter tuning in order to improve my performances.

# Initial Models

<u><b> Initial KNN Model </b></u>

<p align="center">
  <img src="./Mod_3_Project/dota-2-prediction/Images/Initial KNN.png" title="KNN" width="500" height="500">
</p>

<u><b> Initial Logistic Regression </b></u>

<p align="center">
  <img src="./Mod_3_Project/dota-2-prediction/Images/Initial Logistic.png" title="Logistic Regression" width="500" height="500">
</p>

<u><b> Initial Decision Trees </b></u>

<p align="center">
  <img src="./Mod_3_Project/dota-2-prediction/Images/Initial Decision Trees.png" title="Decision Trees" width="500" height="500">
</p>

<u><b> Initial Random Forest </b></u>

<p align="center">
  <img src="./Mod_3_Project/dota-2-prediction/Images/Initial Random Forest.png" title="Random Forest" width="500" height="500">
</p>

Based on these initial models, I knew I had to do some hyperparameter tuning to allow KNN and the tree based models to work with my dataset more effectively. My tree based models have terrible recall since they predict almost everything as a loss. While the models did improve their classification prowess, they did not perform as well as the logistic regression model based on my results on the Kaggle competition. As a result, I focused on maximizing the results of my logistic regression model.
# Feature Engineering

<u><b> Feature Engineering and Model Improvements </b></u>

Based on the results above, the features as they are do not lend themselves well to the classifier models I used as each model has a very hard time recognizing a pattern between the feature values and the result of the game. 

Having experience playing MOBAs, I know that not all heros are created equally and the character each player picks has their own % chance of winning. Since I could not extract external data for this competition, I used the massive amount of data I already had for these 181 thousand matches and calculated each hero's chance of winning.

<u><b> Initial Dataframe </b></u>

<p align="center">
  <img src="./Mod_3_Project/dota-2-prediction/Images/Initial DataFrame.png" title="Dataframe">
</p>

<u><b> After using pd.Melt </b></u>

<p align="center">
  <img src="./Mod_3_Project/dota-2-prediction/Images/Melt.png" title="Dataframe Melted">
</p>

The tricky part was that the champions were distributed in 10 different columns and so I had to stack them into one column, since the same hero could appear in different columns (depends on which player picked the hero). I used the melt command in pandas to achieve this and then I was able to calculate the number of appearances by each unique hero and whether they won or not.

I also expanded on what I did initially, and converted every column that had descriptions of what the 2 teams did and converted them to be in comparison to one another rather than 2 independent features.

After doing all this feature engineering, I reran my models to check for improvements. KNN and the tree based models did improve a bit with random forest doing a lot better than it did initially. However, my 2 best models by far were logistic regression and XGBoost which I ran based on Logistic Regression.

<p align="center">
  <img src="./Mod_3_Project/dota-2-prediction/Images/Logistic Regression.png" title="Logistic Regression">
</p>

<p align="center">
  <img src="./Mod_3_Project/dota-2-prediction/Images/XGBoost.png" title="XGBoost">
</p>

<u><b> Model Optimizations and Conclusion </b></u>

I used gridsearch to optimize the parameters around my logistic regression model to see how hyper parameter tuning would improve my model. I also took a look at the best parameters for my models and it was all of the ones I engineered.

<p align="center">
  <img src="./Mod_3_Project/dota-2-prediction/Images/GridSearch.png" title="Logistic Regression after using GridSearch">
</p>

<p align="center">
  <img src="./Mod_3_Project/dota-2-prediction/Images/Best Features.png" title="Best Features">
</p>

# Conclusion

This project has made me realize that feature engineering may be the most important part of machine learning. If you have features that easily predict the problem, then that is great. However, if you are given limited data points and they don't do a great job explaining the problem by themselves, it really takes a strong understanding of the problem and your features and how they relate with each other in order to fully utilize what you have to make a good model. 

Closely following this would be hyperparameter tuning. While my tree based models were initially terrible, after tuning out my random forest model, I was able to get strong predictions out the model which is a stark contrast with the initial results.

Trying to predict the match outcome with very limited information, such as the data from the first 5 minutes is going to be very difficult. 5 minutes out of a match that typically lasts out 35-45 minutes is roughly 11-15% of the match info. So getting results upwards of 67% is really great because it was only possible through my knowledge of the domain and my ability to engineer new features to help my models succeed. 

With the model above, I have placed 9th out of 31 participants on the kaggle competition with participants from 2-10 having results very similar to one another. 

Feature engineering is an artform and should be heavily prioritized in all projects using machine learning.
