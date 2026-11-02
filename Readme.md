# VIBE - Intelligent NLP-Based Profile Matching
## P.S.- PROFILE SCORING AND FETCHING RANDOM PROFILES

### Demo Website Link - https://toteja-vibe-find-your-match-main-c9dio0.streamlit.app/

## TEAM NAME- BYE

## TEAM MEMBERS-

* Rishit Toteja
* Ishita Jain
* Abhisht Dixit
* Gunraj Singh


## INTRODUCTION-

There are two parts to the problem statement-
1) Profile scoring- For this we are going to evaluate each profile based on different parameters like verified, if 100% completion, number of users
liked him/her, number of matches received, paid subscription or a free profile, verified user or not, number of likes done vs received, dislike
by likes ratio performed and received, etc. This evaluation wold be done based on a scoring function deveoped by us rating the profile on the scale of 1 to 5 stars.

2) Fetching appropriate profiles for the users- For this we will use our rating system to get similar profiles. This is done to make sure all users don't end up seeing the same person. 100 latest accounts, are taken into consideration, shuffling them and showing them to users. So, latest profiles are more visible (Latest as in top 100 latest profiles). Also, for the first 24 hours, free accounts should be given a bias so that more people could view them and the chances for free to pro subscription increases. Also people getting zero or low number of matches should be shown more than people already receiving matches. This is to allow all people to get a match. These should be done according to a scoring mechanism so that we don't end up making too many db calls for each case. End goal is to create a function which takes your profile as an input and sends 10 profiles.


Dating Website Recommendation System
This is a machine learning project aimed at creating a recommendation system for a dating website. The system uses various AI and ML technologies to match users based on their interests, preferences, and behavior on the platform.

## Technologies/Frameworks

Programming Language

1) Python 3.10
2) Google Colaboratory Environment
3) PyCharm 2021.2.3


### Machine Learning Libraries

1) Scikit-learn
2) TensorFlow
3) Keras

### Database and Data Handling

1) Pandas
2) NumPy
3) Excel

### Web Development Framework

1) Streamlit
2) Flask

### Machine Learning Models

We used several machine learning models to train the recommendation system, including:

1) Decision Trees
2) Logisitic Regression
3) Gradient Boosted Trees
4) Support Vector Machines

We implemented these models using Scikit-learn, TensorFlow, and Keras libraries. The best accuracy was found out to be in Random Forest Classifier.

### Web Application

We built a web application using Streamlit to provide a user-friendly interface for the recommendation system. Users can create a profile on the website, input their preferences, and receive personalized recommendations based on their interests and behavior on the platform.

### Future Work

In the future, we plan to deploy the machine learning model on AWS to reduce the inference time and improve front-end of our website
