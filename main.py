import streamlit as st
import joblib
from datetime import date
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import json
import random
import random as cllf

users = pd.read_csv("users.csv")

clf = joblib.load('match_predict.joblib')

ints = ['Art & Craft', 'Bollywood', 'Comedy', 'Design', 'Badminton', 'Singing', 'Writing', 'Dancing', 'Mountains', 'Biryani', 'Badminton', 'Design', 'Writing', 'Open-Minded', 'Coffee', 'Singing', 'Cricket', 'Concerts', 'Vegetarian', 'Tea', 'Design', 'Art & Craft', 'Badminton', 'Cricket', 'Photography', 'Camping', 'Beer', 'Beaches', 'Mountains', 'Foodie', 'Boxing', 'Cycling', 'Gym', 'Backpacking', 'Coffee', 'Writing', 'Photography', 'Gym', 'Festivals', 'Vegetarian', 'Documentaries', 'Horror', 'Cycling', 'Mountains', 'Coffee', 'Badminton', 'Photography', 'Dancing', 'Art & Craft', 'Writing', 'Singing', 'Horror', 'Comedy', 'Gym', 'Swimming', 'Writing', 'Art & Craft', 'Reels', 'Cricket', 'Running', 'Cricket', 'Badminton', 'Comedy', 'Open-Minded', 'Festivals', 'Gym', 'Camping', 'Backpacking', 'Mountains', 'Swimming', 'Art & Craft', 'Writing', 'Basketball', 'Biryani', 'Whiskey', 'Singing', 'Design', 'Singing', 'Clubs', 'Sci-fi', 'Wine', 'Foodie', 'Singing', 'Writing', 'Photography', 'Yoga', 'Gym', 'Yoga', 'Gym', 'Cooking', 'Biryani', 'Coffee', 'LGBTQIA+ ally', 'Writing', 'Biryani', 'Badminton', 'Volleyball', 'Whiskey', 'Anime', 'Writing', 'Cricket', 'Gym', 'Cooking', 'Ambitions', 'Open-Minded', 'Romantic', 'Romance', 'Vegetarian', 'Festivals', 'Badminton', 'Reels', 'Dancing', 'Photography', 'Basketball', 'Vegetarian', 'Singing', 'Romantic', 'Design', 'Romantic', 'Romance', 'Reels', 'Singing', 'Romantic', 'Art & Craft', 'Dancing', 'Open-Minded', 'Vegetarian', 'Romance', 'Festivals', 'Design', 'Singing', 'Design', 'Dancing', 'Writing', 'Singing', 'Festivals', 'Art & Craft', 'Vegetarian', 'Design', 'Writing', 'Reels', 'Festivals', 'Romantic', 'Romance', 'Reels', 'Art & Craft', 'Reels', 'Festivals', 'Dancing', 'Design', 'Design', 'Vegetarian', 'Romantic', 'Singing', 'Singing', 'Design', 'Writing', 'Art & Craft', 'Singing', 'Design', 'Romantic', 'Art & Craft', 'Festivals', 'Romantic', 'Romance', 'Vegetarian', 'Romantic', 'Design', 'Writing', 'Dancing', 'Reels', 'Romance', 'Singing', 'Festivals', 'Romantic', 'Romance', 'Design', 'Dancing', 'Romantic', 'Writing', 'Design', 'Dancing', 'Open-Minded', 'Singing', 'Festivals', 'Writing', 'Open-Minded', 'Reels', 'Singing', 'Festivals', 'Art & Craft', 'Dancing', 'Reels', 'Open-Minded', 'Singing', 'Dancing', 'Writing', 'Festivals', 'Writing', 'Romantic', 'Dancing', 'Design', 'Art & Craft', 'Dancing', 'Singing', 'Vegetarian', 'Festivals', 'Art & Craft', 'Dancing', 'Romance', 'Writing', 'Dancing', 'Vegetarian', 'Reels', 'Romantic', 'Singing', 'Reels', 'Writing', 'Romantic', 'Singing', 'Romance', 'Writing', 'Singing', 'Festivals', 'Design', 'Romance', 'Romance', 'Reels', 'Design', 'Festivals', 'Dancing', 'Reels', 'Open-Minded', 'Art & Craft', 'Festivals', 'Vegetarian', 'Art & Craft', 'Dancing', 'Vegetarian', 'Art & Craft', 'Romantic', 'Open-Minded', 'Reels', 'Romance', 'Open-Minded', 'Singing', 'Romance', 'Romantic', 'Festivals', 'Art & Craft', 'Writing', 'Design', 'Reels', 'Dancing', 'Dancing', 'Writing', 'Romance', 'Reels', 'Festivals', 'Design', 'Romantic', 'Writing', 'Vegetarian', 'Open-Minded', 'Romantic', 'Art & Craft', 'Singing', 'Vegetarian', 'Romance', 'Dancing', 'Vegetarian', 'Reels', 'Festivals', 'Singing', 'Design', 'Dancing', 'Writing', 'Festivals', 'Writing', 'Design', 'Open-Minded', 'Reels', 'Dancing', 'Writing', 'Open-Minded', 'Romantic', 'Vegetarian', 'Open-Minded', 'Writing', 'Romantic', 'Singing', 'Open-Minded', 'Dancing', 'Reels', 'Singing', 'Reels', 'Open-Minded', 'Festivals', 'Romance', 'Vegetarian', 'Design', 'Open-Minded', 'Dancing', 'Open-Minded', 'Romance', 'Singing', 'Dancing', 'Design', 'Art & Craft', 'Reels', 'Art & Craft', 'Reels', 'Singing', 'Romantic', 'Reels', 'Singing', 'Dancing', 'Design', 'Festivals', 'Romantic', 'Romance', 'Singing', 'Vegetarian', 'Writing', 'Design', 'Romantic', 'Vegetarian', 'Open-Minded', 'Singing', 'Art & Craft', 'Writing', 'Romantic', 'Dancing', 'Art & Craft', 'Cricket', 'Dancing', 'Dogs', 'Mystery', 'Mountains', 'Cricket', 'Badminton', 'Singing', 'Anime', 'Superhero', 'Photography', 'Festivals', 'Wine', 'Hip Hop', 'Romance', 'Photography', 'Badminton', 'Basketball', 'Running', 'Mountains', 'Singing', 'Photography', 'Cricket', 'Dogs', 'Cooking', 'Cats', 'Thriller', 'Mystery', 'Horror', 'Mountains', 'Tea', 'Gym', 'Bollywood', 'Family Oriented', 'Open-Minded', 'Gym', 'Dancing', 'Cycling', 'Running', 'Vegetarian', 'Anime', 'Hip Hop', 'Mystery', 'Open-Minded', 'Ambitions', 'Video Games', 'Biryani', 'Animated', 'Sci-fi', 'Superhero', 'Art & Craft', 'Running', 'Yoga', 'Poetry', 'Writing', 'Badminton', 'Clubs', 'Romance', 'Open-Minded', 'Bollywood', 'Singing', 'Art & Craft', 'Dancing', 'Reels', 'Badminton', 'Dancing', 'Volleyball', 'Cricket', 'Photography', 'Ambitions', 'Writing', 'Cricket', 'Running', 'Biryani', 'Tea', 'Badminton', 'Basketball', 'Cats', 'Comedy', 'Open-Minded', 'Writing', 'Photography', 'Volleyball', 'Tea', 'Mystery', 'Writing', 'Cricket', 'Cooking', 'Heritage', 'Mountains', 'K-drama', 'Anime', 'Football', 'Manga', 'Dogs', 'Singing', 'Pizza', 'Poetry', 'Feminism', 'Romance', 'Photography', 'Clubs', 'Romance', 'Romantic', 'Theatre', 'Cycling', 'Yoga', 'Swimming', 'Gym', 'Gymnastics', 'Cricket', 'Gym', 'Cafe-hopping', 'Camping', 'Biryani', 'Gym', 'Badminton', 'Beer', 'Anime', 'K-drama', 'Singing', 'Photography', 'Gym', 'Theatre', 'Clubs', 'Design', 'Gym', 'Volleyball', 'Coffee', 'Pizza', 'Comedy', 'Comic-books', 'Reality shows', 'K-drama', 'Foodie', 'Singing', 'Dancing', 'Cricket', 'Horror', 'Feminism', 'Gym', 'Volleyball', 'Cricket', 'Mountains', 'Stand Up', 'Singing', 'Cricket', 'Yoga', 'Gym', 'Heritage', 'Photography', 'Badminton', 'Tea', 'Ambitions', 'Crime', 'Cricket', 'Photography', 'Art & Craft', 'Badminton', 'Writing', 'Badminton', 'Cricket', 'Photography', 'Mountains', 'Singing', 'Dancing', 'Documentaries', 'Biryani', 'Festivals', 'Cycling', 'Swimming', 'Clubs', 'Cooking', 'Beaches', 'Pizza', 'Dogs', 'Cats', 'Dancing', 'Badminton', 'Singing', 'Coffee', 'Environmentalism', 'Cycling', 'Romance', 'Cricket', 'Clubs', 'Cafe-hopping', 'Camping', 'Foodie', 'Cricket', 'Swimming', 'Stand Up', 'Open-Minded', 'Vegetarian', 'Football', 'Football', 'Photography', 'Superhero', 'Open-Minded', 'Backpacking', 'Biryani', 'Tea', 'Cricket', 'Cooking', 'Festivals', 'Dancing', 'Badminton', 'Reels', 'Cycling', 'Gym', 'Foodie', 'Ambitions', 'Open-Minded', 'Backpacking', 'Cricket', 'Horror', 'Mystery', 'Human Rights', 'Coffee', 'Badminton', 'Cricket', 'Cycling', 'Martial Arts', 'Mountains', 'Writing', 'Cafe-hopping', 'Feminism', 'Open-Minded', 'Biographies', 'Open-Minded', 'Foodie', 'Ambitions', 'Biographies', 'Biryani', 'Singing', 'Dancing', 'Romance', 'Open-Minded', 'Reels', 'Art & Craft', 'Dancing', 'Photography', 'Cricket', 'Dogs', 'Cooking', 'Heritage', 'Tea', 'Biryani', 'Cats', 'Singing', 'Writing', 'Design', 'Photography', 'Cricket', 'Writing', 'Cricket', 'Yoga', 'Gymnastics', 'Comedy', 'Writing', 'Cricket', 'Beer', 'Mystery', 'Comedy', 'Whiskey', 'Pop', 'Dogs', 'Anime', 'Manga', 'LGBTQIA+ ally', 'Singing', 'Open-Minded', 'Badminton', 'Writing', 'Photography', 'Yoga', 'Running', 'Badminton', 'Running', 'Cycling', 'Tennis', 'Cafe-hopping', 'Romance', 'Sci-fi', 'Gym', 'Reels', 'Open-Minded', 'Cricket', 'Badminton', 'Concerts', 'Football', 'Clubs', 'Singing', 'Photography', 'Mountains', 'Beaches', 'Cycling', 'Singing', 'Reels', 'Photography', 'Heritage', 'Biryani', 'Boxing', 'Basketball', 'Swimming', 'Cycling', 'Tennis', 'Concerts', 'Cafe-hopping', 'Cooking', 'Mountains', 'Biryani', 'Writing', 'Badminton', 'Dancing', 'Sci-fi', 'Documentaries', 'Cricket', 'Reels', 'Badminton', 'Cycling', 'Romantic', 'Singing', 'Photography', 'Theatre', 'Clubs', 'Cafe-hopping', 'Writing', 'Photography', 'Basketball', 'Swimming', 'Video Games', 'Cricket', 'Gym', 'Bollywood', 'Environmentalism', 'Feminism', 'Gym', 'Cricket', 'Dancing', 'Concerts', 'Theatre', 'Photography', 'Heritage', 'Backpacking', 'Reels', 'Writing', 'Photography', 'Cricket', 'Cafe-hopping', 'Writing', 'Photography', 'Yoga', 'Concerts', 'Theatre', 'Writing', 'Photography', 'Cycling', 'Cafe-hopping', 'Tea', 'Swimming', 'Singing', 'Art & Craft', 'Dancing', 'Writing', 'Singing', 'Photography', 'Reels', 'Dancing', 'Comic-books', 'Gym', 'Foodie', 'Romance', 'Superhero', 'Open-Minded', 'Singing', 'Badminton', 'Basketball', 'Running', 'Dancing', 'Tennis', 'Singing', 'Writing', 'Cricket', 'Running', 'Football', 'Photography', 'Cricket', 'Yoga', 'Cycling', 'Tennis', 'Crime', 'Tea', 'Cooking', 'Foodie', 'Poetry', 'Photography', 'Badminton', 'Running', 'Tennis', 'Volleyball', 'Art & Craft', 'Singing', 'Gym', 'Tea', 'Coffee', 'Maggi', 'Foodie', 'Tea', 'Biryani', 'Dogs', 'Camping', 'Singing', 'Writing', 'Photography', 'Romantic', 'Badminton', 'Cycling', 'Tennis', 'Cooking', 'Vegetarian', 'Biryani', 'Video Games', 'Football', 'Dancing', 'Singing', 'Boxing', 'Gymnastics', 'Martial Arts', 'Cricket', 'Running', 'Swimming', 'Art & Craft', 'Foodie', 'Swimming', 'Bollywood', 'Documentaries', 'Tea', 'Heritage', 'Writing', 'Theatre', 'Cafe-hopping', 'Tea', 'Comedy', 'Yoga', 'Cycling', 'Biryani', 'Tea', 'Beer', 'Football', 'Gym', 'Foodie', 'Anime', 'Hip Hop', 'Yoga', 'Photography', 'Coffee', 'Maggi', 'Cooking', 'Singing', 'Writing', 'Reels', 'Romance', 'Romantic', 'Cycling', 'Tea', 'Biryani', 'Romance', 'Biryani', 'Beer', 'Whiskey', 'Documentaries', 'Romance', 'Football', 'Tea', 'Mystery', 'Photography', 'Backpacking', 'Writing', 'Photography', 'Martial Arts', 'Gardening', 'Mountains', 'Photography', 'Wine', 'Thriller', 'Pizza', 'Poetry', 'Theatre', 'Mountains', 'Beaches', 'Cats', 'Country', 'Writing', 'Folk & Acoustic', 'Mountains', 'Swimming', 'Poetry', 'Writing', 'Cricket', 'Football', 'Cycling', 'Volleyball', 'Badminton', 'Basketball', 'Cycling', 'Backpacking', 'Mountains', 'Feminism', 'Gym', 'Ambitions', 'Whiskey', 'Coffee', 'Mountains', 'Gym', 'Cooking', 'Beer', 'Mountains', 'Foodie', 'Writing', 'Reels', 'Badminton', 'Photography', 'Cricket', 'Singing', 'Art & Craft', 'Stand Up', 'LGBTQIA+ ally', 'Pizza', 'Heritage', 'Mountains', 'Open-Minded', 'Bollywood', 'Art & Craft', 'Writing', 'Football', 'Cooking', 'Coffee', 'Writing', 'Badminton', 'Gym', 'Badminton', 'Basketball', 'Running', 'Mountains', 'Camping', 'Writing', 'Badminton', 'Cricket', 'Photography', 'Comedy', 'Cycling', 'Cooking', 'Thriller', 'Crime', 'Mystery', 'Badminton', 'Stand Up', 'Cooking', 'Camping', 'Beer', 'Writing', 'Mountains', 'Feminism', 'LGBTQIA+ ally', 'Singing', 'Writing', 'Badminton', 'Cricket', 'Camping', 'Horror', 'Badminton', 'Beer', 'Dogs', 'Folk & Acoustic', 'Photography', 'Football', 'Gymnastics', 'Action', 'Romantic', 'Swimming', 'Cooking', 'Pizza', 'Thriller', 'Mystery', 'Running', 'Vegetarian', 'Mountains', 'Photography', 'Martial Arts', 'Badminton', 'Tea', 'Biographies', 'Open-Minded', 'Poetry', 'Thriller', 'Human Rights', 'Foodie', 'Cricket', 'Poetry', 'Yoga', 'Maggi', 'Writing', 'Running', 'Cycling', 'Gym', 'Cafe-hopping', 'Backpacking', 'Photography', 'Open-Minded', 'Poetry', 'Ambitions', 'Art & Craft', 'Singing', 'Dancing', 'Photography', 'Romance', 'Romantic', 'Board game', 'Foodie', 'Badminton', 'Running', 'Beer', 'Cricket', 'Badminton', 'Gym', 'Dogs', 'Comedy', 'Writing', 'Cricket', 'Tea', 'Maggi', 'Bollywood', 'Singing', 'Reels', 'Photography', 'Cricket', 'Cycling', 'Cricket', 'Bollywood', 'Comedy', 'Tea', 'Biryani', 'Romance', 'Sci-fi', 'Desi', 'Tea', 'Photography', 'Singing', 'Writing', 'Running', 'Swimming', 'Human Rights', 'Singing', 'Writing', 'Cafe-hopping', 'Rock', 'Poetry', 'Singing', 'Art & Craft', 'Design', 'Dancing', 'Writing', 'Singing', 'Gym', 'Clubs', 'Camping', 'Pizza', 'Singing', 'Badminton', 'Football', 'Mountains', 'Pizza', 'Reels', 'Photography', 'Photography', 'Theatre', 'Cooking', 'Romantic', 'Romance', 'Football', 'Cooking', 'Gardening', 'Mountains', 'Ambitions', 'Open-Minded', 'Photography', 'Badminton', 'Beaches', 'Mountains', 'Coffee', 'Singing', 'Badminton', 'Cricket', 'Basketball', 'Tennis', 'Football', 'Tea', 'Swimming', 'Gym', 'Writing', 'Singing', 'Writing', 'Cafe-hopping', 'Foodie', 'Documentaries', 'Writing', 'Photography', 'Dancing', 'Reels', 'Coffee', 'Singing', 'Writing', 'Folk & Acoustic', 'Bollywood', 'Singing', 'Writing', 'Dancing', 'Photography', 'Art & Craft', 'Running', 'Football', 'Foodie', 'Biryani', 'Tea', 'Writing', 'Theatre', 'Stand Up', 'LGBTQIA+ ally', 'Feminism', 'Badminton', 'Swimming', 'Gym', 'Backpacking', 'Dogs', 'Open-Minded', 'Romantic', 'Dancing', 'Dogs', 'Crime', 'Art & Craft', 'Photography', 'Football', 'Mountains', 'Sci-fi', 'Photography', 'Badminton', 'Cooking', 'Mountains', 'Foodie', 'Cricket', 'Badminton', 'Basketball', 'Cycling', 'Hockey', 'Art & Craft', 'Badminton', 'Swimming', 'Cooking', 'Stand Up', 'Mountains', 'Beer', 'Pizza', 'Maggi', 'Whiskey', 'Photography', 'Writing', 'Badminton', 'Theatre', 'Thriller', 'Writing', 'Design', 'Photography', 'Art & Craft', 'Writing', 'Yoga', 'Poetry', 'Romantic', 'Backpacking', 'Coffee', 'Badminton', 'Gym', 'Stand Up', 'Beer', 'Dogs', 'Maggi', 'Cats', 'Anime', 'Drama', 'Cricket', 'Football', 'Writing', 'Documentaries', 'Ambitions', 'Singing', 'Documentaries', 'Romantic', 'Tennis', 'Cycling', 'Pizza', 'Human Rights', 'Cooking', 'Coffee', 'Pizza', 'Anime', 'K-drama', 'Environmentalism', 'Open-Minded', 'Romance', 'Thriller', 'Anime', 'Photography', 'Festivals', 'Board game', 'Cooking', 'Camping', 'Badminton', 'Running', 'Family Oriented', 'Maggi', 'Foodie', 'Basketball', 'Hockey', 'Video Games', 'Sci-fi', 'Art & Craft', 'Dancing', 'Reels', 'Badminton', 'Boxing', 'Writing', 'Cricket', 'Comedy', 'Tea', 'Classical', 'Romance', 'Family Oriented', 'Festivals', 'Singing', 'Design', 'Reels', 'Cricket', 'Swimming', 'Cycling', 'Camping', 'Mountains', 'Video Games', 'Beer', 'Tea', 'Dogs', 'Concerts', 'Boxing', 'Basketball', 'Swimming', 'Gym', 'Football', 'Writing', 'Vegetarian', 'Open-Minded', 'Fantasy', 'Poetry', 'Dancing', 'Badminton', 'Cooking', 'Singing', 'Dancing', 'Cricket', 'Basketball', 'Running', 'Badminton', 'Beer', 'Ambitions', 'Tea', 'Mystery', 'Singing', 'Writing', 'Photography', 'Dancing', 'Cooking', 'Dancing', 'Badminton', 'Cooking', 'Art & Craft', 'Foodie', 'Desi', 'Romance', 'Cricket', 'Basketball', 'Running', 'Football', 'Volleyball', 'Football', 'Foodie', 'Cycling', 'Beaches', 'Camping', 'Anime', 'Comedy', 'Family Oriented', 'Foodie', 'Cricket', 'Cricket', 'Swimming', 'Football', 'Romance', 'Romantic', 'Dancing', 'Design', 'Badminton', 'Tennis', 'Football', 'Photography', 'Cricket', 'Poetry', 'Feminism', 'Singing', 'Writing', 'Photography', 'Badminton', 'Tea', 'Writing', 'Theatre', 'Festivals', 'Beaches', 'Fantasy', 'Writing', 'Tea', 'Poetry', 'Romance', 'LGBTQIA+ ally', 'Heritage', 'Beer', 'Biryani', 'Poetry', 'Thriller', 'Anime', 'Camping', 'Beaches', 'Cafe-hopping', 'Environmentalism', 'Singing', 'K-drama', 'LGBTQIA+ ally', 'Running', 'Dancing', 'Feminism', 'LGBTQIA+ ally', 'Yoga', 'Sci-fi', 'Poetry', 'Singing', 'Writing', 'Yoga', 'Foodie', 'Cats', 'Basketball', 'Dancing', 'Writing', 'Fantasy', 'Drama', 'Cycling', 'Yoga', 'Dancing', 'Pizza', 'Cats', 'Badminton', 'Trans ally', 'Open-Minded', 'Swimming', 'Concerts', 'Singing', 'Yoga', 'Mountains', 'Coffee', 'Cats', 'Dancing', 'Badminton', 'K-drama', 'Foodie', 'Beaches', 'Photography', 'Singing', 'Badminton', 'Tea', 'Biryani', 'Art & Craft', 'Writing', 'Photography', 'Tea', 'Dogs', 'Singing', 'Writing', 'Photography', 'Cricket', 'Badminton', 'Singing', 'Dancing', 'Camping', 'Horror', 'Anime', 'Photography', 'Vegetarian', 'Coffee', 'Gardening', 'Design', 'Tea', 'Cats', 'Documentaries', 'Ambitions', 'Romance', 'Art & Craft', 'Swimming', 'Vegetarian', 'Mystery', 'Feminism', 'Singing', 'Dancing', 'Reels', 'Badminton', 'Photography', 'Singing', 'Art & Craft', 'Writing', 'Dancing', 'Design', 'Biryani', 'Tea', 'Photography', 'Romance', 'Poetry', 'Theatre', 'Singing', 'Writing', 'Dancing', 'Art & Craft', 'Dancing', 'Badminton', 'Boxing', 'Running', 'Rock', 'Dogs', 'Beer', 'Badminton', 'Singing', 'Writing', 'Dancing', 'Art & Craft', 'Basketball', 'Singing', 'Art & Craft', 'Photography', 'Dancing', 'Swimming', 'Singing', 'Dancing', 'Badminton', 'Beaches', 'Foodie', 'Mountains', 'Beaches', 'Running', 'Biryani', 'Anime', 'Writing', 'Photography', 'Yoga', 'Volleyball', 'Heritage', 'Art & Craft', 'Singing', 'Dancing', 'Design', 'Reels', 'Design', 'Martial Arts', 'Mountains', 'Wine', 'EDM', 'Dancing', 'Writing', 'Badminton', 'Photography', 'Cycling', 'Yoga', 'Cafe-hopping', 'Thriller', 'Gym', 'Cooking', 'Art & Craft', 'Writing', 'Poetry', 'Open-Minded', 'Singing', 'Writing', 'Martial Arts', 'Cooking', 'Mountains', 'Jazz', 'Rock', 'Photography', 'Design', 'Writing', 'Poetry', 'Romance', 'Romantic', 'Badminton', 'Dogs', 'Coffee', 'Foodie', 'Bollywood', 'Ambitions', 'Feminism', 'Open-Minded', 'Dancing', 'Festivals', 'Cooking', 'Foodie', 'Pizza', 'Art & Craft', 'Cooking', 'Biryani', 'Pizza', 'Bollywood', 'Writing', 'Design', 'Dancing', 'Romantic', 'Romance', 'Coffee', 'Romance', 'Environmentalism', 'Open-Minded', 'Superhero', 'Dancing', 'Design', 'Writing', 'Singing', 'Reels', 'Concerts', 'Theatre', 'Comedy', 'Foodie', 'Stand Up', 'Mountains', 'Foodie', 'Poetry', 'Ambitions', 'Romantic', 'Writing', 'Concerts', 'Cafe-hopping', 'Romantic', 'Thriller', 'Punjabi', 'Stand Up', 'Bollywood', 'Foodie', 'Maggi', 'Romance', 'Photography', 'Art & Craft', 'Concerts', 'Theatre', 'Coffee', 'Video Games', 'Comic-books', 'Singing', 'Writing', 'Art & Craft', 'Design', 'Badminton', 'Writing', 'Singing', 'Biryani', 'Tea', 'Bollywood', 'Sci-fi', 'Ambitions', 'Mountains', 'Biryani', 'Romance', 'Singing', 'Ambitions', 'Cafe-hopping', 'Open-Minded', 'Comedy', 'Dancing', 'Foodie', 'Sci-fi', 'Mystery', 'Romance', 'Cafe-hopping', 'Singing', 'Maggi', 'Pizza', 'Thriller', 'Vegetarian', 'Foodie', 'Fantasy', 'Romantic', 'Human Rights', 'Basketball', 'Concerts', 'Cafe-hopping', 'Cycling', 'Swimming', 'Singing', 'Dogs', 'LGBTQIA+ ally', 'Horror', 'Pop', 'Photography', 'Art & Craft', 'Badminton', 'Clubs', 'Bollywood', 'Art & Craft', 'Stand Up', 'Mountains', 'Foodie', 'Pop', 'K-drama', 'Pizza', 'Thriller', 'Feminism', 'Open-Minded', 'Cafe-hopping', 'Foodie', 'Stand Up', 'Horror', 'Art & Craft', 'Singing', 'Dancing', 'Basketball', 'Concerts', 'Art & Craft', 'Clubs', 'Beaches', 'Coffee', 'Foodie', 'Reels', 'Cafe-hopping', 'Tea', 'Foodie', 'Gymnastics', 'Writing', 'Dancing', 'Mountains', 'Vegetarian', 'Dogs', 'Camping', 'Mountains', 'Pizza', 'Foodie', 'Tea', 'Swimming', 'Backpacking', 'Mountains', 'Open-Minded', 'LGBTQIA+ ally', 'Photography', 'Tea', 'Family Oriented', 'Open-Minded', 'Foodie', 'Foodie', 'Stand Up', 'Concerts', 'Dancing', 'Bollywood', 'Backpacking', 'Art & Craft', 'Badminton', 'Ambitions', 'Photography', 'Art & Craft', 'Concerts', 'Clubs', 'Bollywood', 'Cafe-hopping', 'Coffee', 'Gym', 'Singing', 'Writing', 'Stand Up', 'Ambitions', 'Family Oriented', 'Gym', 'Volleyball', 'Coffee', 'Folk & Acoustic', 'Sci-fi', 'Photography', 'Cooking', 'Open-Minded', 'Environmentalism', 'Cats', 'Martial Arts', 'Concerts', 'Tea', 'Maggi', 'Dogs', 'Dancing', 'Foodie', 'Biryani', 'Bollywood', 'Crime', 'Writing', 'Dancing', 'Cricket', 'Boxing', 'Yoga', 'Writing', 'Singing', 'Swimming', 'Yoga', 'Cycling', 'Gardening', 'Singing', 'Classical', 'Beaches', 'Reels', 'Romantic', 'Photography', 'Singing', 'Art & Craft', 'Writing', 'Reels', 'Photography', 'Art & Craft', 'Environmentalism', 'Feminism', 'Country', 'Basketball', 'Cricket', 'Dancing', 'Festivals', 'Crime', 'Photography', 'Art & Craft', 'Writing', 'Design', 'Dancing', 'Open-Minded', 'Dogs', 'Romance', 'Bollywood', 'Cooking', 'Singing', 'Dancing', 'Photography', 'Art & Craft', 'Design', 'Art & Craft', 'Design', 'Basketball', 'Coffee', 'Photography', 'Art & Craft', 'Clubs', 'Reels', 'Festivals', 'Photography', 'Art & Craft', 'Cycling', 'Camping', 'Mountains', 'Dancing', 'Badminton', 'Coffee', 'Pizza', 'Foodie', 'Photography', 'Art & Craft', 'Dancing', 'Badminton', 'Design', 'Writing', 'Foodie', 'Feminism', 'Open-Minded', 'Poetry', 'Photography', 'Singing', 'Writing', 'Football', 'Gym', 'Photography', 'Coffee', 'Design', 'Mountains', 'Anime', 'Singing', 'Reels', 'Basketball', 'Foodie', 'Anime', 'Singing', 'Horror', 'Vegetarian', 'Superhero', 'Art & Craft', 'Dancing', 'Gymnastics', 'Basketball', 'Gym', 'Pizza', 'Crime', 'Superhero', 'Romance', 'Feminism', 'Mystery', 'Tea', 'Reels', 'Dancing', 'Biryani', 'Reels', 'Boxing', 'Cycling', 'Volleyball', 'Singing', 'Dancing', 'Singing', 'Writing', 'Art & Craft', 'Dancing', 'Ambitions', 'Vegetarian', 'Comedy', 'Bollywood', 'Open-Minded', 'Family Oriented', 'Feminism', 'LGBTQIA+ ally', 'Voter rights', 'Poetry', 'Romance', 'Dancing', 'Writing', 'Gym', 'Bollywood', 'Backpacking', 'Singing', 'Writing', 'Cats', 'Human Rights', 'Foodie', 'Punjabi', 'Gardening', 'Dancing', 'Mountains', 'Dancing', 'Cooking', 'Bollywood', 'K-drama', 'Romance', 'Photography', 'Mountains', 'Tea', 'Human Rights', 'Dancing', 'Art & Craft', 'Theatre', 'Comedy', 'Anime', 'Writing', 'Photography', 'Art & Craft', 'Mountains', 'Camping', 'Badminton', 'Cafe-hopping', 'Mountains', 'Dogs', 'Horror', 'Art & Craft', 'Cooking', 'Dogs', 'Bollywood', 'K-drama', 'Pizza', 'K-drama', 'Open-Minded', 'Art & Craft', 'Photography', 'Singing', 'Folk & Acoustic', 'Anime', 'Photography', 'Writing', 'Art & Craft', 'Badminton', 'Cycling', 'Wine', 'Dogs', 'Comedy', 'Romance', 'Romantic', 'Art & Craft', 'Dancing', 'Swimming', 'K-drama', 'Football', 'Cooking', 'Hockey', 'Gym', 'Cycling', 'Art & Craft', 'Reels', 'K-drama', 'Horror', 'Vegetarian', 'Mountains', 'Comic-books', 'Country', 'Jazz', 'Poetry', 'Clubs', 'Cats', 'Horror', 'K-drama', 'Superhero', 'Art & Craft', 'Dancing', 'Foodie', 'Anime', 'Dogs', 'Feminism', 'Reality shows', 'Foodie', 'Photography', 'Wine', 'Swimming', 'Yoga', 'Beaches', 'Writing', 'Foodie', 'Cats', 'Poetry', 'Open-Minded', 'Dancing', 'Mystery', 'Mountains', 'Singing', 'Theatre', 'Art & Craft', 'Dancing', 'Writing', 'Maggi', 'Cats', 'Romance', 'K-drama', 'Gym', 'Biryani', 'Drama', 'Open-Minded', 'Mountains', 'Writing', 'Dancing', 'Crime', 'Thriller', 'Comedy', 'Family Oriented', 'Mystery', 'Stand Up', 'Thriller', 'Ambitions', 'Dancing', 'Basketball', 'Badminton', 'Yoga', 'Coffee']

def one_hot(lis):
    ret = [1 if value in lis else 0 for value in ints]

    return ret

st.title("VIBE - Find your Match")


genders = ('MALE', 'FEMALE', 'ANYONE')
selected_gender = st.selectbox('Whom are you looking for ?', genders)

options = ints[:10]

# create a multiselect widget
selected_options = st.multiselect('Select your interests', options)

if(selected_gender):
    users_new=users
    users_new = users_new[users_new["gender"]==selected_gender[0:1]]
    users_new['date'] = pd.to_datetime(users_new['updatedAt'].str.replace('Z', '+00:00'))
    users_new = users_new.sort_values('date')
    users_new = users_new[:101]


    users_new["is_verified"] = users_new["is_verified"].apply(lambda x: 0 if x == False else 1)
    users_new["status"] = users_new["status"].apply(lambda x: 0 if x == "False" else 1)
    users_new["is_subscribed"] = users_new["is_subscribed"].apply(lambda x: 0 if x == False else 1)
    users_new = users_new[["gender", "interests", "is_verified", "status", "is_subscribed"]]
    data = list(users_new['interests'])
    list_of_lists = [json.loads(s) for s in data]
    users_new['interests'] = list_of_lists

    inr = one_hot(selected_options)
    is_verified1 = 1
    status1 = 1
    g = 0
    is_subscribed1 = 1
    if(selected_gender=="MALE"):
        g = 0
    if (selected_gender == "MALE"):
        g = 1
    else:
        g = 2

    top = []

    for idx in range(100):
        row_list = users_new.iloc[idx].values.tolist()

        new_list = [g, is_verified1, status1, is_subscribed1]

        if (row_list[0] == 'M'):
            new_list.append(0)

        else:
            new_list.append(1)

        new_list = new_list + [row_list[2]] + [row_list[3]] + [row_list[4]]

        s = one_hot(inr)
        t = one_hot(row_list[1])

        for i in s:
            new_list.append(i)

        for i in t:
            new_list.append(i)

        X = np.array(new_list)
        X = X.reshape(1, 3820)
        out = clf.predict_proba(X)

        sc = 0.5 + cllf.random() / 2
        top.append((sc, idx))

    cllf.shuffle(top)
    print(top)
    top = sorted(top, key = lambda x:x[0], reverse=True)
    st.write("Top Results: ")

    idx = top[0][1]
    df = users_new[idx:idx + 1]
    g = list(df["gender"].values)[0]
    ge = "Male"
    if(g=='F'):
        ge = "Female"

    i = list(df["interests"].values)[0]

    st.write("Your Top Match is!!..")
    t = top[0][0]*100
    print(t)
    st.write("Profile Match Percentage - {}".format(t))
    st.markdown("Name: XXXX")
    st.markdown("Gender: {}".format(ge))
    st.write(i)

    for i in range(2, 11):
        idx = top[i][1]

        df = users_new[idx:idx + 1]
        g = list(df["gender"].values)[0]
        ge = "Male"
        if (g == 'F'):
            ge = "Female"

        it = list(df["interests"].values)[0]

        st.write("Match No - {} is!!..".format(i))
        t = top[i][0] * 100
        st.write("Profile Match Percentage - {}".format(t))
        st.markdown("Name: XXXX")
        st.markdown("Gender: {}".format(ge))
        st.write(it)

