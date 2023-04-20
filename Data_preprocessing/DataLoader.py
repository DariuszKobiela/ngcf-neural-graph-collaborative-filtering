#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import os
from datetime import datetime
import ast
##########################################################
DATASETS_PATH = 'datasets/'
MOVIELENS_PATH = DATASETS_PATH + 'ml-1m/'
YELP2022_PATH = DATASETS_PATH + 'yelp2022/'
YELP2018_PATH = DATASETS_PATH + 'yelp2018/'
NYC2014_PATH = DATASETS_PATH + 'dataset_tsmc2014/'
TKY2014_PATH = DATASETS_PATH + 'dataset_tsmc2014/'
GOWALLA_PATH = DATASETS_PATH + 'gowalla/'
AMAZON_BOOK_PATH = DATASETS_PATH + 'amazon-book/'
##########################################################
DATASETS = {
    0: ['movielens', MOVIELENS_PATH],
    1: ['yelp2022', YELP2022_PATH],
    2: ['yelp2018', YELP2018_PATH],
    3: ['nyc2014', NYC2014_PATH],
    4: ['tky2014', TKY2014_PATH],
    5: ['gowalla', GOWALLA_PATH],
    6: ['amazon-book', AMAZON_BOOK_PATH]
}
##########################################################
SUBSET_COLUMNS = ['user_id', 'item_id']
DATASET_NAME_ORDINAL_NUMBER = 0
DATASET_PATH_ORDINAL_NUMBER = 1
##########################################################

CHOSEN_DATASET_NUMBER = 3 # here choose the dataset

##########################################################
DATASET_PATH = DATASETS[CHOSEN_DATASET_NUMBER][DATASET_PATH_ORDINAL_NUMBER]
DATASET_NAME = DATASETS[CHOSEN_DATASET_NUMBER][DATASET_NAME_ORDINAL_NUMBER]
##########################################################

def lists_to_delimited_numbers(df_lists):
    new_remap_id_item_list = list()

    for element in list(df_lists):
        cleared_element = str(element).strip('[]"').replace(',', '')
        new_remap_id_item_list.append(cleared_element)

    return new_remap_id_item_list

##########################################################
# START SCRIPT EXECUTION
##########################################################

# MEASURE DATASET LOADING TIME
start_time = datetime.now()
if DATASET_NAME == 'movielens':
    columns = ['UserID', 'MovieID','Rating', 'Timestamp']
    df = pd.read_csv('ml-1m/ratings.dat', header=0, sep='::', names=columns, engine='python')
    print(f"Unique users: {df.UserID.nunique()}, unique movies: {df.MovieID.nunique()}")
    print(df.info())
    df_subset = df[['UserID', 'MovieID']]
    df_subset.columns = SUBSET_COLUMNS
    print(df_subset)
    end_time = datetime.now()
    dataset_loading_time = end_time - start_time
    print(f'Dataset loading duration: {dataset_loading_time} days:hours:minutes.seconds')
    # end
    #
elif DATASET_NAME == 'yelp2022':
    iter_csv = pd.read_json(f"{DATASET_PATH}yelp_academic_dataset_review.json", lines=True, chunksize = 100000)  #nrows=100000 
    df = pd.concat([chunk for chunk in iter_csv])
    df = df.drop(['business_id', 'stars', 'useful', 'funny', 'cool', 'text', 'date'], axis=1)
    print(f"Unique users: {df.user_id.nunique()}, unique movies: {df.review_id.nunique()}")
    print(df.info())
    df_subset = df[['user_id', 'review_id']]
    df_subset.columns = SUBSET_COLUMNS
    print(df_subset)
    end_time = datetime.now()
    dataset_loading_time = end_time - start_time
    print(f'Dataset loading duration: {dataset_loading_time} days:hours:minutes.seconds')
    # end
    #
elif DATASET_NAME == 'yelp2018':
    iter_csv = pd.read_json(f"{DATASET_PATH}yelp_academic_dataset_review.json", lines=True, chunksize = 100000)  #nrows=100000 
    df = pd.concat([chunk for chunk in iter_csv])
    df = df.drop(['business_id', 'stars', 'useful', 'funny', 'cool', 'text', 'date'], axis=1)
    print(f"Unique users: {df.user_id.nunique()}, unique movies: {df.review_id.nunique()}")
    print(df.info())
    df_subset = df[['user_id', 'review_id']]
    df_subset.columns = SUBSET_COLUMNS
    print(df_subset)
    end_time = datetime.now()
    dataset_loading_time = end_time - start_time
    print(f'Dataset loading duration: {dataset_loading_time} days:hours:minutes.seconds')
    # end
    #
elif DATASET_NAME == 'nyc2014':
    columns = ['user_id', 'venue_id', 'venue_cat_id', 'venue_cat_name', 'lat', 'lon', 'timezone', 'UTC_time']
    df = pd.read_csv(f"{DATASET_PATH}dataset_TSMC2014_NYC.txt", sep='\t', header=None, names=columns, encoding='latin-1')
    print(f"Unique users: {df.user_id.nunique()}, unique venue_ids: {df.venue_id.nunique()}")
    print(df.info())
    df_subset = df[['user_id', 'venue_id']]
    df_subset.columns = SUBSET_COLUMNS
    print(df_subset)
    end_time = datetime.now()
    dataset_loading_time = end_time - start_time
    print(f'Dataset loading duration: {dataset_loading_time} days:hours:minutes.seconds')
    # end
    #
elif DATASET_NAME == 'tky2014':
    columns = ['user_id', 'venue_id', 'venue_cat_id', 'venue_cat_name', 'lat', 'lon', 'timezone', 'UTC_time']
    df = pd.read_csv(f"{DATASET_PATH}dataset_TSMC2014_TKY.txt", sep='\t', header=None, names=columns, encoding='latin-1')
    print(f"Unique users: {df.user_id.nunique()}, unique venue_ids: {df.venue_id.nunique()}")
    print(df.info())
    df_subset = df[['user_id', 'venue_id']]
    df_subset.columns = SUBSET_COLUMNS
    print(df_subset)
    end_time = datetime.now()
    dataset_loading_time = end_time - start_time
    print(f'Dataset loading duration: {dataset_loading_time} days:hours:minutes.seconds')
    # end
    #
elif DATASET_NAME == 'gowalla':
    print("No preprocessing needed.")
    # end
    #
elif DATASET_NAME == 'amazon-book':
    print("No preprocessing needed.")
    #end
    #

##########################################################

# MEASURE DATASET PREPROCESSING TIME
start_time = datetime.now()

saving_directory = f'{DATASET_PATH}{DATASET_NAME}/'
isExist = os.path.exists(saving_directory)
if not isExist:
    os.mkdir(saving_directory)
    print("Directory '% s' created" % saving_directory)

# ITEM LIST
item_list = pd.DataFrame(df_subset.item_id.unique(), columns=['item_id'])
item_list['item_remap_id'] = list(range(len(item_list.index)))
item_list.to_csv(f'{saving_directory}item_list.csv')
item_list.to_csv(f'{saving_directory}item_list.txt', index=False, sep=' ')
print(item_list)

# USER LIST
user_list = pd.DataFrame(df_subset.user_id.unique(), columns=['user_id'])
user_list['user_remap_id'] = list(range(len(user_list.index)))
user_list.to_csv(f'{saving_directory}user_list.csv')
user_list.to_csv(f'{saving_directory}user_list.txt', index=False, sep=' ')
print(user_list)

# SUBSET WITH REMAPPED USERS AND ITEMS
subset_users_remapped = df_subset.merge(user_list)
subset_users_remapped = subset_users_remapped.merge(item_list)
print(subset_users_remapped)

# ONLY REMAPPED USERS AND ITEMS
subset_remapped_only = subset_users_remapped[['user_remap_id', 'item_remap_id']].reset_index(drop=True)
print(subset_remapped_only)

# GROUPBY USERS
items_grouped = subset_remapped_only.groupby('user_remap_id')['item_remap_id'].apply(list)
print(items_grouped)
items_grouped_df = pd.DataFrame(items_grouped)
items_grouped_df.to_csv(f'{saving_directory}interactions_basic.csv')
print(items_grouped_df)
items_grouped_df = pd.read_csv(f'{saving_directory}interactions_basic.csv')
print(items_grouped_df)

# PREPARE ITEMS
new_remap_id_item_list = list()
for element in list(items_grouped_df.item_remap_id):
    cleared_element = element.strip('[]').replace(',', '')
    new_remap_id_item_list.append(cleared_element)
    
print(new_remap_id_item_list[0:2])
items_grouped_df['new_item_remap_id'] = new_remap_id_item_list
print(items_grouped_df)
print(items_grouped_df.item_remap_id[0])

# ADD LEN AND STRING LEN
items_grouped_df['item_remap_id_string_len'] = items_grouped_df['item_remap_id'].apply(lambda x: len(x))
items_grouped_df['item_remap_id_len'] = items_grouped_df['item_remap_id'].apply(lambda x: int((len(x)/3)-2))
print(items_grouped_df)

# TAKE ONLY USERS WITH NUMBER OF ITEMS > 10
print(f'All users: {len(items_grouped_df)}')
number_of_users_with_more_than_10_items = items_grouped_df.item_remap_id_len[items_grouped_df.item_remap_id_len > 10].count()
print(f'Users with number of items > 10: {number_of_users_with_more_than_10_items}')
new_items_grouped_df = items_grouped_df[items_grouped_df.item_remap_id_len > 10].reset_index(drop=True)
new_items_grouped_df['new_user_remap_id'] = list(range(len(new_items_grouped_df.index)))
print(new_items_grouped_df)

# UPDATE REMAPPED USERS TABLE
interesting_cut = new_items_grouped_df[['user_remap_id', 'new_user_remap_id']]
print(interesting_cut)
new_user_ids = user_list.merge(interesting_cut, left_on='user_remap_id', right_on='user_remap_id', how='outer')
new_user_ids = new_user_ids.drop('user_remap_id', axis=1)
print(new_user_ids)
new_user_ids = new_user_ids.dropna().reset_index(drop=True)
# second_new_user_ids = new_user_ids.drop('remap_id', axis=1)
second_new_user_ids = new_user_ids
print(second_new_user_ids)
second_new_user_ids.new_remap_id_user = second_new_user_ids.new_user_remap_id.astype(int)
second_new_user_ids.to_csv(f'{saving_directory}new_user_list.csv')
second_new_user_ids.to_csv(f'{saving_directory}new_user_list.txt', index=False, sep=' ')

# CONTINUE DATA CLEANING
print(new_items_grouped_df)
new_items_grouped_df = new_items_grouped_df.drop('user_remap_id', axis=1)
dataset_interactions = new_items_grouped_df[['new_user_remap_id', 'new_item_remap_id']]
dataset_interactions.to_csv(f'{saving_directory}interactions.csv')

# CREATE TRAIN/TEST SETS
temporary_cut = new_items_grouped_df[['new_user_remap_id', 'new_item_remap_id']]
print(temporary_cut)

TRAIN_SET_RATIO = 0.7 # 70%
TEST_SET_RATIO = 1.0-TRAIN_SET_RATIO
user_id_list = list()
train_set_list = list()
test_set_list = list()

for index, row in temporary_cut.iterrows():
    #print(f"index {index}")
    user = row['new_user_remap_id']
    # print(user)
    #print(user)
    user_items = row['new_item_remap_id']
    # print(user_items)
    # make_list_type_list = ast.literal_eval(user_items) # TOCHECK
    make_list_type_list = user_items.strip("'").split(" ")
    # print(type(make_list_type_list))
    # print(make_list_type_list)
    #list_of_integers = [int(i) for i in make_list_type_list] # TOCHECK
    list_of_integers = [int(i) for i in make_list_type_list] # TOCHECK
    #res = [eval(i) for i in make_list_type_list]
    #print(list_of_integers)
    
    user_id_list.append(user)
#     print(f"Current user: {user}")
#     print(f"Dataset before splitting: {user_items}")
#     print(f"Dataset type: {type(user_items)}")
    user_dataset_amount = len(list_of_integers)
#     print(f"Dataset len: {user_dataset_amount}")
#     print()
    train_set_amount = int(TRAIN_SET_RATIO*len(list_of_integers))
#     print(f"Train set amount: {train_set_amount}")
#     print(f"Train set: {user_items[0:train_set_amount]}")
    train_set_list.append(list_of_integers[0:train_set_amount])
#     print()
#     print(f"Test set amount: {user_dataset_amount-train_set_amount}")
#     print(f"Test set: {user_items[train_set_amount:]}")
    test_set_list.append(list_of_integers[train_set_amount:])
#     print()
#     if user>2:
#         break

final_dataset = pd.DataFrame(list(zip(user_id_list, train_set_list, test_set_list)), 
                                 columns=['user_id', 'train_set', 'test_set'])
final_dataset['new_train_set'] = lists_to_delimited_numbers(final_dataset.train_set)
final_dataset['new_test_set'] = lists_to_delimited_numbers(final_dataset.test_set)
print(final_dataset)

# SAVE TRAIN SET
train_interactions = final_dataset[['user_id', 'new_train_set']]
train_interactions.to_csv(f'{saving_directory}train_interactions.csv')
train_interactions.to_csv(f'{saving_directory}train_interactions.txt', index=False, sep=' ')
print(train_interactions)

# SAVE TEST SET
test_interactions = final_dataset[['user_id', 'new_test_set']]
test_interactions.to_csv(f'{saving_directory}test_interactions.csv')
test_interactions.to_csv(f'{saving_directory}test_interactions.txt', index=False, sep=' ')
print(test_interactions)

# MEASURE DATASET PREPROCESSING TIME
end_time = datetime.now()
dataset_preprocessing_time = end_time - start_time
print(f'Dataset preprocessing duration: {dataset_preprocessing_time} days:hours:minutes.seconds')

