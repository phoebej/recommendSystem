# coding: utf-8
#---@author: Zoe Zhao 
import numpy as np
import pandas as pd
import os
import time

np.random.seed(1)

#load the training data and make it become what we want---train.csv 

if os.path.exists('./data-new/train.csv'):
    print ("train.csv already exists,you can use it now")
else:
    with open('./data-new/train.txt', 'r') as f_train: 
        data = f_train.readlines()
        user_num = 0
        for line in data:
            if '|' in line:
                user_num += 1
        print('train_Users: ' + str(user_num))

        train = []
        idx_line = 0
        for i in range(user_num):
            idx = str(i) + '|'
            if idx in data[idx_line]:
                num_idx = len(idx)
                num = int(data[idx_line][num_idx: ])
                start_idx = idx_line + 1
                end_idx = start_idx + num 
                for line in data[start_idx : end_idx]:
                    odom = line.split()
                    train.append([i, int(odom[0]), int(odom[1])])
                idx_line = end_idx

        train = np.array(train)
        print('Load training datas done.')

        df = pd.DataFrame({'userId':train[:, 0], 'itemId': train[:, 1], 'rating': train[:, 2]})
        userId_col = df.pop('userId')
        df.insert(0,'userId',userId_col)
        df.to_csv('./data-new/train.csv', index = False)
        print('Write training datas to csv done.')

#load the testind data and make it become what we want---test.csv 
if os.path.exists('./data-new/test.csv'):
    print ('test.csv already exists,you can use it now')
else:
    with open('./data-new/test.txt','r') as f_test:
        data = f_test.readlines()
        user_num = 0
        for line in data:
            if '|' in line:
                user_num += 1
        print('test_Users: ' + str(user_num))

        test = []
        idx_line = 0
        for i in range(user_num):
            idx = str(i) + '|'
            if idx in data[idx_line]:
                num_idx = len(idx)
                num = int(data[idx_line][num_idx: ])
                start_idx = idx_line + 1
                end_idx = start_idx + num 
                for line in data[start_idx : end_idx]:
                    odom = line.split()
                    test.append([i, int(odom[0])])
                idx_line = end_idx

        test = np.array(test)
        print('Load test datas done.')

        df = pd.DataFrame({'userId':test[:, 0], 'itemId': test[:, 1]})
        userId_col = df.pop('userId')
        df.insert(0,'userId',userId_col)
        df.to_csv('./data-new/test.csv', index = False)
        print('Write testing datas to csv done.')

if os.path.exists('./data-new/train_without0.csv'):
    pass
else:
    with open('./data-new/train.txt', 'r') as f:
        data = f.readlines()
        user_num = 0
        for line in data:
            if '|' in line:
                user_num += 1
        print('Users: ' + str(user_num))

        train = []
        idx_line = 0
        for i in range(user_num):
            idx = str(i) + '|'
            if idx in data[idx_line]:
                num_idx = len(idx)
                num = int(data[idx_line][num_idx: ])
                start_idx = idx_line + 1
                end_idx = start_idx + num 
                for line in data[start_idx : end_idx]:
                    odom = line.split()
                    if int(odom[1])==0 :
                        continue
                    else:
                        train.append([i, int(odom[0]), int(odom[1])])
                idx_line = end_idx

        train = np.array(train)
        print('Load data done.')

        save1 = pd.DataFrame({'userId':train[:, 0], 'itemId': train[:, 1], 'rating': train[:, 2]})
        userId_col = save1.pop('userId')
        save1.insert(0,'userId',userId_col)
        save1.to_csv('./data-new/train_without0.csv', index = False)
        print('Write to csv done.')


#shuffle the data
if os.path.exists('./data-new/train_shuffle.csv'):
    pass
else:
    df = pd.read_csv('./data-new/train_without0.csv')
    #print (type(df))
    rows = len(df)
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    df.to_csv('./data-new/train_shuffle.csv', index = False)
    print ('shuffle already done')




#built the model
gradients = ["dL_db", "dL_dbu", "dL_dbv", "dL_dU", "dL_dV"]

class Model(object):
    def __init__(self, latent_factors_size, L2_bias=0, L2_emb=0):
        self.model_parameters = []
        self.gradients = []
        for (name, value) in self.initialize_parameters(latent_factors_size):
            setattr(self, name, value)
            self.gradients.append("dL_d%s" % name)
            self.model_parameters.append(name)
    
    # Used to save parameters during the optimization
    def save_parameters(self):
        return [(name, np.copy(getattr(self, name))) for name in self.model_parameters]
    
    # Used to reload the best parameters once the optimization is finished
    def load_parameters(self, parameters):
        for (name, value) in parameters:
            setattr(self, name, value)
    
    # Random embedding generation from normal distribution, given a size and variance
    def initialize_parameters(self, latent_factors_size=100, std=0.05):
        U = np.random.normal(0., std, size=(n_users + 1, latent_factors_size))
        V = np.random.normal(0., std, size=(n_items + 1, latent_factors_size))
        u = np.zeros(n_users + 1)
        v = np.zeros(n_items + 1)
        b = 0
        return zip(("b", "u", "v", "U", "V"), (b, u, v, U, V))
            
    # Compute the gradients of the biases and embeddings, given the user-item
    def compute_gradient(self, user_ids, item_ids, ratings):
        predicted_ratings = self.predict(user_ids, item_ids)
        residual = ratings - predicted_ratings

        # biases
        dL_db = -2 * residual
        dL_dbu = -2 * residual
        dL_dbv = -2 * residual

        # embeddings
        eu = self.U[user_ids]
        ev = self.V[item_ids]

        dL_dU = -2 * residual * ev
        dL_dV = -2 * residual * eu

        # Regularization
        l2 = 0.1
        dl2eu_dU = l2 * 2*eu
        dl2ev_dV = l2 * 2*ev
        dl2bu_dbu = l2 * 2*self.u[user_ids]
        dl2bv_dbv = l2 * 2*self.v[item_ids]
        
        dL_dbu = dL_dbu + dl2bu_dbu
        dL_dbv = dL_dbv + dl2bv_dbv
        dL_dU = dL_dU + dl2eu_dU
        dL_dV = dL_dV + dl2ev_dV
        
        return dict([(g, eval(g)) for g in gradients])
    
    # Sum of the biases and dot product of the embeddings
    def predict(self, user_ids, item_ids):
        user_ids = user_ids.astype('int')
        item_ids = item_ids.astype('int')
        return sum([self.b, 
                    self.u[user_ids], 
                    self.v[item_ids], 
                    (self.U[user_ids] * self.V[item_ids]).sum(axis=-1)])
    
    # Perform a gradient descent step
    def update_parameters(self, user, item, rating, learning_rate = 0.001):
        gradients = self.compute_gradient(user, item, rating)
        self.b = self.b - learning_rate * gradients['dL_db']
        self.u[user] = self.u[user] - learning_rate * gradients['dL_dbu']
        self.v[item] = self.v[item] - learning_rate * gradients['dL_dbv']
        self.U[user] = self.U[user] - learning_rate * gradients['dL_dU']
        self.V[item] = self.V[item] - learning_rate * gradients['dL_dV']

#some useful functions
def sample_random_training_index():
    return np.random.randint(0, len(train_set))

# Compute root mean squared error between x and y
def compute_rmse(x, y):
    return ((x - y)**2).mean()**0.5

# utilitary functions for getting the train/valid/test
def get_rmse(ratings):
    return compute_rmse(model.predict(*ratings.T[:2]), ratings.T[2])

def get_trainset_rmse():
    return get_rmse(train_set)

def get_validset_rmse():
    return get_rmse(valid_set)
def get_testset_rmse():
    return get_rmse(test_set)
#Get the basic statistics
def get_basic_statistics_with0():
    data = pd.read_csv('./data-new/train.csv')
    users_num = data.userId.nunique()
    items_num = data.itemId.nunique()
    ratings = len(data)
    return users_num,items_num,ratings

def get_basic_statistics_without0():
    data = pd.read_csv('./data-new/train_shuffle.csv')
    users_num = data.userId.nunique()
    items_num = data.itemId.nunique()
    ratings = len(data)
    return users_num,items_num,ratings

ori_users_num,ori_items_num,ori_ratings = get_basic_statistics_with0()
print ('\nIn this project,there are:\n%d users, \n%d items, \n%d ratings \nwhen we have not filter 0.\n' % (ori_users_num,ori_items_num,ori_ratings))
users_num,items_num,ratings = get_basic_statistics_without0()
print ('In this project,there are:\n%d users \n%d items \n%d ratings \nwe gonna use to train the system.\n' % (users_num,items_num,ratings))
interation = 0
train_errors = []
valid_errors = []
test_errors = []
best_parameters = None
best_validation_rmse = 9999
users_num = 0
items_num = 0
ratings_num = 0
sgd_iteration_count = 0
start_time = time.time()

train_set_rmse = float(0)
valid_set_rmse = float(0)
test_set_rmse = float(0)

reader = pd.read_csv('./data-new/train_shuffle.csv',chunksize = 1000000)
print ('we split the data into some chunks to evade the memory error\n')

for training_data in reader:
    interation = interation+1
    ratings = training_data[ [ 'userId','itemId','rating' ] ].values
    np.random.shuffle(ratings)
    n_users, n_items, _ = ratings.max(axis=0) + 1
    rows = len(ratings)
    
    #split the data 
    n = len(ratings)
    split_ratios = [0, 0.85, 0.95, 1]
    train_set, valid_set, test_set = [ratings[int(n*lo):int(n*up)] for (lo, up) in zip(split_ratios[:-1], split_ratios[1:])]
    if interation == 1 :
    	model = Model(latent_factors_size=100)
    
    model.b = train_set[:,2].mean()
    
    patience = 0
    update_frequency = 10000
    best_validation_rmse_chunk = 999
   
    while True:
        try:
            
            if sgd_iteration_count%update_frequency == 0:
                train_set_rmse = get_trainset_rmse()
                valid_set_rmse = get_validset_rmse()
                test_set_rmse = get_testset_rmse()
                
                train_errors.append(train_set_rmse)
                valid_errors.append(valid_set_rmse)
                test_errors.append(test_set_rmse)

                print ('In the No.%d chunk,Iteration:' % interation),
                print (sgd_iteration_count)
                print ('Validation RMSE: '),
                print (valid_set_rmse)

                if valid_set_rmse < best_validation_rmse_chunk and valid_set_rmse > 1:
                    print ('Test RMSE      :'),
                    print (test_set_rmse)
                    print ('It is the best validation error this chunk!')
                    patience = 0
                    best_validation_rmse_chunk = valid_set_rmse
                    if best_validation_rmse_chunk <best_validation_rmse:
                        best_validation_rmse = best_validation_rmse_chunk
                        best_parameters = model.save_parameters()
                        print ('Best validation error up to now!!!')
                else:
                    patience += 1
                    if patience >= 20:
                        print ('This chunk stop now!')
                        break
                    
            training_idx = sample_random_training_index()
            user, item, rating = train_set[training_idx]
            model.update_parameters(user, item, rating)
            sgd_iteration_count += 1
               
                
        except KeyboardInterrupt:
            print ('Stopped Optimization')
            print ('Current valid set performance=%s' % compute_rmse(model.predict(*valid_set.T[:2]), valid_set[:,2]))
            print ('Current test set performance=%s' % compute_rmse(model.predict(*valid_set.T[:2]), valid_set[:,2]))
            break


model.load_parameters(best_parameters)
print ('The best validation error is:'),
print (best_validation_rmse)
stop_time = time.time()

print ('Optimization time : '),
print ((stop_time - start_time)/60.), 
print ('minutes')

begin_time = time.time()
test = pd.read_csv('./data-new/test.csv')
test = test[['userId', 'itemId']].values
test_predictions = model.predict(*test.T[:2])
test_df = pd.DataFrame({'userId': test[:, 0],
                        'itemId': test[:, 1],
                        'prediction': test_predictions})



userId_col = test_df.pop('userId')
test_df.insert(0,'userId',userId_col)
test_df.to_csv('./data-new/test_with_rating.csv', index = False)
end_time = time.time()
print ('Rating already done!')
print ('The reasult has been save in test_with_rating.csv')
print ('predict time : '),
print ((end_time - begin_time)/60.), 
print ('minutes')

# write to test.txt
data = pd.read_csv('./data-new/test_with_rating.csv')
items = data['itemId'].values
pred = data['prediction'].values
num = data.userId.nunique()
print(num)

with open('./data-new/test.txt', 'a') as f:
    for i in range(num):
        idx_line = i * 6
        f.write('\n')
        f.write(str(i) + '|6')
        for j in range(6):
            f.write('\n')
            f.write(str(items[idx_line + j]) + ' ' + str(pred[idx_line + j]))

    f.close()  #oh~yeah~        espresso 赛高