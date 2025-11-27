# 25-2 Deep Learning 2022320140 김준영
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
 
class nn_linear_layer:
    
    # linear layer.
    # randomly initialized by creating matrix W and bias b
    def __init__(self, input_size, output_size, std=1):
        self.W = np.random.normal(0,std,(output_size,input_size))
        self.b = np.random.normal(0,std,(output_size,1))
    
    ######
    ## Q1
    def forward(self,x):
        '''
        x: (B, I)
        y: (B, O)
        W: (O, I)
        b: (O, 1)
        '''
        y = x @ self.W.T + self.b.T
        return y
     
    ######
    ## Q2
    ## returns three parameters
    def backprop(self,x,dLdy):
        dydW_b = x
        dydb_bo = np.ones((x.shape[0], 1))
        dydx_oi = self.W

        dLdW = dLdy.T @ dydW_b
        dLdb = dLdy.T @ dydb_bo
        dLdx = dLdy @ dydx_oi

        return dLdW,dLdb,dLdx

    def update_weights(self,dLdW,dLdb):

        # parameter update
        self.W=self.W+dLdW
        self.b=self.b+dLdb

def sigmoid(t_vec):
    return 1 / (1 + np.exp(-t_vec))
class nn_activation_layer:
    
    def __init__(self):
        pass
    
    ######
    ## Q3
    def forward(self,x):
        return sigmoid(x)
    
    ######
    ## Q4
    def backprop(self,x,dLdy):
        s = sigmoid(x)
        dydx = s * (1 - s) # (B, I)

        dLdx = dLdy * dydx  # activation 연산은 element wise 임.
        return dLdx

def softmax(x_vec):
    x_shift = x_vec - np.max(x_vec, axis=1, keepdims=True)
    exp_x = np.exp(x_shift)
    return exp_x / exp_x.sum(axis=1, keepdims=True)
class nn_softmax_layer:
    def __init__(self):
        pass
    ######
    ## Q5
    def forward(self,x):
        return softmax(x) # (B, I)
    
    ######
    ## Q6
    def backprop(self, x, dLdy):
        """
        x: logits (B, C)
        dLdy: upstream grad wrt softmax output (B, C)
        returns dLdx: grad wrt logits (B, C)
        Jacobian-vector product: J = diag(s) - s s^T
        dL/dx = s * (dL/dy - <dL/dy, s>)
        """
        s = self.forward(x)  # (B, C)
        dot = np.sum(dLdy * s, axis=1, keepdims=True)  # (B,1)
        dLdx = s * (dLdy - dot)  # (B, C)
        return dLdx
    
    # def backprop(self,x,dLdy):
    #     s = softmax(x)
    #     dydx = s * (1 - s)
    #     dLdx = dLdy * dydx
    #     return dLdx # (B, I)

class nn_cross_entropy_layer:
    def __init__(self):
        pass
        
    ######
    ## Q7
    # def forward(self, x, y):
    #     logx = np.log(x)
    #     return - np.mean((1-y) * logx[:, [0]] + y * logx[:, [1]])
    def forward(self,x,y):
        # x: probabilities (B, 2), y: labels (B, 1) with values {0,1}
        y_flat = y.reshape(-1, 1)
        # Build one-hot: class 0 is index 0 when y=0
        Y = np.concatenate([1.0 - y_flat, y_flat], axis=1)
        loss = -np.mean(np.sum(Y * np.log(x), axis=1))
        return loss
    
    ######
    ## Q8
    def backprop(self,x,y):
        # Gradient of cross-entropy w.r.t probabilities x
        y_flat = y.reshape(-1, 1)
        Y = np.concatenate([1.0 - y_flat, y_flat], axis=1)
        B = x.shape[0]
        dLdx = -(Y / x) / B
        return dLdx
    # def backprop(self, x, y):
    #     d0 = - (1-y) / x[:, [0]]   # (B,1)
    #     d1 = - y / x[:, [1]]    # (B,1)
    #     return np.hstack([d0, d1])   # (B,2)
    

if __name__ == '__main__':

    # number of data points for each of (0,0), (0,1), (1,0) and (1,1)
    num_d=5

    # number of test runs
    num_test=40

    ## Q9. Hyperparameter setting
    ## learning rate (lr)and number of gradient descent steps (num_gd_step)
    ## This part is not graded (there is no definitive answer).
    ## You can set this hyperparameters through experiments.
    lr= 0.1
    num_gd_step=8000

    # dataset size
    batch_size=4*num_d

    # number of classes is 2
    num_class=2

    # variable to measure accuracy
    accuracy=0

    # set this True if want to plot training data
    show_train_data=True

    # set this True if want to plot loss over gradient descent iteration
    show_loss=True

    ################
    # create training data
    ################

    m_d1 = (0, 0)
    m_d2 = (1, 1)
    m_d3 = (0, 1)
    m_d4 = (1, 0)

    sig = 0.05
    s_d1 = sig ** 2 * np.eye(2)

    d1 = np.random.multivariate_normal(m_d1, s_d1, num_d)
    d2 = np.random.multivariate_normal(m_d2, s_d1, num_d)
    d3 = np.random.multivariate_normal(m_d3, s_d1, num_d)
    d4 = np.random.multivariate_normal(m_d4, s_d1, num_d)

    # training data, and has shape (4*num_d,2)
    x_train_d = np.vstack((d1, d2, d3, d4))
    # training data lables, and has shape (4*num_d,1)
    y_train_d = np.vstack((np.zeros((2 * num_d, 1), dtype='uint8'), np.ones((2 * num_d, 1), dtype='uint8')))
    # y_train_d = np.vstack((np.ones((2 * num_d, 1), dtype='uint8'), np.zeros((2 * num_d, 1), dtype='uint8')))

    if (show_train_data):
        plt.grid()
        plt.scatter(x_train_d[range(2 * num_d), 0], x_train_d[range(2 * num_d), 1], color='b', marker='o')
        plt.scatter(x_train_d[range(2 * num_d, 4 * num_d), 0], x_train_d[range(2 * num_d, 4 * num_d), 1], color='r',
                    marker='x')
        plt.show()

    ################
    # create layers
    ################

    # hidden layer
    # linear layer
    layer1 = nn_linear_layer(input_size=2, output_size=4, )
    # activation layer
    act = nn_activation_layer()

    # output layer
    # linear
    layer2 = nn_linear_layer(input_size=4, output_size=2, )
    # softmax
    smax = nn_softmax_layer()
    # cross entropy
    cent = nn_cross_entropy_layer()

    # variable for plotting loss
    loss_out = np.zeros((num_gd_step))

    ################
    # do training
    ################

    for i in range(num_gd_step):
        
        # fetch data
        x_train = x_train_d
        y_train = y_train_d
            
        ################
        # forward pass
        
        # hidden layer
        # linear
        l1_out = layer1.forward(x_train)
        # activation
        a1_out = act.forward(l1_out)
        
        # output layer
        # linear
        l2_out = layer2.forward(a1_out)
        # softmax
        smax_out = smax.forward(l2_out)
        # cross entropy loss
        loss_out[i] = cent.forward(smax_out, y_train)
        
        ################
        # perform backprop
        # output layer
        # cross entropy
        b_cent_out = cent.backprop(smax_out, y_train)
        # softmax
        b_nce_smax_out = smax.backprop(l2_out, b_cent_out)
        
        # linear
        b_dLdW_2, b_dLdb_2, b_dLdx_2 = layer2.backprop(x=a1_out, dLdy=b_nce_smax_out)
        
        # backprop, hidden layer
        # activation
        b_act_out = act.backprop(x=l1_out, dLdy=b_dLdx_2)
        # linear
        b_dLdW_1, b_dLdb_1, b_dLdx_1 = layer1.backprop(x=x_train, dLdy=b_act_out)
        
        ################
        # update weights: perform gradient descent
        layer2.update_weights(dLdW=-b_dLdW_2 * lr, dLdb=-b_dLdb_2 * lr)
        layer1.update_weights(dLdW=-b_dLdW_1 * lr, dLdb=-b_dLdb_1 * lr)
        
        if (i + 1) % 2000 == 0:
            print('gradient descent iteration:', i + 1)

    # set show_loss to True to plot the loss over gradient descent iterations
    if (show_loss):
        plt.figure(1)
        plt.grid()
        plt.plot(range(num_gd_step), loss_out)
        plt.xlabel('number of gradient descent steps')
        plt.ylabel('cross entropy loss')
        plt.show()

    ################
    # training done
    # now testing

    num_test = 100

    for j in range(num_test):
        
        predicted = np.ones((4,))
        
        # dispersion of test data
        sig_t = 1e-2
        
        # generate test data
        # generate 4 samples, each sample nearby (1,1), (0,0), (1,0), (0,1) respectively
        t11 = np.random.multivariate_normal((1,1), sig_t**2*np.eye(2), 1)
        t00 = np.random.multivariate_normal((0,0), sig_t**2*np.eye(2), 1)
        t10 = np.random.multivariate_normal((1,0), sig_t**2*np.eye(2), 1)
        t01 = np.random.multivariate_normal((0,1), sig_t**2*np.eye(2), 1)
        
        # predicting label for test sample nearby (1,1)
        l1_out = layer1.forward(t11)
        a1_out = act.forward(l1_out)
        l2_out = layer2.forward(a1_out)
        smax_out = smax.forward(l2_out)
        predicted[0] = np.argmax(smax_out)
        print('softmax out for (1,1)', smax_out, 'predicted label:', int(predicted[0]))
        
        # predicting label for test sample nearby (0,0)
        l1_out = layer1.forward(t00)
        a1_out = act.forward(l1_out)
        l2_out = layer2.forward(a1_out)
        smax_out = smax.forward(l2_out)
        predicted[1] = np.argmax(smax_out)
        print('softmax out for (0,0)', smax_out, 'predicted label:', int(predicted[1]))
        
        # predicting label for test sample nearby (1,0)
        l1_out = layer1.forward(t10)
        a1_out = act.forward(l1_out)
        l2_out = layer2.forward(a1_out)
        smax_out = smax.forward(l2_out)
        predicted[2] = np.argmax(smax_out)
        print('softmax out for (1,0)', smax_out, 'predicted label:', int(predicted[2]))
        
        # predicting label for test sample nearby (0,1)
        l1_out = layer1.forward(t01)
        a1_out = act.forward(l1_out)
        l2_out = layer2.forward(a1_out)
        smax_out = smax.forward(l2_out)
        predicted[3] = np.argmax(smax_out)
        print('softmax out for (0,1)', smax_out, 'predicted label:', int(predicted[3]))
        
        print('total predicted labels:', predicted.astype('uint8'))
        
        accuracy += (predicted[0] == 0) & (predicted[1] == 0) & (predicted[2] == 1) & (predicted[3] == 1)
        
        if (j + 1) % 10 == 0:
            print('test iteration:', j + 1)

    print('accuracy:', accuracy / num_test * 100, '%')






