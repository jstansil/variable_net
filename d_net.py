import numpy as np

class v_net: 
    inp = hid = num_hid = out = 0
    alpha = 10 #affects the magnitude of weight updates
    syn = [] 

    #(number of inputs, number of hidden neurons, number of hidden layers, number of outputs)
    def __init__(self, inp, hid, num_hid, out):
        self.inp = inp + 1 #+1 bias neuron
        self.hid = hid
        self.num_hid = num_hid
        self.out = out
    
        for i in range(num_hid + 1): 
            if(i == 0): 
                new_syn = 2*np.random.random((self.inp,self.hid)) - 1
                self.syn.append(new_syn)
            elif(i == num_hid): 
                new_syn = 2*np.random.random((self.hid,self.out)) - 1
                self.syn.append(new_syn)
            else: 
                new_syn = 2*np.random.random((self.hid,self.hid)) - 1
                self.syn.append(new_syn)

    #activation function
    def nonlin(self, x, deriv = False):
        if(deriv == True):
            return (x*(1-x))
        return 1/(1+np.exp(-x))

    #returns output from a single input
    def sInput(self, given, out_only = True):
        activ = []
        init_activation = self.nonlin(np.dot(given, self.syn[0]))
        activ.append(init_activation)
        for j in range(1, self.num_hid + 1): 
            a = self.nonlin(np.dot(activ[j-1], self.syn[j]))
            activ.append(a)
        if(out_only == True):
            return activ[self.num_hid]
        return activ

    #feed given data through network
    def train(self, given, target, iterations):
        for t in range(iterations):
            net_size = self.num_hid #used for easy indexing. disregards input layer
            activations = []
            error = [None]*(net_size + 1)
            delta = [None]*(net_size + 1)
            
            #determine activations for every layer (excluding input)
            activations = self.sInput(given, False)
            
            #determine weight changes based on error
            output_error = target - activations[net_size]

            #progress report
            if(t % 5000 == 0):
                print("output:")
                print(activations[net_size])
                print("target:")
                print(target)
                print("error:")
                print(np.mean(np.abs(output_error)))
                print("--------")
                
            error[net_size] = output_error
            output_delta = output_error*self.nonlin(activations[net_size], deriv=True)
            delta[net_size] = output_delta
            for k in range((net_size - 1), -1, -1):
                e = np.dot(delta[k+1], (self.syn[k+1].T))
                error[k] = e
                d = e * self.nonlin(activations[k])
                delta[k] = d

            #update synapses using calculated delta values
            for l in range(net_size, 0, -1):
                self.syn[l] += self.alpha*np.dot(activations[l-1].T, delta[l])
            self.syn[0] += self.alpha*np.dot(given.T, delta[0])

        print("Error after training: " + str(np.mean(np.abs(output_error))))

    #set last element in each row as 1 for the bias neuron at input
    def setBias(self, data):
        for row in range(len(data)):
            for col in range(len(data[0])):
                if (col == len(data[0])-1):
                    data[row][col] = 1
        return data
                
    #returns a given amount of random matricies (sized for the net) for input or output
    def randData(self, trials, output=False): 
        if(output == True):
            data = np.random.random((trials, self.out))
            return data
        data = 2*np.random.random((trials, self.inp)) - 1
        data = self.setBias(data)
        return data

    #returns a given amount of binary matricies (sized for the net) for input or output
    def randBinary(self, trials, output=False):
        if(output == True):
            data = np.random.randint(2, size=(trials, self.out))
            return data
        data = np.random.randint(2, size=(trials, self.inp))
        data = self.setBias(data)
        return data

    #train on randomly generated data that is by default binary
    def quickTrain(self, iterations=100000, binary=True):       
        size = np.random.randint(1,10)
        if(binary == True):
            i = self.randBinary(size)
            o = self.randBinary(size, True)
            self.train(i,o, iterations)
        else:
            i = self.randData(size)
            o = self.randData(size, True)
            self.train(i,o, iterations)


    
    
 
