# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 18:12:43 2020

@author: samwh


Section 1: Class to construct and manipulate keras models
Section 2: General functions required for new network interface
Section 3: Classes to allow for creation of sparse pseudo-networks

Notes:
    - NeuralNet objects updated to allow for use of different activation and
    optimizer functions. Currently this can only be changed for the entire
    model rather than individual layers.
    - Currently no functionality to allow for dropout layer, but this could be
    implemented if needed.
"""

import random
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models     import Sequential
from keras.layers     import Dense
from keras.models     import load_model


# lists containing model activations and optimizers to allow for randomisation
hidden_layer_activations = ['tanh', 'relu', 'sigmoid', 'linear', 'softmax']
model_optimizers = ['adam', 'rmsprop']


###############################################################################


'''
Section 1
    
Class to define Neural Net object, capable of loading an existing network from a .h5 file:
    network_name = NeuralNet(file="Neural_Network.h5") 

OR training a new network from a 'training_data' array with specified 'architecture' (layer structure):
    network_name = NeuralNet(training_data=array_name, architecture=[10,10,10]) 

OR creating an untrained neural net with specified 'input_size', 'output_size' and 'architecture':
    network_name = NeuralNet(input_size=4, output_size=2, architecture=[10,10,10])

Function overview:
    - To train a model, call 'fit(training_data)' where training_data is a list of inputs and outputs
    - Network can be saved to a .h5 file by calling the 'save(file_name)' method
    - To make a prediction, call 'predict(input_data)' method where input_data is an array of observations
    - Stored network architecture can be updated (e.g. after mutation) by calling 'get_architecture()' method
    - Return lists containing network weights and biases by calling 'get_weights()' (also updates internal variables)
    - Set network weights by calling 'set_weights(weights, biases)', input same data format as get_weights() output
'''

class NeuralNet():
    
    def __init__(self, 
                 file = None, 
                 architecture = None, 
                 input_size = None, 
                 output_size = None, 
                 training_data = None,
                 training_epochs = 1,
                 activation = 'sigmoid',
                 optimizer = 'adam'
                 ):
        
        def build_model(input_size, output_size, architecture, activation, optimizer):
            
            model = Sequential()
            
            if len(architecture) > 0:
            
                model.add(Dense(architecture[0], input_dim=input_size, activation=activation))
                for i in range(1, len(architecture)):
                    model.add(Dense(architecture[i], activation=activation))
                model.add(Dense(output_size, activation='linear'))
                
            else: model.add(Dense(output_size, input_dim=input_size, activation='linear'))
                            
            model.compile(loss='mse', optimizer=optimizer)
           
            return model

        if training_data: input_size, output_size = len(training_data[0][0]), len(training_data[0][1])
        
        if file:
            
            self.model = load_model(file)
            print('Loaded ' + file, '\n')
            
        else: self.model = build_model(input_size, output_size, architecture, activation, optimizer)
        
        if training_data: self.fit(training_data, training_epochs)
        
        self.get_architecture()
        self.get_weights()
        self.activation, self.optimizer = activation, optimizer
        
    
    # fits model to given training data    
    def fit(self, training_data, epochs=10):
            
        inputs = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
        outputs = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))
        self.model.fit(inputs, outputs, epochs=epochs)
            
    
    # saves network inc. weights to a .h5py file with a given name
    def save(self, name='Neural_Network'):
        
        self.model.save(name + '.h5')
        print('Saved to {}'.format(name + '.h5'), '\n')
        
    
    # use the model to generate an output from a given input
    def predict(self, input_data):
        
        return self.model.predict(np.array(input_data).reshape(-1, len(input_data)))[0]
    
    
    # returns architecture (hidden layer structure) of network, also updates internal variables
    def get_architecture(self):
        
        architecture = []
        for layer in self.model.layers: architecture.append(layer.output_shape[1])
        
        self.output_size = architecture[-1]
        architecture.pop()
        self.architecture = architecture
        
        return architecture
    
    
    # outputs weights and biases of network, also updates internal variables
    def get_weights(self, _print=False):    
    
        weights, biases = [], []
        for layer in self.model.layers:
            weights.append(layer.get_weights()[0])
            biases.append(layer.get_weights()[1])
        
        if _print: print('Weights:', weights, '\n\nBiases:', biases, '\n')
        
        self.weights, self.biases = weights, biases
        self.input_size = len(weights[0])
        
        return weights, biases
    
    
    # accepts lists containing weights and biases and uses these to replace existing network weights
    def set_weights(self, weights, biases):
        
        count = 0
        for layer in self.model.layers:
            layer.set_weights([weights[count], biases[count]])
            count += 1
            
        self.get_weights()
    

###############################################################################
        
        
'''
Section 2

Generic functions to be used by later methods:
    - Function to reset all weights to zero
    - Function to add a node with given weight
    - Function to delete a node
'''

# function to take arrays for weights and biases and set all values to zero
def set_zero_weights(weights, biases):

    # set weights values
    for layer in range(len(weights)):
        for node in range(len(weights[layer])):
            for connection in range(len(weights[layer][node])): weights[layer][node][connection] = 0
      
    # set bias values          
    for layer in range(len(biases)):
        for node in range(len(biases[layer])): biases[layer][node] = 0
    
    return weights, biases


# function to handle adding a given number ('diff') of nodes to a network architecture
# accepts the weights, biases and architecture of a network, the layer which the nodes will be added to
# the number of new nodes 'diff' and a bool to allow for randomising new node connection weights
def add_node(weights, biases, architecture, layer, diff=1, rand=False):
    
    # subfunction returns zero unless rand=True, in which case returns a random number from -1 to 1
    def new_val():
        
        if rand: return 2*random.random() - 1
        else: return 0
    
    # ensure diff is always positive
    if diff < 0: diff = -diff
    
    # add nodes to architecture  
    if len(architecture) == 0: architecture.append(0) 
    architecture[layer] += diff
    
    # handle addition of weights into the new nodes    
    new_layer = []
    for i in range(len(weights[layer])):
        l = [elem for elem in weights[layer][i]]
        for j in range(diff): l.append(new_val())
        new_layer.append(np.array(l))
    
    weights[layer] = np.array(new_layer)
    
    # handle addition of weights out of the new nodes 
    new_layer = []
    for i in range(len(weights[layer+1])): new_layer.append(np.array([elem for elem in weights[layer+1][i]]))
    for i in range(diff): new_layer.append(np.array([new_val() for j in range(len(weights[layer+1][0]))]))
    
    weights[layer+1] = np.array(new_layer)
    
    # new biases are always set to zero
    biases[layer] = np.append(biases[layer], [0 for i in range(diff)]) 
    
    return weights, biases, architecture


# handles deletion of a given number 'diff' of nodes from a specified layer 
def del_node(weights, biases, architecture, layer, diff=-1):
    
    # ensure diff is always negative
    if diff > 0: diff = -diff
    
    # delete nodes from architecture
    architecture[layer] += diff
    
    # delete associated weights and biases
    weights[layer] = np.delete(weights[layer], [len(weights[layer][0]) - 1 - i for i in range(abs(diff))], 1)
    weights[layer+1] = np.delete(weights[layer+1], [len(weights[layer+1]) - 1 - i for i in range(abs(diff))], 0)
    
    biases[layer] = np.delete(biases[layer], [len(biases[layer]) - 1 - i for i in range(abs(diff))])
    
    return weights, biases, architecture


###############################################################################
  
    
'''
Section 3

Classes for use in creating a 'pseudo-network', which can be sparsely rather than densely connected:
    - Connections between nodes are stored as ConnectionGene objects
    - Nodes are stored as NodeGene objects
    - Networks are stored as Genome objects, which take connection and node genes as arguments
    
Create a node as follows (layers are number in range [0,1], where 0,1 are input/ouput layers):
    node = NodeGene(bias=0, num=1, layer=0)
    
Connect two nodes and create a connection gene (cannot be in same layer):
    connection = node.add_connection(other_node, weight=0.5)
    
Create a genome:
    genome = genome(connection_genes = [#list of connection genes#]
                            node_genes = [#list of node genes#]
                            )
                            
Compile genetic information into a keras model:
    model = genome.compile_network()
    
Mutate a genome:
    genome.mutate(gin = #global innovation number object,
                  add_node_prob = #probability to add a new node
                  add_connection_prob = #probability to add a new connection
                  alt_weight_prob = #probability to perturb each weight,
                  alt_weight_gauss = #True to perturb with gaussian noise, False for new random value,
                  alt_weight_std = #Standard deviation for gaussian perturbation,
                  alt_bias_prob = #probability to add or delete biases for each node
                  )
'''

# define object to contain information relating to a connection between nodes
class ConnectionGene():
    
    def __init__(self, 
                 weight = 0,                    # weight of connection
                 vector = [],                   # start and end point (NodeGene objects)
                 innovation = 1,                # innovation number (NEAT)
                 activated = True,              # set to False to deactivate gene
                 
                 # not necessary for initialisation
                 disjoint = False,
                 excess = False
                 ):
        
        self.weight = weight
        self.vector = vector
        self.innovation = innovation
        self.activated = activated
        self.disjoint = disjoint
        self.excess = excess
        
        
    # function to deactivate gene
    def deactivate(self):
        
        self.activated = False
        
        
    # function to produce a copy of the original connection with optional altered weight
    def copy(self, weight=None):
        
        if not weight: weight = self.weight
        
        return ConnectionGene(weight = weight,
                              vector = self.vector,
                              innovation = self.innovation,
                              activated = self.activated
                              )
                
        
# object to store information relating to a node in the network
# both node genes and connection genes can be 'turned off' by setting activated=False
# layer number must be in range [0,1] where 0 = input layer, 1 = output layer
class NodeGene():
    
    def __init__(self, 
                 num,                           # each node has an associated reference number
                 layer,                         # in range [0,1]          
                 activated = True,              # set to False to deactivate gene
                 
                 # not necessary for initialisation
                 eff_layer = None, 
                 eff_node_index = None,
                 init_vector = None,
                 init_innovations = None
                 ):

        if layer == 0:
            eff_layer = 0
            self.type = 'input'
            
        elif layer == 1: self.type = 'output'
            
        else: self.type = 'hidden'
        
        self.num = num
        self.layer = layer
        self.activated = activated
        
        self.eff_layer = eff_layer
        self.eff_node_index = eff_node_index
        self.init_vector = init_vector
        self.init_innovations = init_innovations

    
    # call this function to add a connection to another node (must be in a different layer)
    # returns a connection gene object
    def add_connection(self, other, weight=0, innovation=0, activated=True):
        
        if self.layer == other.layer: 
            
            print('Tried to connect two nodes in same layer.\n')
            return None
            
        if self.layer < other.layer: vector = [self, other]
        elif self.layer > other.layer: vector = [other, self]
                  
        # creates a new connection gene and adds it to lists for both nodes
        new_connection = ConnectionGene(weight=weight, 
                                        vector=vector, 
                                        innovation=innovation, 
                                        activated=activated
                                        )
        
        return new_connection


# class to store genetic information of an object, as well as model information and fitness
# model can be generated from node and connection genes by calling the compile_network function
class Genome():
    
    def __init__(self, 
                 connection_genes = [],         # ConnectionGene objects
                 node_genes = [],               # NodeGene objects
                 biased_nodes = [],             # list of NodeGene objects with bias applied
                 input_size = None, 
                 output_size = None,
                 activation = 'sigmoid',    
                 optimizer='adam',              # not necessary for purely evolutionary methods
                 fitness = None,                # fitness value
                 species = 0                    # species of genome in NEAT population
                 ):
        
        self.connection_genes = connection_genes 
        self.node_genes = node_genes
        self.biased_nodes = biased_nodes
        self.input_size = input_size 
        self.output_size = output_size
        self.activation = activation 
        self.optimizer = optimizer
        self.get_outer_nodes()
        self.fitness = fitness
        self.species = species
        
        self.model = None
        self.adjusted_fitness = None
     
    # deep copy function
    def copy(self):
        
        return Genome(connection_genes = [c.copy() for c in self.connection_genes],
                      node_genes = self.node_genes,
                      biased_nodes = self.biased_nodes,
                      input_size = self.input_size, 
                      output_size = self.output_size,
                      activation = self.activation, 
                      optimizer = self.optimizer,
                      fitness = self.fitness,
                      species = self.species
                      )
    
        
    # add or delete bias to node
    def alt_bias(self, node):
        
        if node in self.node_genes: 
            
            # either adds or deletes node from bias list depending on 'add' bool
            if node in self.biased_nodes: self.biased_nodes.remove(node)
            else: self.biased_nodes.append(node)
 
    
    # generate list of input and output nodes, returns list of all outer nodes
    def get_outer_nodes(self):
        
        self.input_nodes = [node for node in self.node_genes if node.layer == 0]
        self.output_nodes = [node for node in self.node_genes if node.layer == 1]
        self.outer_nodes = self.input_nodes + self.output_nodes
        
        return self.input_nodes, self.output_nodes, self.outer_nodes


    # prints node information for the genome
    def get_node_info(self, _print=True):
        
        if _print:
            for node in self.node_genes: 
                
                if node in self.biased_nodes: bias_text = ' (biased=True)'
                else: bias_text = '' 
                
                print('Node {}: (type={}){}'.format(node.num, node.type, bias_text))                   
                print('Forward connections:', [n.num for n in [c.vector[1] for c in self.connection_genes if c.vector[0] == node]])
                print('Backward connections:', [n.num for n in [c.vector[0] for c in self.connection_genes if c.vector[1] == node]], '\n')
        
        return [n.num for n in self.node_genes]
    
    
    # prints connection info for the genome
    def get_connection_info(self, _print=True):
        
        if _print:
            
            for c in (c for c in self.connection_genes if c.activated):
                print(c.vector[0].num, '->', c.vector[1].num, '(i = {}):'.format(c.innovation), c.weight)
                
            print('\n')
        

    # function to compile a keras model from the connection and node genes of the genome
    # saves model to object attributes and also returns the new model
    def compile_network(self):
    
        # find input and output sizes from node gene list
        if not self.input_size: self.input_size = len([n for n in self.node_genes if n.layer == 0])
        if not self.output_size: self.output_size = len([n for n in self.node_genes if n.layer == 1])
        
        architecture = []
        current_layer = []
        current_eff_layer = 1
        current_eff_node_index = 0
        
        # remove any deactivated genes
        connection_genes = [c for c in self.connection_genes if c.activated == True]
        node_genes = [n for n in self.node_genes if n.activated == True]
        
        # remove connection genes if they reference a deactivated or non-existent node
        temp = [c for c in connection_genes]
        for c in temp: 
            if c.vector[0] not in node_genes or c.vector[1] not in node_genes:
                del connection_genes[connection_genes.index(c)]
        
        # remove duplicate connections in list, taking only the most recently added connection gene
        temp = [c for c in connection_genes]
        for c in temp: 
            if c.vector in [x.vector for x in connection_genes[connection_genes.index(c)+1:]]: 
                del connection_genes[connection_genes.index(c)]
                
        # remove unconnected nodes
        temp = [n for n in node_genes]
        for n in temp:
            if n not in [c.vector[0] for c in connection_genes] + [c.vector[1] for c in connection_genes]:
                del node_genes[node_genes.index(n)]
        
        # set node indices for input and output nodes
        for i in range(2):
            count = 0
            for n in (n for n in node_genes if n.layer == i): 
                n.eff_node_index = count
                count += 1
        
        # iterate through node genes to generate an architecture with layers of unconnected nodes
        node_genes.sort(key=lambda x: x.layer)
        for node_gene in (n for n in node_genes if n.layer not in [0,1]):
            
            # if node connected to current layer, move to next layer
            connected_to_layer = False
            for n in [c.vector[0] for c in connection_genes if c.vector[1] == node_gene]: 
                if n in current_layer: connected_to_layer = True
            
            if connected_to_layer:
    
                architecture.append(len(current_layer))
                current_layer = []
                current_eff_layer += 1
                current_eff_node_index = 0
                
            current_layer.append(node_gene)
            node_gene.eff_layer = current_eff_layer
            node_gene.eff_node_index = current_eff_node_index
            current_eff_node_index += 1
            
        # set node indices for input and output nodes
        for i in range(2):
            count = 0
            for n in (n for n in node_genes if n.layer == i): 
                n.eff_node_index = count
                count += 1
                if i == 0: n.eff_layer = 0
                elif len(current_layer) == 0: n.eff_layer = 1
                else: n.eff_layer = current_eff_layer + 1
            
        if len(current_layer) > 0: architecture.append(len(current_layer))
        
        # get shapes of weight and bias arrays
        temp_network = NeuralNet(None, architecture, self.input_size, self.output_size)
        weights, biases = temp_network.get_weights()
        weights, biases = set_zero_weights(weights, biases)
        
        # add all connection weights for adjacent layers
        for c in (c for c in connection_genes if c.vector[1].eff_layer - c.vector[0].eff_layer == 1):
            weights[c.vector[0].eff_layer][c.vector[0].eff_node_index][c.vector[1].eff_node_index] = c.weight

        # iterate through all node genes to add biases and remaining weights
        node_genes.sort(key=lambda x: x.layer, reverse=True)
        for node_gene in node_genes:
            
            # set bias value for node
            if node_gene.layer != 0 and node_gene in self.biased_nodes: 
                biases[node_gene.eff_layer-1][node_gene.eff_node_index] = 1
            
            # iterate through backwards node connections that span greater than one effective layer
            for n in (n for n in [c.vector[0] for c in connection_genes if c.vector[1] == node_gene] if node_gene.eff_layer - n.eff_layer > 1):              
                for layer in range(n.eff_layer, node_gene.eff_layer - 1):
                    
                    # add new node to layer with weights of 0
                    weights, biases, architecture = add_node(weights, biases, architecture, layer)                
                    
                    # connect new node to original node with weight of 1
                    if layer == n.eff_layer: weights[layer][n.eff_node_index][-1] = 1
                    else: weights[layer][-1][-1] = 1
                    
                    # connect the new pathway to the final node with the original connection weight
                    if layer == node_gene.eff_layer - 2: 
                        
                        connection_gene = [c for c in connection_genes if c.vector == [n, node_gene]][0]
                        weights[layer+1][-1][node_gene.eff_node_index] = connection_gene.weight  
           
        # generate keras model             
        self.model = NeuralNet(architecture = architecture, 
                               input_size = self.input_size, 
                               output_size = self.output_size,
                               activation = self.activation,
                               optimizer = self.optimizer
                               )
        
        # set the weights with the arrays generated above
        self.model.set_weights(weights, biases)    
        
        self.architecture = architecture   
    
        return self.model
    
    
    # function to mutate the genome in accordance with NEAT method
    def mutate(self,
               params = None,                   # object containing mutation parameters, e.g. NEAT
               gin = None,                      # global innovation number
               add_node_prob = 0.1,              
               add_connection_prob = 0.1,
               alt_weight_prob = 0.1,
               alt_weight_gauss = True,         # set to True to perturb using gaussian noise
               alt_weight_std = None,           # optional std for gaussian noise
               alt_bias_prob = 0.1
               ):
        
        # inherits mutation parameters from provided parameters, e.g. NEAT object
        if params:
            
            add_node_prob = params.add_node_prob
            add_connection_prob = params.add_connection_prob
            alt_weight_prob = params.alt_weight_prob
            alt_weight_gauss = params.alt_weight_gauss
            alt_weight_std = params.alt_weight_std
            alt_bias_prob = params.alt_bias_prob
            
            gin = params.gin
            all_node_genes = params.node_genes
        
        else: all_node_genes = self.node_genes
        
        # subfunction to handle mutations with no global innovation number
        def gin_i():
            
            if gin: return gin.i()
            else: return 0
        
        # add new node
        if random.random() < add_node_prob:
            
            # select random connection gene and deactivate it
            old_connection = random.choice(self.connection_genes)
            old_connection.deactivate()
            
            # check if node already exists in NEAT gene list
            new_node = None
            for n in (n for n in all_node_genes if n.init_vector == old_connection.vector):
                    
                new_node = n
                in_innovation, out_innovation = n.init_innovations
                
            # generate new node, layer is average of start and end points of old connection
            if not new_node: 
                
                in_innovation, out_innovation = gin_i(), gin_i()
                
                new_node = NodeGene(num = len(all_node_genes) + 1, 
                                    layer = 0.5*sum(n.layer for n in old_connection.vector),
                                    init_vector = old_connection.vector,
                                    init_innovations = [in_innovation, out_innovation]
                                    )
                
                all_node_genes.append(new_node)
                        
            self.node_genes.append(new_node)
            
            # generate connection into new node with weight 1
            in_connection = new_node.add_connection(old_connection.vector[0],
                                                    weight = 1,
                                                    innovation = in_innovation
                                                    )   
            
            # generate connection out of new node with weight equal to old connection
            out_connection = new_node.add_connection(old_connection.vector[1],
                                                     weight = old_connection.weight,
                                                     innovation = out_innovation
                                                     )  
            
            # append connections to genome
            self.connection_genes += [in_connection, out_connection]
    
        # add connection
        if random.random() < add_connection_prob:
            
            try: 
                
                # choose random node that is not connected to all nodes in other layers
                node1 = random.choice([n for n in self.node_genes if len([c for c in self.connection_genes if c.vector.count(n) > 0]) < len([m for m in self.node_genes if m.layer != n.layer])])
                
                # choose second random node that is unconnected to the first and in a different layer
                node1_connected_nodes = [c.vector[1-c.vector.index(node1)] for c in self.connection_genes if node1 in c.vector]
                node2 = random.choice([n for n in self.node_genes if n.layer != node1.layer and n not in node1_connected_nodes])
                
                # ensure node1 has lower layer number than node2
                if node1.layer > node2.layer: vector = [node2, node1]
                else: vector = [node1, node2]
                
                # check if new connection vector is identical to a previous NEAT innovation
                if params and vector not in params.connection_vectors: 
                    
                    innovation = gin_i()
                    params.connection_vectors.append(vector)
                    
                elif params: innovation = params.connection_vectors.index(vector)
                else: innovation = 0
                
                # generate connection between them
                new_connection = node1.add_connection(node2, 
                                                      weight = 2*random.random() - 1,
                                                      innovation = innovation
                                                      )
                
                self.connection_genes.append(new_connection)
             
            # catches error given if there are no two unconnected nodes
            except IndexError: pass
        
        # perturb weights
        for connection in (c for c in self.connection_genes if random.random() < alt_weight_prob):
            
            # either perturb weight by gaussian noise or select random new value in range [-1, 1]
            if alt_weight_gauss: 
                
                # if no gaussian std defined, choose random value between [-0.5, 0.5]
                if not alt_weight_std: alt_weight_std = random.random() - 0.5
                
                new_weight = random.gauss(connection.weight, alt_weight_std)
                
                # prevent new weight exceeding range [-1, 1]
                if new_weight < -1: new_weight = -1
                elif new_weight > 1: new_weight = 1
            
            else: new_weight = 2*random.random() - 1
            
            # replace existing connection with new differently weighted connection gene object
            index = self.connection_genes.index(connection)
            self.connection_genes[index] = connection.copy(weight=new_weight) 
            
        # alter biases for nodes not in input layer
        for node in (n for n in self.node_genes if n.layer != 0 and random.random() < alt_bias_prob):
            
            # adds bias if node not in bias list, otherwise deletes node from bias list
            self.alt_bias(node)
        

###############################################################################

'''
input1 = NodeGene(num=1, layer=0)
input2 = NodeGene(num=2, layer=0)
input3 = NodeGene(num=3, layer=0)
input4 = NodeGene(num=4, layer=0)
hidden5 = NodeGene(num=5, layer=0.5)
output6 = NodeGene(num=6, layer=1)
output7 = NodeGene(num=7, layer=1)

hidden8 = NodeGene(num=8, layer=0.75)
hidden9 = NodeGene(num=9, layer=0.25)

con1 = input1.add_connection(output6, weight=0.1)
con2 = input2.add_connection(output6, weight=0.2)
con3 = input3.add_connection(output7, weight=0.3)
con4 = input4.add_connection(output7, weight=0.4)
#con5 = hidden5.add_connection(output7, weight=0.5)

genome = Genome(connection_genes=[con1, con2, con3, con4],
                node_genes=[input1, input2, input3, input4,
                            hidden5, hidden8, hidden9,
                            output6, output7]
                )

genome.compile_network()

print('nodes: ', [n.num for n in genome.node_genes])
genome.node_genes.append(NodeGene(num=10, layer=0.1))
print('nodes: ', [n.num for n in genome.node_genes])

genome.compile_network()
'''
'''
# example:
input1 = NodeGene(num=1, layer=0)
input2 = NodeGene(num=2, layer=0)
input3 = NodeGene(num=3, layer=0)
input4 = NodeGene(num=4, layer=0)
hidden5 = NodeGene(num=5, layer=0.5)
output6 = NodeGene(num=6, layer=1)
output7 = NodeGene(num=7, layer=1)

print(input3.type)

con1 = input1.add_connection(hidden5, weight=0.1)
con2 = input2.add_connection(hidden5, weight=0.2)
con3 = input3.add_connection(hidden5, weight=0.3)
con4 = input4.add_connection(output7, weight=0.4)
con5 = hidden5.add_connection(output7, weight=0.5)
con6 = hidden5.add_connection(output7, weight=0.6)

activation=hidden_layer_activations[random.randint(0,len(hidden_layer_activations)-1)]
optimizer=model_optimizers[random.randint(0,len(model_optimizers)-1)]

genome = Genome(connection_genes=[con1, con2, con3, con4, con5],
                node_genes=[input1, input2, input3, input4,
                            hidden5,
                            output6, output7]
                )

genome.get_node_info()

NN = genome.compile_network()
NN.get_weights(_print=True) 

#gin = GIN()

genome.mutate(gin=None)
genome.get_node_info() 
NN = genome.compile_network()
# NN.get_weights(_print=True) 
'''