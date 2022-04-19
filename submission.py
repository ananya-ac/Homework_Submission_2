#!/usr/bin/python

import random
from typing import Callable, Dict, List, Tuple, TypeVar

from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    if x=='':
        raise Exception("No input")
    dict_count={}
    for word in set([word for word in x.split()]):
        dict_count[word]=x.count(word)

    return dict_count
    # END_YOUR_CODE


############################################################
# Problem 3b: stochastic gradient descent

T = TypeVar('T')


def learnPredictor(trainExamples: List[Tuple[T, int]],
                   validationExamples: List[Tuple[T, int]],
                   featureExtractor: Callable[[T], FeatureVector],
                   numEpochs: int, eta: float) -> WeightVector:
    '''
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Notes:
    - Only use the trainExamples for training!
    - You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch.
    - The predictor should output +1 if the score is precisely 0.
    '''
    weights = {}  # feature => weight
    
    def get_prediction(x):
        """ takes in example x, returns prediction y """
        score=dotProduct(weights,x)   
        prediction=1 if score>=0 else -1
        return prediction
   

    def update_weights(x,y,lr):
        update={}
        if get_prediction(x)*y<1:
            increment(update,-y,x) 
            increment(weights,-lr,update)
    
        

    # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
    x_trains=[featureExtractor(t[0]) for t in trainExamples]
    y_trains=[t[1] for t in trainExamples]
    x_validation=[featureExtractor(t[0]) for t in validationExamples]
    y_validation=[t[1] for t in validationExamples]
    trains=list(zip(x_trains,y_trains))
    
    for i in range(numEpochs):
        randomized_trains=random.sample(trains,len(trains))
        total_loss=0
        
        for x,y in randomized_trains:
            loss=max(0,1-get_prediction(x)*y)
            update_weights(x,y,eta)
            total_loss+=loss
            
        total_loss=total_loss/len(trainExamples)
        train_error=evaluatePredictor(list(zip(x_trains,y_trains)),get_prediction)
        validation_error=evaluatePredictor(list(zip(x_validation,y_validation)),get_prediction)
        print("total_loss after {} epochs={}".format(i,total_loss))
        print("train_error after {} epochs={}".format(i,train_error))
        print("validation_error after {} epochs ={}".format(i,validation_error))
       
        print("")
        print("")
            
        
        
    
    
    
    # END_YOUR_CODE
    return weights


############################################################
# Problem 3c: generate test case


def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
     random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a score for the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    # y should be 1 if the score is precisely 0.

    # Note that the weight vector can be arbitrary during testing.
    def generateExample() -> Tuple[Dict[str, int], int]:
        # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
        total_features=list(weights.keys())
        example_features=random.sample(total_features,random.sample(range(1,len(total_features)+1),1)[0])
        phi={example_feature:random.random() for example_feature in example_features}
        y=1 if dotProduct(weights,phi)>=0 else -1
        # END_YOUR_CODE
        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 3d: character features


def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x: str) -> Dict[str, int]:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
          def extract(x: str) -> Dict[str, int]:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        x=x.replace(' ','')
        n_grams=[]
        for i in range(len(x)-(n-1)):
            n_grams.append(x[i:i+n])
        dict_features={gram:x.count(gram) for gram in n_grams}
        return dict_features
        # END_YOUR_CODE

    return extract


############################################################
# Problem 3e:


def testValuesOfN(n: int):
    '''
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    '''
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples,
                             validationExamples,
                             featureExtractor,
                             numEpochs=20,
                             eta=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights,
                        'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(
        trainExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(
        validationExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" %
           (trainError, validationError)))


############################################################
# Problem 5: k-means
############################################################




def kmeans(examples: List[Dict[str, float]], K: int,
           maxEpochs: int) -> Tuple[List, List, float]:
    '''
    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxEpochs: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 28 lines of code, but don't worry if you deviate from this)
    if(K>len(examples)):
        raise Exception("invalid K")
        
    centers=random.sample(examples,K) #initializing the centroids
    i=0
    assignments=[]
    previous_epoch_loss=0
    while(i in range(maxEpochs) ):
        
        assignments_with_loss=update_assignments_get_loss(examples,centers,assignments)
        assignments=[assignment_with_loss[0] for assignment_with_loss in assignments_with_loss]
        loss_per_example=[assignment_with_loss[1] for assignment_with_loss in assignments_with_loss]
        total_reconstruction_loss=sum(loss_per_example)
        if total_reconstruction_loss==previous_epoch_loss:
            print("converged at Epoch:"+str(i))
            break
        previous_epoch_loss=total_reconstruction_loss
        centers=update_centroids(examples,assignments,K)
        i+=1
        print("Loss at Epoch"+str(i)+"="+str(total_reconstruction_loss))
       
        
    
    return centers,assignments,total_reconstruction_loss
 
    
    # END_YOUR_CODE



#Helper funtions

def update_assignments_get_loss(train_examples:List[Dict[str, float]],
                      centroids:List[Dict[str,float]], assigns):
    
    return [(get_min_centroid_with_loss(centroids,example)) for example in train_examples]

            
def get_min_centroid_with_loss(centroids,example):
    
    min_pos=0
    min_dist=squared_distance(example,centroids[min_pos])
   
    for j,centroid in enumerate(centroids):
        dist=squared_distance(example,centroid)
        if dist<min_dist:
            min_dist,min_pos=dist,j
     
    return min_pos,min_dist

def update_centroids(train_examples, assigns,num_centroids):

    sum_centroids=[{} for i in range(num_centroids)]
    new_centroids=[{} for i in range(num_centroids)]
    ct_centroids=[0 for i in range(num_centroids)]
   

    for i, (example,assign) in enumerate(zip(train_examples,assigns)):
        increment(sum_centroids[assign],1,example)
        ct_centroids[assign]+=1
   
    for new_centroid,ct,sum_centroid in zip (new_centroids,ct_centroids,sum_centroids):
        
        if ct==0:
            continue
        else:
            increment(new_centroid,1/ct,sum_centroid)
       
    
    
            
    return new_centroids

def squared_distance(d1, d2):
    """ returns squared distance between two sparse vectors d1 and d2"""
    d3={k:0 for k in set(d1.keys()).union(set(d2.keys()))}
    common_keys=set(d1.keys()).intersection(set(d2.keys()))
    keys_d1_minus_d2=set(d1.keys()).difference(set(d2.keys()))
    keys_d2_minus_d1=set(d2.keys()).difference(set(d1.keys()))
    for key in common_keys:
        d3[key]=d1[key]-d2[key]
    for key in keys_d1_minus_d2:
        d3[key]=d1[key]
    for key in keys_d2_minus_d1:
        d3[key]=-1*d2[key]
    
    return dotProduct(d3,d3)

    
    

                
            
        
                