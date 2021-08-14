# Gesture Recogniton for the Smart Applicance System

This is an experimentation notebook , with 3D convolution based neural network plus 2D
convolution and RNN based neural network built from Scratch for the 
video classification Task.

## Experimentation Environment.
* Tensorflow == 2.5.0
* Nvidia T4 - 16 GB RAM
* Intel(R) Xeon(R) CPU @ 2.20GHz, 4 Core, 15 GB RAM
* Cuda 10.1
* Due to the privacy policy, the Dataset details cannot be made public.

Refer requirements.txt file for all the dependencies.

## Network Architectures

### 3D Convolution Architecture

• Version 1

    ◦ Two Conv3D layers, one with 32 filters of 3x3x3 volumetric feature map and the other one 64 filters with 3x3x32 
      volumetric feature map. Both of them are followed by Batch_Norm and then 3D max-pooling of feature map 2x2x2.
    ◦ This variant uses only one feature map 3x3, which means the spatial region selecting to form the convoluted 
      feature is a small region only, learning the micro patterns in the images across the temporal dimensions.
    ◦ This model does not seem to be having the capacity to learn high-level spatial regions since both layers have 
      the same feature map dimensions.
    ◦ The second layer has 64 filters, therefore each filter is learning a different representation of the input 
      feature map with the volumetric region of 3x3x32. Therefore, there is variation learned by the second layer, 
      but with the same volumetric exposure.

• Version 2

    ◦ Version 2 adds two more layers on top of version 1. We add another layer of 3x3x32 with 64 filters and then we 
      have added a 5x5x64 dimensional volumetric filter with a total of 128 feature maps. This introduces a higher 
      level of feature map learning with more variation learned over the volumetric tensor, compared to the previous 
      variant. We cannot add another level of higher spatial feature learning kernel due to the memory constraints. 
      Overall this variant has a higher learning capacity compared to version 1.

### 2D CNN-RNN Architecture

We are considering only unidirectional RNN architecture for obvious reasons.

• Version - 1 [LSTM]

    ◦ Taking inspiration from the 3D convolution network, here too, we are using the 2 2D convolution layer with 32 and 
      64 filters each of size 3x3, then batch normalization and then Max pooling operation of 2x2.
    ◦ The reason to go with 32 3x3 and then 64 3x3 filters is to gradually increase the feature variation in the network
      going from 32 to 64.
    ◦ Then GlobalMaxpooling operation is used to get the network output at the step as the flattened one.
    ◦ All the layers above mentioned are time distributed, nice we are feeding a sequence of frames to the model.
    ◦ One LSTM layer with 512 is added followed by a dense layer of 512 neurons and then the softmax layer. 

• Version - 1 [GRU]

    ◦ This version follows the exact same architecture as the above only difference, we replace the LSTM with the 
    GRU layer, to see if there is any difference in the network evaluation.

• Version - 2 [LSTM]

    ◦ In this version, we added another layer in the feature extractor i.e. the 2D CNN component of the model. 
      A layer with 128 filters of the dimension 3x3 is added. This further increases the variation in the feature 
      space of the extracted features by the CNN component. This is followed by batch normalization 
      and then max-pooling 2D.
    ◦ Next, we have the global max-pooling 2D, similar to what we had in the version-1 architecture. This and all the 
      layers above are time distributed.
    ◦ Then we have added 2 LSTM layers with 512 cells followed by the Dense layer with 256 neurons, which is then 
      followed by the softmax layer.
    ◦ This variant has 2 LSTMS, which helps to add a prolonged memory component to the sequence learning capability 
      of the network.

• Version - 2[GRU]

    ◦ This version follows the same architecture as the above, except the fact, we replace the LSTM with GRU 
      layers, to check if there is any performance gain with the GRU layer added.

## Results

Experiment  | Architecture | Hyper-Parameter Setting | Validation Set Accuracy| Experiment Detail|
--- | --- | --- | --- | --- |
1 | Conv3D Version 1 | Batch size = 4 Epochs = 10 Learning rate = 0.001 Optimizer = Adam | 0.5800 |Starting with Low batch size as the rudimentary run of the model.The model is expected to have a low performance since it is the most simplistic architecture that we have developed and we are training it for 10 epochs only.
2 | Conv3D Version 1 | (16,10,0.001, Adam) | 0.6214 | The increase in the batch size to 16 provides a smoother gradient update in the training and therefore we see an increase in the validation accuracy of the model.
3 | Conv3D Version 1 | (32, 10, 0.001, Adam) | OOM | Due to the GPU memory limitations, we are getting OOM on this particular hyperparameter setting.
4 | Conv3D Version 1 | (23, 10, 0.001, Adam) | 0.600 | Maximum batch size the GPU can hold for the current network architecture. Even with the increased batch size, there is no significant change in the validation accuracy of the model, it is a pure case of underfitting, need to increase the model complexity and rerun the experiment.
5 | Conv3D Version 2 | (22, 10, 0.001, Adam) | 0.6571 | There is a significant increase in the validation accuracy, but it is not that significant to say, the model has started taking large steps towards the minima.
6 | Conv3D Version 2 | (22, 20, 0.01, Adam) | 0.6571 | In this run, we ran the training for a longer period of time to see, where we hit the plateau in terms of learning and reducing the starting learning rate.
7 | Conv3D Version 2 | (22, 40, 0.01, Adam) | 0.6571 | In this run, we further try to train the model for a longer duration, expecting it to do better than the previous run, but as we see, the model with the current dataset and the model capacity has reached its learning threshold. It is giving the same output as the previous run.
8 | Conv3D Version 2 | (22, 40, 0.01, Nadam) | 0.5857 | Adam uses Vanilla momentum implementation, while the NAG Nesterov accelerated Gradient is the superior version of the same momentum. Trying a different optimizer is another strategy we experimented with, but the results got deteriorated. This means we need to stick to the Adam optimizer itself.
9 | CNN-RNN-1(LSTM) | (32, 10, 0.01, Adam) | OOM | The network parameters space doesn’t allow a batch of 32 videos to be fed in a go, therefore, leads to the OOM error.
10 | CNN-RNN-1(LSTM) | (16, 10, 0.01, Adam) | 0.5500 | Starting with the first run for the CNNRNN network, there is still room for improvement in the validation accuracy of the model. Next, run it for a longer period of time.
11 | CNN-RNN-1(LSTM) | (16, 20, 0.01, Adam) | 0.5857 | In this run, we observe an improvement in the validation accuracy, but we would want to keep the model running further and check if there is any gain in the performance.
12 | CNN-RNN-1(LSTM) | (16, 40, 0.01, Adam) | 0.6929 | This run, beat the best model till now, which is the 3D Convolution - version1, with 0.3571. This becomes our current best mode.
13 | CNN-RNN-1(LSTM) | (16, 40, 0.01, Nadam) | 0.6214 | Trying a different optimizer to see if there is any improvement in the validation evaluation. We observe, there is degradation in the performance, pointing to the fact that Nadam is not fit for our case.
14 | CNN-RNN-2(LSTM) | (16, 10, 0.01, Adam) | 0.6929 | This version of the architecture has a larger parameter space with more convolution layers and more LSTM layers, it is expected to have a complex learning function. Therefore, in 10 epochs of the run, it reached the best validation accuracy that we have observed so far.
14 | CNN-RNN-2(LSTM) | (16, 20, 0.01, Adam) | 0.6314 | Increasing the training time to 20 epochs to see, if there is gain in the validation accuracy. It is not the case. There is a degradation we see, this is because of the stochasticity in the process.
15 | CNN-RNN-2(LSTM) | (16, 40, 0.01, Adam) | 0.6571 | In this run, we further try to run till 40 epochs and see, if there are any gains in the validation performance. It is doing better than most of the previous architecture, but cannot beat the best-known run.
16 | CNN-RNN-2(LSTM) | (16, 40, 0.01, Nadam) | 0.6929 | In this run, we try another different optimizer. This time, we see Nadam has performed on par with Adam and is equalling the best validation accuracy known till now.
17 | CNN-RNN-1(GRU) | (32, 10, 0.01, Adam) | OOM | GPU memory constraint, leading to out-of-memory for the hyperparameter setting.
18 | CNN-RNN-1(GRU) | (16, 10, 0.01, Adam) | 0.5857 | This architecture uses the GRU layer instead of the LSTM layer. We observe it is par with the LSTM(0.2500) for the same hyperparameter setting run model and doing slightly better than that.
19 | CNN-RNN-1(GRU) | (16, 20, 0.01, Adam) | 0.6571 | Increasing the batch from the previous run, we observe that we get gains in the learning capacity of the model.
20 | CNN-RNN-1(GRU) | (16, 40, 0.01, Adam) | 0.7286 | Further increasing the training time, we observe, the model has hit the new best result for the validation accuracy. This shows GRU is performing better than LSTM and the 3D Convolution Network for the use case.
21 | CNN-RNN-1(GRU) | (16, 40, 0.01, Nadam) | 0.6571 | Changing the optimizer, we observe that, its validation set performance has degraded, which proves, Nadam is not a good fit with GRU. 
22 | CNN-RNN-2(GRU) | (16, 10, 0.01, Adam) | 0.500 | This architecture has more convolution plus the GRU layers and thus a higher learning capacity. We observe the first run does not give us the expected validation accuracy which we got with LSTM(0.3929).
23 | CNN-RNN-2(GRU) | (16, 20, 0.01, Adam) | 0.6571 | Further increasing the training runs, give us an increase in the validation accuracy, which is on par with LSTM(0.3214) run for the same architecture + hyperparameter setting.
24 | CNN-RNN-2(GRU) | (16, 40, 0.01, Adam) | 0.6929 | Further, increasing the training time gives us further improvement in the validation accuracy of the network, which is the 2nd best till now. Considering the learning capacity of the model, it is worth training the model further and sees, if it gives us any improvements.
25 | CNN-RNN-2(GRU) | (16, 40, 0.01, Nadam) | 0.6571 | The performance of the network with Nadam optimizer, which results better than most fo the runs but is not the best, as we observe in the above runs, there using the Adam is a better and a safe in the final model.


