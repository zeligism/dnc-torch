# DNC in PyTorch
This is a reimplementation of the [Differentiable Neural Computer (DNC)](https://deepmind.com/blog/differentiable-neural-computers/) in PyTorch.
This implementation uses some functions from the latest version of PyTorch (0.4),
so make sure you update your PyTorch before running this code.

I wrote this code in a way that is accessible to people who want to learn how the DNC works.
It is very well-documented as far as I'm concerned, and uses simple language and short-hand operations whenever possible
to improve readability and facilitate understanding, especially for beginners.
It is easier to see how the DNC works from a PyTorch implementation vs. a TensorFlow implementation
since the latter is written in a functional manner, whereas PyTorch follows a simpler procedural step-by-step approach.

## Running the code
If you want to run the code, you have to check the hyperparameters first in `training_configs.py` and see if you like them.
After that, you can simply run the code using:

```
python train.py
```

(I'm assuming you are using python 3. If not, please consider upgrading/adapting.)

## Dataset
The code uses a repeat-copy dataset that simply gives a sequence of words that are `num_bits` long,
where `num_bits` is a given parameter.
The sequence length has a lower and upper range, which are, again, parameters given by the user.
The input starts with a signal from the start-channel, and ends with a signal in the repeat-channel.
The repeat-channel indicates the number of times that this sequence has to be repeated as an output.
The range of the number of repeats possible is given by the user as a parameter as well.
After the predictions are made, we expect the DNC to output a signal in the end-channel.

One thing I should mention is that the repeat-channel in my implementation is a one hot encoding of the repeat number,
meaning that if the maximum number of allowed repeat is 4, then there would be 5 repeat-channels,
each signifying a number between 0 and 4, inclusive. Yes, I included 0 repeats as well.
I think it's difficult for a neural network to tell the difference between a 3 and a 4 in a counting sense,
which is why I used one hot encoding for representing the number of repeats.

In addition to that, I calculated the loss differently. I really didn't understand how DeepMind calculated their cross entropy loss.
I only implemented a loss that made sense to me, which is simply the sum of distances between prediction and target at each time step.

## Does it run, though?
Yes...

Well, I like to believe that it does, since I was able to get some good results out of it.
Using my code, you will be able to output signals that, after rounding, will correspond to the right bit almost perfectly.
In other words, if you round the output to the nearest integer,
then the predicted output will be perfectly similar to the true output almost all of the time.
This happens after about 10,000 examples (with a batch size of 8).
