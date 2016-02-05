# Model 
This is an implementation of the deep residual network used
for [Mini-Places2](http://6.869.csail.mit.edu/fa15/project.html) as
described in [He et. al., "Deep Residual Learning for Image
Recognition"](http://arxiv.org/abs/1512.03385).  The model is
structured as a very deep network with skip connections designed to
have convolutional parameters adjusting to residual activations. The
training protocol uses minimal pre-processing (mean subtraction) and
very simple data augmentation (shuffling, flipping, and cropping).
All model parameters (even batch norm parameters) are updated using
simple stochastic gradient descent with weight decay. The learning
rate is dropped only twice (at 90 and 135 epochs in the paper).

### Acknowledgments
Many thanks to Dr. He and his team at MSRA for their helpful input in
replicating the model as described in their paper.

### Model script
The model train script is included at ([miniplaces_msra.py](./miniplaces_msra.py)).

### Trained weights
The trained weights file can be downloaded from AWS
([miniplaces_msra_e66.pkl](https://s3-us-west-1.amazonaws.com/nervana-modelzoo/miniplaces_msra_e66.pkl))

### Performance
Training this model with the options described below should be able to achieve roughly 17.5% top-5
error using only mean subtraction, random cropping, and random flips. With multiscale evaluation (see the evaluation script),
the model should achieve roughly 14.6% top-5 error.

## Instructions
This script was tested with [neon version 1.2](https://github.com/NervanaSystems/neon/tree/v1.2.0).
Make sure that your local repo is synced to this commit and run the [installation
procedure](http://neon.nervanasys.com/docs/latest/user_guide.html#installation) before proceeding.
Commit SHA for v1.2 is  `385483881ee1fe1f0445fc78d7edf5b8ddc5c8c5`

This example uses the `ImageLoader` module to load the images for consumption while applying random
cropping, flipping, and shuffling.  Prior to beginning training, you need to write out the padded
mini-places2 images into a macrobatch repository. See [miniplaces_batchwriter.sh](./miniplaces_batchwriter.sh).

Note that it is good practice to choose your `data_dir` to be local to your machine in order to
avoid having `ImageLoader` module perform reads over the network.

Once the batches have been written out, you may initiate training:
```
miniplaces_msra.py -r 0 -vv \
    --log <logfile> \
    --epochs 80 \
    --save_path <model-save-path> \
    --eval_freq 1 \
    --backend gpu \
    --data_dir <path-to-saved-batches>
```

If you just want to run evaluation, you can use the much simpler script that loads the serialized
model and evaluates it on the validation set:

```
miniplaces_eval.py -vv --model_file <model-save-path>
```
