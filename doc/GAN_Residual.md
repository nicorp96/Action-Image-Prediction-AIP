# Conditioned GAN

## Concatenation:

In this method, the conditioning information (TCP coordinates) is concatenated with the input of the generator and/or discriminator. This additional information can be in the form of vectors.

Generator:
z′=[z,c]
z′=[z,c]
where z is the random noise vector and c is the conditioning vector (TCP coordinates).

Discriminator:

Similarly, the discriminator can also receive the conditioning information concatenated with the input.

## Conditional Batch Normalization:

In Conditional Batch Normalization, the batch normalization parameters (mean, variance) are computed conditionally based on the conditioning information.

## Auxiliary Classifier GAN (AC-GAN):

In AC-GANs, the discriminator not only outputs the probability of the data being real or fake but also tries to predict the conditioning information.

## Projection-Based Conditioning:

In this method, the conditioning information is projected using a learned linear mapping and then combined with the feature maps in the discriminator. The conditioning vector is projected to match the dimension of the feature map and added element-wise.

## Conditional Instance Normalization:

This method applies instance normalization conditioned on external information. Similar to conditional batch normalization, the normalization parameters are conditioned on the additional information (TCP coordinates).

## FiLM (Feature-wise Linear Modulation):

FiLM layers use the conditioning information to linearly modulate the activation maps in the generator and/or discriminator. Parameters of the affine transformatio (scaling and shifting) are functions of the conditioning information.


# GAN with Residual Blocks

1. Using Only Images in Residual Blocks:
   - Common Use Cases:
     - Tasks where maintaining the visual continuity and features of the image itself is most important.
     - Image super-resolution or denoising tasks where additional inputs like actions might not be relevant.
   - Advantages:
     - Focuses purely on spatial features, ensuring that the intrinsic details of the image are preserved and enhanced.

2. Combining Images with Actions in Residual Blocks:
   - Common Use Cases:
     - Tasks where the image needs to be transformed based on an external factor (like an action or an event).
     - Video frame prediction, robotic vision, and control tasks where the next image frame depends on both the current frame and the performed action.
   - Advantages:
     - Integrates contextual and dynamic information, allowing the network to generate more informed and relevant transformations.
     - Facilitates better modeling of time-dependent or action-dependent changes in the image.