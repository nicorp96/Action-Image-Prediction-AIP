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