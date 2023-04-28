
Fruit Ripening Process - v2 2022-03-10 3:46pm
==============================

This dataset was exported via roboflow.com on November 2, 2022 at 7:02 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

It includes 18074 images.
Fruit are annotated in Multi-Class Classification format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 416x416 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Randomly crop between 0 and 20 percent of the image
* Random brigthness adjustment of between -25 and +25 percent
* Salt and pepper noise was applied to 5 percent of pixels


