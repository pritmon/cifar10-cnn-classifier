This folder holds a few test images to run predictions on.

To fill it with real CIFAR-10 examples, run from the project root:

    python make_sample_images.py

That saves files like cat.png, dog.png, airplane.png, ship.png, frog.png here.

You can also drop in your own JPG/PNG images and predict on them:

    python predict.py sample_images/my_photo.jpg
