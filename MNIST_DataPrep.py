import gzip

#create a file object with our data
f = gzip.open('train-images-idx3-ubyte.gz','r')

image_size = 28
num_images = 60000

#Not sure why we need this, but it makes the image display properly
f.read(16)

#Make a buffer of the content, then convert that to a numpy array
buf = f.read(image_size * image_size * num_images)
train_data = np.frombuffer(buf, dtype=np.uint8)
print('train_data type:',type(train_data))
print('train_data shape:',train_data.shape)

#Reshape array
train_data = train_data.reshape(num_images, image_size, image_size)
print('train_data shape:',train_data.shape)
print('first image shape:',train_data[0].shape)

#Plot the first image to verify it all worked
import matplotlib.pyplot as plt

image = train_data[0]
plt.imshow(image)
plt.show()

############################
# Same stuff for the labels:

f = gzip.open('train-labels-idx1-ubyte.gz','r')
f.read(8)
buf = f.read(60000)
train_labels = np.frombuffer(buf, dtype=np.uint8)
print('train_labels shape:',train_labels.shape)
print('first training image label:',train_labels[0])
