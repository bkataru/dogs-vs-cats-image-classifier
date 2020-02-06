import keras
from keras.models import Model
from keras.layers import Dense
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

train_gen = ImageDataGenerator()
training_set = train_gen.flow_from_directory(directory="dataset/training_set",target_size=(224,224))
test_gen = ImageDataGenerator()
test_set = test_gen.flow_from_directory(directory="dataset/test_set", target_size=(224,224))

from keras.applications.vgg16 import VGG16
vggmodel = VGG16(weights='imagenet', include_top=True)

print(vggmodel.summary())

for layers in (vggmodel.layers)[:19]:
    print(layers)
    layers.trainable = False

X= vggmodel.layers[-2].output
predictions = Dense(2, activation="softmax")(X)
model_final = Model(input = vggmodel.input, output = predictions)

model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
print(model_final.summary())

from keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint("vgg16_pretrained.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')
hist = model_final.fit_generator(generator= training_set, steps_per_epoch= 2, epochs= 100, validation_data= test_set, validation_steps=1, callbacks=[checkpoint,early])

print("pretrained_vgg16_class_indices", training_set.class_indices)
f = open("pretrained_vgg16_class_indices.txt", "w")
f.write(str(training_set.class_indices))
f.close()

import matplotlib.pyplot as plt

plt.plot(hist.history["accuracy"])
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()


