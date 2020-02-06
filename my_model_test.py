import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model

model = load_model('my_model.h5')
test_image = image.load_img('final-test-images/cat2.jpg', target_size = (224, 224))
test_image = np.asarray(test_image)
plt.show(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)

if result[0][0] > 0.5:
    prediction = 'cat'
else:
    prediction = 'dog'
print(prediction)
print("=" * 50)

print(result)
