import numpy as np
import keras.backend as K




np_x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype='float32')

with K.get_session() as sess:
    sess.as_default()
    x = K.ones((2,2), dtype='float32', name='x')
    eye = K.eye(2, dtype='float32', name='eye')

    K.set_value(x, np_x)
    x = K.print_tensor(x, message='x is ') # will print the tensor if the result returned is being used
    print(K.get_value(eye / x))

