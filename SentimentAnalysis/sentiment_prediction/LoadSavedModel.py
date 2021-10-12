
import os
import tensorflow as tf
SAVED_MODEL = os.getcwd()+'\saved_models\model_10'
class LoadSavedModelSingleton(object):

    _instance = None

    def __new__(self):
        if self._instance is None:
            print('Creating the object')
            self._instance = super(LoadSavedModelSingleton, self).__new__(self)
            self.model = tf.keras.models.load_model(SAVED_MODEL)
            # Put any initialization here.
        return self._instance