from transformers import TFAutoModel
import tensorflow as tf
import tensorflow_addons as tfa


class BertSentiment:

    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)


    def load_model(self,model_path: str,do_lower_case=False):

        bert = TFAutoModel.from_pretrained('bert-base-cased')
        #bert = TFAutoModel.from_pretrained('princeton-nlp/sup-simcse-bert-base-uncased')#princeton-nlp/unsup-simcse-bert-base-uncased')
        #bert = TFAutoModel.from_pretrained('princeton-nlp/unsup-simcse-bert-base-uncased')
        # we can view the model using the summary method
        bert.summary()
    
        # two input layers, we ensure layer name variables match to dictionary keys in TF dataset
        input_ids = tf.keras.layers.Input(shape=(512,), name='input_ids', dtype='int32')
        mask = tf.keras.layers.Input(shape=(512,), name='attention_mask', dtype='int32')
        #entities = tf.keras.layers.Input(shape=(512,), name='entities', dtype='int32')

    
        # we access the transformer model within our bert object using the bert attribute (eg bert.bert instead of bert)
        embeddings = bert.bert(input_ids, attention_mask=mask)[1]  # access final activations (alread max-pooled) [1]
       
        # convert bert embeddings into 2 output classes
        #x = tf.keras.layers.Dense(1024, activation='relu')(embeddings)
        #y = tf.keras.layers.Dense(1, activation='sigmoid', name='outputs')(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(embeddings)
        y = tf.keras.layers.Dense(2, activation='softmax', name='outputs')(x)
        
        model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)

        # (optional) freeze bert layer
        model.layers[2].trainable = False

        # print out model summary
        model.summary()

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, decay=1e-6)
        loss = tf.keras.losses.CategoricalCrossentropy()
        acc = tf.keras.metrics.CategoricalAccuracy('accuracy')
        #f1 = tfa.metrics.F1Score(num_classes=1, threshold=0.8)
        #f1  = tf.keras.metrics.CategoricalAccuracy('f1') num_classes=1, threshold=0.5
        recall = tf.keras.metrics.Recall()
        precision = tf.keras.metrics.Precision()

        model.compile(optimizer=optimizer, loss=loss, metrics=[acc, recall, precision])
    
        return model

