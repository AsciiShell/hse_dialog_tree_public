def make_model(loss):
    inp = layers.Input(shape=(None, 512))
    x = inp
    x = layers.LSTM(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(512)(x)
    model = Model(inputs=inp, outputs=x, name='03_recurrent')
    model.compile(loss=loss, optimizer='adam', metrics=['CosineSimilarity', 'MSE'])
    model.summary()
    return model

make_experiment(make_model, 'LSTM128 BN ReLU Dense512', exps)

def make_model(loss):
    inp = layers.Input(shape=(None, 512))
    x = inp
    x = layers.Bidirectional(layers.LSTM(128))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(512)(x)
    model = Model(inputs=inp, outputs=x, name='03_recurrent')
    model.compile(loss=loss, optimizer='adam', metrics=['CosineSimilarity', 'MSE'])
    model.summary()
    return model

make_experiment(make_model, 'BiLSTM128 BN ReLU Dense512', exps)

def make_model(loss):
    inp = layers.Input(shape=(None, 512))
    x = inp
    x = layers.LSTM(512)(x)
    model = Model(inputs=inp, outputs=x, name='03_recurrent')
    model.compile(loss=loss, optimizer='adam', metrics=['CosineSimilarity', 'MSE'])
    model.summary()
    return model

make_experiment(make_model, 'LSTM512', exps)

def make_model(loss):
    inp = layers.Input(shape=(None, 512))
    x = inp
    x = layers.Bidirectional(layers.LSTM(256))(x)
    model = Model(inputs=inp, outputs=x, name='03_recurrent')
    model.compile(loss=loss, optimizer='adam', metrics=['CosineSimilarity', 'MSE'])
    model.summary()
    return model

make_experiment(make_model, 'BiLSTM256', exps)

def make_model(loss):
    inp = layers.Input(shape=(None, 512))
    x = inp
    x = layers.GRU(512)(x)
    model = Model(inputs=inp, outputs=x, name='03_recurrent')
    model.compile(loss=loss, optimizer='adam', metrics=['CosineSimilarity', 'MSE'])
    model.summary()
    return model

make_experiment(make_model, 'GRU512', exps)

def make_model(loss):
    inp = layers.Input(shape=(None, 512))
    x = inp
    x = layers.Bidirectional(layers.GRU(256))(x)
    model = Model(inputs=inp, outputs=x, name='03_recurrent')
    model.compile(loss=loss, optimizer='adam', metrics=['CosineSimilarity', 'MSE'])
    model.summary()
    return model

make_experiment(make_model, 'BiGRU256', exps)

def make_model(loss):
    def self_attention(x, f=512, k=1):
        x = layers.Conv1D(filters=f, kernel_size=k, padding='same')(x)
        x = layers.Attention(use_scale=True)([x, x])
        return x
    inp = layers.Input(shape=(None, 512))
    x = inp
    x = self_attention(x, 64, 1)
    x = self_attention(x, 128, 2)
    x = self_attention(x, 512, 3)
    x = layers.GlobalAveragePooling1D()(x)
    model = Model(inputs=inp, outputs=x, name='03_recurrent')
    model.compile(loss=loss, optimizer='adam', metrics=['CosineSimilarity', 'MSE'])
    model.summary()
    return model

make_experiment(make_model, 'SA(f,k)=(conv(f,k) Attention) SA(64,1) SA(128,2) SA(512,3) GAP', exps)

def make_model(loss):
    def self_attention(x, f=512, k=1):
        x = layers.Conv1D(filters=f, kernel_size=k, padding='same')(x)
        x = layers.Attention(use_scale=True)([x, x])
        return x
    inp = layers.Input(shape=(None, 512))
    x = inp
    x = self_attention(x, 64, 5)
    x = self_attention(x, 128, 3)
    x = self_attention(x, 512, 2)
    x = layers.GlobalAveragePooling1D()(x)
    model = Model(inputs=inp, outputs=x, name='03_recurrent')
    model.compile(loss=loss, optimizer='adam', metrics=['CosineSimilarity', 'MSE'])
    model.summary()
    return model

make_experiment(make_model, 'SA(f,k)=(conv(f,k) Attention) SA(64,5) SA(128,3) SA(512,2) GAP', exps)

def make_model(loss):
    inp = layers.Input(shape=(None, 512))
    x = inp
    x = layers.Bidirectional(layers.LSTM(128,))(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Attention(use_scale=True)([x, x]) 
    x = layers.Dense(512)(x)
    
    model = Model(inputs=inp, outputs=x, name='03_recurrent')
    model.compile(loss=loss, optimizer='adam', metrics=['CosineSimilarity', 'MSE'])
    model.summary()
    return model

make_experiment(make_model, 'BiLSTM(128) Drop0.1 Dense128 Relu Attention Dense512', exps)

def make_model(loss):
    inp = layers.Input(shape=(None, 512))
    x = inp
    x = layers.Bidirectional(layers.LSTM(128,))(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Attention(use_scale=True)([x, x])
    x = layers.Attention(use_scale=True)([x, x]) 
    x = layers.Dense(512)(x)
    
    model = Model(inputs=inp, outputs=x, name='03_recurrent')
    model.compile(loss=loss, optimizer='adam', metrics=['CosineSimilarity', 'MSE'])
    model.summary()
    return model

make_experiment(make_model, 'BiLSTM(128) Drop0.1 Dense128 Relu Attention Attention Dense512', exps)

def make_model(loss):
    inp = layers.Input(shape=(None, 512))
    x = inp
    x = layers.Bidirectional(layers.LSTM(128,))(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Attention(use_scale=True)([x, x])
    x = layers.Attention(use_scale=True)([x, x]) 
    x = layers.Attention(use_scale=True)([x, x]) 
    x = layers.Dense(512)(x)
    
    model = Model(inputs=inp, outputs=x, name='03_recurrent')
    model.compile(loss=loss, optimizer='adam', metrics=['CosineSimilarity', 'MSE'])
    model.summary()
    return model

make_experiment(make_model, 'BiLSTM(128) Drop0.1 Dense128 Relu Attention Attention Attention Dense512', exps)

def make_model(loss):
    inp = layers.Input(shape=(None, 512))
    x = inp
    x = layers.Attention(use_scale=True)([x, x])
    x = layers.Bidirectional(layers.LSTM(128,))(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(512)(x)
    
    model = Model(inputs=inp, outputs=x, name='03_recurrent')
    model.compile(loss=loss, optimizer='adam', metrics=['CosineSimilarity', 'MSE'])
    model.summary()
    return model

make_experiment(make_model, 'Attention BiLSTM(128) Drop0.1 Dense128 Relu Dense512', exps)

def make_model(loss):
    inp = layers.Input(shape=(None, 512))
    x = inp
    x = layers.Attention(use_scale=True)([x, x])
    x = layers.Attention(use_scale=True)([x, x])
    x = layers.Bidirectional(layers.LSTM(128,))(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(512)(x)
    
    model = Model(inputs=inp, outputs=x, name='03_recurrent')
    model.compile(loss=loss, optimizer='adam', metrics=['CosineSimilarity', 'MSE'])
    model.summary()
    return model

make_experiment(make_model, 'Attention Attention BiLSTM(128) Drop0.1 Dense128 Relu Dense512', exps)

def make_model(loss):
    inp = layers.Input(shape=(None, 512))
    x = inp
    x = layers.Attention(use_scale=True)([x, x])
    x = layers.Attention(use_scale=True)([x, x])
    x = layers.Attention(use_scale=True)([x, x])
    x = layers.Bidirectional(layers.LSTM(128,))(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(512)(x)
    
    model = Model(inputs=inp, outputs=x, name='03_recurrent')
    model.compile(loss=loss, optimizer='adam', metrics=['CosineSimilarity', 'MSE'])
    model.summary()
    return model

make_experiment(make_model, 'Attention Attention Attention BiLSTM(128) Drop0.1 Dense128 Relu Dense512', exps)

def make_model(loss):
    inp = layers.Input(shape=(None, 512))
    x = inp
    x = layers.Attention(use_scale=True)([x, x])
    x = layers.Bidirectional(layers.LSTM(128,))(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(512)(x)
    
    model = Model(inputs=inp, outputs=x, name='03_recurrent')
    model.compile(loss=loss, optimizer='adam', metrics=['CosineSimilarity', 'MSE'])
    model.summary()
    return model

make_experiment(make_model, 'Attention BiLSTM(128) Drop0.1 Dense512', exps)

def make_model(loss):
    inp = layers.Input(shape=(None, 512))
    x = inp
    x = layers.Attention(use_scale=True)([x, x])
    x = layers.Attention(use_scale=True)([x, x])
    x = layers.Bidirectional(layers.LSTM(128,))(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(512)(x)
    
    model = Model(inputs=inp, outputs=x, name='03_recurrent')
    model.compile(loss=loss, optimizer='adam', metrics=['CosineSimilarity', 'MSE'])
    model.summary()
    return model

make_experiment(make_model, 'Attention Attention BiLSTM(128) Drop0.1 Dense512', exps)

def make_model(loss):
    inp = layers.Input(shape=(None, 512))
    x = inp
    x = layers.Attention(use_scale=True)([x, x])
    x = layers.Attention(use_scale=True)([x, x])
    x = layers.Attention(use_scale=True)([x, x])
    x = layers.Bidirectional(layers.LSTM(128,))(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(512)(x)
    
    model = Model(inputs=inp, outputs=x, name='03_recurrent')
    model.compile(loss=loss, optimizer='adam', metrics=['CosineSimilarity', 'MSE'])
    model.summary()
    return model

make_experiment(make_model, 'Attention Attention Attention BiLSTM(128) Drop0.1 Dense512', exps)


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = layers.MultiHeadAttention(num_heads, d_model)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        attn_output = self.mha(x, x, x)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2
        
def make_model(loss):
    inp = layers.Input(shape=(None, 512))
    x = inp
    x = EncoderLayer(512, 16, 8)(x)     
    x = layers.GlobalAveragePooling1D()(x)
    model = Model(inputs=inp, outputs=x, name='03_recurrent')
    model.compile(loss=loss, optimizer='adam', metrics=['CosineSimilarity', 'MSE'])
    model.summary()
    return model

make_experiment(make_model, 'EncoderLayer(512,16,8) GAP', exps, 10)

def make_model(loss):
    inp = layers.Input(shape=(None, 512))
    x = inp
    x = EncoderLayer(512, 16, 8)(x) 
    x = EncoderLayer(512, 16, 8)(x) 
    x = EncoderLayer(512, 16, 8)(x) 
    x = EncoderLayer(512, 16, 8)(x)     
    x = layers.GlobalAveragePooling1D()(x)
    model = Model(inputs=inp, outputs=x, name='03_recurrent')
    model.compile(loss=loss, optimizer='adam', metrics=['CosineSimilarity', 'MSE'])
    model.summary()
    return model

make_experiment(make_model, 'EncoderLayer(512,16,8)x4 GAP', exps, 10)