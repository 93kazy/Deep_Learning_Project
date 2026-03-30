import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers, regularizers
import gc
import golois

planes  = 31
moves   = 361
N       = 10000
batch   = 256
filters = 64
epochs  = 3000

input_data = np.random.randint(2, size=(N, 19, 19, planes)).astype('float32')
policy     = keras.utils.to_categorical(np.random.randint(moves, size=(N,))).astype('float32')
value      = np.random.randint(2, size=(N,)).astype('float32')
end        = np.random.randint(2, size=(N, 19, 19, 2)).astype('float32')
groups     = np.zeros((N, 19, 19, 1)).astype('float32')

print ("Tensorflow version", tf.__version__)
print ("getValidation", flush = True)
golois.getValidation(input_data, policy, value, end)

def augment_batch(x, p, v):
    k    = np.random.randint(0, 4)  
    flip = np.random.randint(0, 2)  

    x_aug = np.rot90(x, k, axes=(1, 2))
    p_aug = np.rot90(p.reshape(-1, 19, 19), k, axes=(1, 2)).reshape(-1, 361)

    if flip:
        x_aug = np.flip(x_aug, axis=2).copy()
        p_aug = np.flip(p_aug.reshape(-1, 19, 19), axis=2).reshape(-1, 361).copy()

    return x_aug, p_aug, v


def se_block(x, filters, ratio):
    se = layers.GlobalAveragePooling2D()(x)                          
    se = layers.Dense(filters // ratio, activation='relu',kernel_regularizer=regularizers.l2(0.0001))(se) 
    se = layers.Dense(filters, activation='sigmoid',kernel_regularizer=regularizers.l2(0.0001))(se) 
    se = layers.Reshape((1, 1, filters))(se)                         
    return layers.Multiply()([x, se])                               


def separable_residual_block(x, filters):
    shortcut = x

    x = layers.SeparableConv2D(
        filters, 3, padding='same', use_bias=False,
        depthwise_regularizer=regularizers.l2(0.0001),
        pointwise_regularizer=regularizers.l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.SeparableConv2D(
        filters, 3, padding='same', use_bias=False,
        depthwise_regularizer=regularizers.l2(0.0001),
        pointwise_regularizer=regularizers.l2(0.0001))(x)
    x = layers.BatchNormalization()(x)

    x = se_block(x, filters,8)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


input = keras.Input(shape=(19, 19, planes), name='board')

x = layers.Conv2D(filters, 3, padding='same', use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(input)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

for _ in range(7):
    x = separable_residual_block(x, filters)

policy_head = layers.Conv2D(4, 1, padding='same', use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(x)
policy_head = layers.BatchNormalization()(policy_head)
policy_head = layers.Activation('relu')(policy_head)
policy_head = layers.Conv2D(1, 1, padding='same', use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(policy_head)
policy_head = layers.Flatten()(policy_head)
policy_head = layers.Activation('softmax', name='policy')(policy_head)

value_head = layers.Conv2D(32, 1, padding='same', use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(x)
value_head = layers.BatchNormalization()(value_head)
value_head = layers.Activation('relu')(value_head)
value_head = layers.GlobalAveragePooling2D()(value_head)
value_head = layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.0001))(value_head)
value_head = layers.Dense(1, activation='sigmoid', name='value',kernel_regularizer=regularizers.l2(0.0001))(value_head)

model = keras.Model(inputs=input, outputs=[policy_head, value_head])
model.summary()

total_steps = epochs * (N // batch)

lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=total_steps,
    alpha=1e-6    
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    loss={
        'policy': 'categorical_crossentropy',
        'value':  'binary_crossentropy'
    },
    loss_weights={'policy': 2.0, 'value': 1.0},
    metrics={
        'policy': 'categorical_accuracy',
        'value':  'mae'
    }
)


for i in range(1, epochs + 1):
    print(f'\nEpoch {i}/{epochs}')

    golois.getBatch(input_data, policy, value, end, groups, i * N)

    x_aug, p_aug, v_aug = augment_batch(input_data, policy, value)

    model.fit(
        x_aug,
        [p_aug, v_aug],
        epochs=1,
        batch_size=batch,
        verbose=1
    )

    if i % 10 == 0:
        gc.collect()

    if i % 50 == 0:
        golois.getValidation(input_data, policy, value, end)
        val = model.evaluate(input_data, [policy, value],verbose=0, batch_size=batch)
        print(f"Validation epoch {i} : loss={val[0]:.4f} | "f"policy_acc={val[3]:.4f} | value_mae={val[4]:.4f}")
        model.save(f'test_{i}.h5')

model.save('test.h5')