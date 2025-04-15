import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Datasetni joylashgan yo'llar
train_dir = "dataset/train"
val_dir = "dataset/val"

# Data augmentation (rasmni tasodifiy o'zgartirish)
train_gen = ImageDataGenerator(
    rescale=1./255,  # Rangni normalizatsiya qilish
    rotation_range=30,
    zoom_range=0.3,
    horizontal_flip=True,
)

val_gen = ImageDataGenerator(rescale=1./255)

# Generatorlar yaratish (rasmlar uchun)
train_data = train_gen.flow_from_directory(
    train_dir, target_size=(128, 128), batch_size=32, class_mode="sparse"
)

val_data = val_gen.flow_from_directory(
    val_dir, target_size=(128, 128), batch_size=32, class_mode="sparse"
)

# MobileNetV2 bazasida model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Transfer learning (baza modelini o'zgartirmaslik)

# Modelning oxirgi qatlamlarini yaratish
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
output = Dense(6, activation='softmax')(x)  # 6 ta sinf uchun softmax

# Model yaratish
model = Model(inputs=base_model.input, outputs=output)

# Modelni kompilyatsiya qilish
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modelni o‘qitish
model.fit(train_data, epochs=10, validation_data=val_data)

# Modelni saqlash
model.save("animal_model_mobilenet.h5")
