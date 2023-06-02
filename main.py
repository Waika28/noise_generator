import numpy as np
import gradio as gr
import random

def add_gaussian_noise(img, scale):
    normal_distribution = np.random.normal(0, scale, img.shape) * 255
    return img + normal_distribution.astype(np.uint8)


def mul_gaussian_noise(img, scale):
    normal_distribution = np.random.normal(1, scale, img.shape)
    result = img * normal_distribution
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)


def pepper_noise(img, percentage):
    pepper = np.random.rand(*img.shape[:2])
    pepper = np.vectorize(lambda x: 0 if x < percentage else 1)(pepper)
    pepper = np.dstack((pepper, pepper, pepper))
    return img * pepper


def salt_noise(img, percentage):
    salt = np.random.rand(*img.shape[:2])
    salt = np.vectorize(lambda x: 255 if x < percentage else 0)(salt)
    salt = np.dstack((salt, salt, salt))
    return np.clip(img + salt, 0, 255)


def channel_shuffle(img):
    shuffled = list(range(3))
    random.shuffle(shuffled)
    return img[:, :, shuffled]


def main(img,
         use_gaussian_noise,
         gaussian_noise_type,
         gaussian_noise_scale,
         add_pepper,
         pepper_percentage,
         add_salt,
         salt_percentage,
         shuffle_channels):
    if shuffle_channels:
        img = channel_shuffle(img)
    if use_gaussian_noise:
        if gaussian_noise_type == "Мультипликативный":
            img = mul_gaussian_noise(img, gaussian_noise_scale)
        if gaussian_noise_type == "Аддитивный":
            img = add_gaussian_noise(img, gaussian_noise_scale)
    if add_pepper:
        img = pepper_noise(img, pepper_percentage)
    if add_salt:
        img = salt_noise(img, salt_percentage)
    return img



demo = gr.Interface(
    main,
    [
        gr.Image(label="Изображение"),
        gr.Checkbox(label="Добавить шум Гаусса"),
        gr.Radio(["Мультипликативный", "Аддитивный"], label="Тип шума Гаусса"),
        gr.Number(label="Интенсивность шума Гаусса"),
        gr.Checkbox(label="Поперчить"),
        gr.Number(label="Интенсивность"),
        gr.Checkbox(label="Посолить"),
        gr.Number(label="Интенсивность"),
        gr.Checkbox(label="Переставить каналы")
    ],
    "image"
)
demo.launch()
