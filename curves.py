import cv2
import numpy as np
import skimage
import random

class ImageBrightnessAdjuster:
    def __init__(self):
        pass

    def darken_image_tone_mapping(image):
        # 将输入图像从RGB空间转换到HSV空间
        #低光：curve_intensity=6
        #高光：curve_intensity=0.1
        ctrlnum = random.randint(0, 1)
        if (ctrlnum % 2 == 0):
            curve_intensity = 6
        else:
            curve_intensity = 0.1

        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # 提取V通道（亮度）
        V_channel = hsv_image[:, :, 2]

        curve_table = np.zeros((256,), dtype=np.uint8)
        for i in range(256):
            curve_table[i] = np.clip(pow(i / 255.0, curve_intensity) * 255.0, 0, 255)

        # 对V通道进行曲线映射
        V_channel = cv2.LUT(V_channel, curve_table)

        # 将修改后的V通道和原始的H和S通道合并
        hsv_image[:, :, 2] = V_channel
        darkened_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

        return darkened_image

    def gamma_correction(image):
        # 将输入图像从RGB空间转换到灰度空间
        ctrlnum = random.randint(0, 1)
        if (ctrlnum % 2 == 0):
            x = random.uniform(0, 0.2)
        else:
            x = random.uniform(5, 8)


        adjusted_image = skimage.exposure.adjust_gamma(image, gamma=x)


        return adjusted_image

    def inverse_proportion(self, image, a=1.0):
        # 将输入图像从RGB空间转换到HSV空间
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # 提取V通道（亮度）
        V_channel = hsv_image[:, :, 2]

        # 计算反比例函数的映射表
        curve_table = np.zeros((256,), dtype=np.uint8)
        for i in range(256):
            curve_table[i] = np.clip((a / i), 0, 255)

        # 对V通道进行曲线映射
        V_channel = cv2.LUT(V_channel, curve_table)

        # 将修改后的V通道和原始的H和S通道合并
        hsv_image[:, :, 2] = V_channel
        adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

        return adjusted_image

    def log_curve(image, a=10):
        #大于25为过曝，小于25为低光
        # 将输入图像从RGB空间转换到HSV空间
        ctrlnum = random.randint(0, 1)
        if (ctrlnum % 2 == 0):
            a=10
        else:
            a=40
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # 提取V通道（亮度）
        V_channel = hsv_image[:, :, 2]

        # 计算对数函数的映射表
        curve_table = np.zeros((256,), dtype=np.uint8)
        for i in range(256):
            curve_table[i] = np.clip((a * np.log(1 + i)), 0, 255)

        # 对V通道进行曲线映射
        V_channel = cv2.LUT(V_channel, curve_table)

        # 将修改后的V通道和原始的H和S通道合并
        hsv_image[:, :, 2] = V_channel
        adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

        return adjusted_image

    def exponential_curve(self, image, a=1.0):
        # 将输入图像从RGB空间转换到HSV空间
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # 提取V通道（亮度）
        V_channel = hsv_image[:, :, 2]

        # 计算指数函数的映射表
        curve_table = np.zeros((256,), dtype=np.uint8)
        for i in range(256):
            curve_table[i] = np.clip((a * (np.exp(i / 255.0) - 1)), 0, 255)

        # 对V通道进行曲线映射
        V_channel = cv2.LUT(V_channel, curve_table)

        # 将修改后的V通道和原始的H和S通道合并
        hsv_image[:, :, 2] = V_channel
        adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

        return adjusted_image

    def polynomial_curve(self, image, a=1.0, b=0.5):
        # 将输入图像从RGB空间转换到HSV空间
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # 提取V通道（亮度）
        V_channel = hsv_image[:, :, 2]

        # 计算多项式函数的映射表
        curve_table = np.zeros((256,), dtype=np.uint8)
        for i in range(256):
            curve_table[i] = np.clip((a * (i / 255.0) ** b), 0, 255)

        # 对V通道进行曲线映射
        V_channel = cv2.LUT(V_channel, curve_table)

        # 将修改后的V通道和原始的H和S通道合并
        hsv_image[:, :, 2] = V_channel
        adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

        return adjusted_image

    def hyperbolic_curve(self, image, a=1.0, b=1.0):
        # 将输入图像从RGB空间转换到HSV空间
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # 提取V通道（亮度）
        V_channel = hsv_image[:, :, 2]

        # 计算双曲线函数的映射表
        curve_table = np.zeros((256,), dtype=np.uint8)
        for i in range(256):
            curve_table[i] = np.clip((a / (i / 255.0 + b)), 0, 255)

        # 对V通道进行曲线映射
        V_channel = cv2.LUT(V_channel, curve_table)

        # 将修改后的V通道和原始的H和S通道合并
        hsv_image[:, :, 2] = V_channel
        adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

        return adjusted_image