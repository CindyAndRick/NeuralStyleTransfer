from keras_preprocessing.image import load_img, img_to_array
import numpy as np
from keras.applications import vgg19
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b
import imageio
import time
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# target_image_path = './img/monai.jpeg'              # 想要变换的图像路径
# style_reference_image_path = './img/fangao.jpeg'    # 目标风格的图像路径


def neuralStyleTransfer(target_image_path, style_reference_image_path, iterations):

    # 设置生成图像的尺寸
    width, height = load_img(target_image_path).size
    img_height = 400
    img_width = int(width * img_height / height)

    # 预处理
    def preprocess_image(image_path):
        img = load_img(image_path, target_size=(img_height, img_width))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = vgg19.preprocess_input(img)
        return img

    # 后处理
    def deprocess_image(x):
        # vgg19.preprocess的作用为减去ImageNet的平均像素值，使其中心为0。这里相当于其逆操作
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        # 将图像由BGR格式转换为RGB格式。同样为vgg19.preprocess_input逆操作的一部分
        x = x[:, :, ::-1]
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    # 构建vgg19网络，将三张图片（风格参考图像、目标图像和用于保存生成图像的占位符，一个符号张量）的批量作为输入
    target_image = K.constant(preprocess_image((target_image_path)))
    style_reference_image = K.constant(preprocess_image(style_reference_image_path))
    # 用于保存生成图像的占位符（符号张量）
    combination_image = K.placeholder((1, img_height, img_width, 3))

    # 将三张图像合并为一个批量
    input_tensor = K.concatenate([target_image,
                                  style_reference_image,
                                  combination_image], axis=0)

    # 利用三张图片组成的批量作为输入来构建VGG19网络
    # 加载模型将使用预训练的ImageNet权重
    model = vgg19.VGG19(input_tensor=input_tensor,
                        weights='imagenet',
                        include_top=False)
    print('Model loaded.')

    # 内容损失
    def content_loss(base, combination):
        return K.sum(K.square(combination - base))

    # 风格损失工具函数
    def gram_matrix(x):
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        gram = K.dot(features, K.transpose(features))
        return gram

    # 风格损失函数
    def style_loss(style, combination):
        S = gram_matrix(style)
        C = gram_matrix(combination)
        channels = 3
        size = img_height * img_width
        return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

    # 总变差损失
    def total_variation_loss(x):
        a = K.square(
            x[:, :img_height - 1, :img_width - 1, :] -
            x[:, 1:, :img_width - 1, :]
        )
        b = K.square(
            x[:, :img_height - 1, :img_width - 1, :] -
            x[:, :img_height - 1, 1:, :]
        )
        return K.sum(K.pow(a + b, 1.25))

    # 定义需要最小化的最终损失
    # 首先将层的名称映射为激活张量的字典
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    content_layer = 'block5_conv2'       # 用于内容损失的层
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']     # 用于风格损失的层
    # 损失分量的加权平均所使用的权重
    total_variation_weight = 1e-4
    style_weight = 1.
    content_weight = 0.025

    # 添加内容损失
    loss = K.variable(0.)
    layer_features = outputs_dict[content_layer]
    target_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * content_loss(target_image_features, combination_features)

    # 添加每个目标层的风格损失变量
    for layer_name in style_layers:
        layer_features = outputs_dict[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss = loss + (style_weight / len(style_layers)) * sl

    # 添加总变差损失
    loss = loss + total_variation_weight * total_variation_loss(combination_image)

    # 使用L-BFGS算法进行优化
    grads = K.gradients(loss, combination_image)[0]     # 获取损失相对于生成图像的梯度

    # 用于获取当前损失值和当前梯度值的函数
    fetch_loss_and_grads = K.function([combination_image], [loss, grads])

    # 该类将fetch_loss_and_grads包装，使得可以利用两个单独的方法调用来获取损失和梯度，这是我们要使用的SciPy优化器所要求的的
    class Evaluator(object):

        def __int__(self):
            self.loss_value = None
            self.grad_values = None

        def loss(self, x):
            # assert self.loss_value is None
            x = x.reshape((1, img_height, img_width, 3))
            outs = fetch_loss_and_grads([x])
            loss_value = outs[0]
            grad_values = outs[1].flatten().astype('float64')
            self.loss_value = loss_value
            self.grad_values = grad_values
            # self.loss_value = outs[0]
            # self.grad_values = outs[1].flatten().astype('float64')
            return self.loss_value

        def grads(self, x):
            assert self.loss_value is not None
            grad_values = np.copy(self.grad_values)
            self.loss_value = None
            self.grad_values = None
            return grad_values

    evaluator = Evaluator()

    # 风格迁移循环
    # result_prefix = './result/my_result'
    # iterations = 20

    x = preprocess_image(target_image_path)     # 初始状态目标图像
    x = x.flatten()     # 展平图像，因为scipy.optimize.fmin_l_bfgs_b只能处理展平的问题
    for i in range(iterations):
        print('Start of iteration', i)
        start_time = time.time()
        # 对生成图像的像素运行L-BFGS最优化，以将神经风格损失最小化。注意：必须将计算损失的函数和计算梯度的函数作为两个单独的参量传入
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss,
                                         x,
                                         fprime=evaluator.grads,
                                         maxfun=20)
        print('Current loss value:', min_val)
        # img = x.copy().reshape((img_height, img_width, 3))
        # img = deprocess_image(img)
        # fname = result_prefix + '_at_iteration_%d.png' % i
        # imageio.imsave(fname, img)
        # print('Image saved as', fname)
        end_time = time.time()
        print('Iteration %d completed in %ds' % (i, end_time - start_time))
        if i == iterations - 1:
            img = x.copy().reshape((img_height, img_width, 3))
            img = deprocess_image(img)
            imageio.imsave("result.jpg", img)

