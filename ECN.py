import tensorflow as tf 
import tensorflow.contrib.slim as slim

'''
============================================================================
EfficientConvNet for Sermantic Segmentation
============================================================================
'''

def ECN(inputs, num_classes,batch_size=4, reuse=None, is_training=True, outer_scope='EfficientConvNet'):
    '''
    The EfficientConvNet model for real-time semantic segmentation!

    INPUTS:
    - inputs(Tensor): a 4D Tensor of shape [batch_size, image_height, image_width, num_channels] that represents one batch of preprocessed images.
    - num_classes(int): an integer for the number of classes to predict. This will determine the final output channels as the answer.
    - batch_size(int): the batch size to explictly set the shape of the inputs in order for operations to work properly.
    - reuse(bool): Whether or not to reuse the variables for evaluation.
    - is_training(bool): if True, switch on batch_norm and prelu only during training, otherwise they are turned off.
    - scope(str): a string that represents the scope name for the variables.

    OUTPUTS:
    - net(Tensor): a 4D Tensor output of shape [batch_size, image_height, image_width, num_classes], where each pixel has a one-hot encoded vector
                      determining the label of the pixel.
    '''

    # Set the shape of the inputs first to get the batch_size information
    inputs_shape = inputs.get_shape().as_list()
    inputs.set_shape(shape=(batch_size, inputs_shape[1], inputs_shape[2], inputs_shape[3]))

    """with tf.name_scope(outer_scope):
        with tf.name_scope('1'):
            with tf.variable_scope('1', reuse=reuse):
                net = slim.conv2d(inputs, num_outputs=200, kernel_size=[3,3], stride=2, scope='down')
                print(net.get_shape().as_list())

        with tf.name_scope('2'):
            with tf.variable_scope('2', reuse=reuse):
                logits = slim.conv2d_transpose(net, num_outputs=num_classes, kernel_size=[2, 2], stride=2, padding='SAME', scope='up')
                print(logits.get_shape().as_list())
                probabilities = tf.nn.softmax(logits, name='logits_to_softmax')
                print(probabilities.get_shape().as_list())

    return logits, probabilities"""

    ##########
    with tf.name_scope(outer_scope):

        with tf.name_scope('Downsampler_Block_1'):
            with tf.variable_scope('Downsampler_Block_1', reuse=reuse):

                # print(scope)

                conv = slim.conv2d(inputs, num_outputs=13, kernel_size=[3,3], stride=2, activation_fn=None, scope='conv')
                pool = slim.max_pool2d(inputs, kernel_size=[2,2], stride=2, padding='SAME', scope='pool')

                net = tf.concat([conv, pool], axis=3)
                net = tf.nn.relu(net)

                print(net.get_shape().as_list())

        with tf.name_scope('Downsampler_Block_2'):
            with tf.variable_scope('Downsampler_Block_2', reuse=reuse):
                # print(scope)

                conv = slim.conv2d(net, num_outputs=48, kernel_size=[3, 3], stride=2, activation_fn=None, scope='conv')
                pool = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, padding='SAME', scope='pool')

                net = tf.concat([conv, pool], axis=3)
                net = tf.nn.relu(net)

                print(net.get_shape().as_list())

        with tf.name_scope('Non-bt-1D-1'):
            with tf.variable_scope('Non-bt-1D-1', reuse=reuse):
                # print(scope)

                for i in range(4):
                    previous_block = net
                    net = slim.conv2d(net, 64, kernel_size=[3, 1], scope='sep_conv_%d_1' % (i + 1))
                    net = slim.conv2d(net, 64, kernel_size=[1, 3], scope='sep_conv_%d_2' % (i + 1))
                    net = slim.batch_norm(net, is_training=is_training)
                    net = slim.conv2d(net, 64, kernel_size=[3, 1], scope='sep_conv_%d_3' % (i + 1))
                    net = slim.conv2d(net, 64, kernel_size=[1, 3], scope='sep_conv_%d_4' % (i + 1))
                    net = slim.batch_norm(net, is_training=is_training)
                    net = tf.add(net, previous_block)

                    print(net.get_shape().as_list())

        with tf.name_scope('Downsampler_Block_3'):
            with tf.variable_scope('Downsampler_Block_3', reuse=reuse):
                # print(scope)

                conv = slim.conv2d(net, num_outputs=64, kernel_size=[3, 3], stride=2, activation_fn=None, scope='conv')
                pool = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, padding='SAME', scope='pool')

                net = tf.concat([conv, pool], axis=3)
                net = tf.nn.relu(net)

                print(net.get_shape().as_list())

        with tf.name_scope('Non-bt-1D-dilated2_1'):
            with tf.variable_scope('Non-bt-1D-dilated2_1', reuse=reuse):
                # print(scope)

                rate=2

                previous_block = net

                net = slim.conv2d(net, 128, kernel_size=[3, 1], scope='sep_conv_dilated_1')
                net = slim.conv2d(net, 128, kernel_size=[1, 3], scope='sep_conv_dilated_2')
                net = slim.batch_norm(net, is_training=is_training)
                net = slim.conv2d(net, 128, kernel_size=[3, 1], rate=rate, scope='sep_conv_dilated_3')
                net = slim.conv2d(net, 128, kernel_size=[1, 3], rate=rate, scope='sep_conv_dilated_4')
                net = slim.batch_norm(net, is_training=is_training)
                net = tf.add(net, previous_block)

                print(net.get_shape().as_list())

        with tf.name_scope('Non-bt-1D-dilated4_2'):
            with tf.variable_scope('Non-bt-1D-dilated4_2', reuse=reuse):
                # print(scope)

                rate=4

                previous_block = net

                net = slim.conv2d(net, 128, kernel_size=[3, 1], scope='sep_conv_dilated_1')
                net = slim.conv2d(net, 128, kernel_size=[1, 3], scope='sep_conv_dilated_2')
                net = slim.batch_norm(net, is_training=is_training)
                net = slim.conv2d(net, 128, kernel_size=[3, 1], rate=rate, scope='sep_conv_dilated_3')
                net = slim.conv2d(net, 128, kernel_size=[1, 3], rate=rate, scope='sep_conv_dilated_4')
                net = slim.batch_norm(net)
                net = tf.add(net, previous_block)

                print(net.get_shape().as_list())

        with tf.name_scope('Non-bt-1D-dilated8_3'):
            with tf.variable_scope('Non-bt-1D-dilated8_3', reuse=reuse):
                # print(scope)

                rate=8

                previous_block = net

                net = slim.conv2d(net, 128, kernel_size=[3, 1], scope='sep_conv_dilated_1')
                net = slim.conv2d(net, 128, kernel_size=[1, 3], scope='sep_conv_dilated_2')
                net = slim.batch_norm(net, is_training=is_training)
                net = slim.conv2d(net, 128, kernel_size=[3, 1], rate=rate, scope='sep_conv_dilated_3')
                net = slim.conv2d(net, 128, kernel_size=[1, 3], rate=rate, scope='sep_conv_dilated_4')
                net = slim.batch_norm(net, is_training=is_training)
                net = tf.add(net, previous_block)

                print(net.get_shape().as_list())

        with tf.name_scope('Non-bt-1D-dilated16_4'):
            with tf.variable_scope('Non-bt-1D-dilated16_4', reuse=reuse):
                # print(scope)

                rate=16

                previous_block = net

                net = slim.conv2d(net, 128, kernel_size=[3, 1], scope='sep_conv_dilated_1')
                net = slim.conv2d(net, 128, kernel_size=[1, 3], scope='sep_conv_dilated_2')
                net = slim.batch_norm(net, is_training=is_training)
                net = slim.conv2d(net, 128, kernel_size=[3, 1], rate=rate, scope='sep_conv_dilated_3')
                net = slim.conv2d(net, 128, kernel_size=[1, 3], rate=rate, scope='sep_conv_dilated_4')
                net = slim.batch_norm(net, is_training=is_training)
                net = tf.add(net, previous_block)

                print(net.get_shape().as_list())

        with tf.name_scope('Non-bt-1D-dilated2_5'):
            with tf.variable_scope('Non-bt-1D-dilated2_5', reuse=reuse):
                # print(scope)

                rate=2

                previous_block = net

                net = slim.conv2d(net, 128, kernel_size=[3, 1], scope='sep_conv_dilated_1')
                net = slim.conv2d(net, 128, kernel_size=[1, 3], scope='sep_conv_dilated_2')
                net = slim.batch_norm(net, is_training=is_training)
                net = slim.conv2d(net, 128, kernel_size=[3, 1], rate=rate, scope='sep_conv_dilated_3')
                net = slim.conv2d(net, 128, kernel_size=[1, 3], rate=rate, scope='sep_conv_dilated_4')
                net = slim.batch_norm(net, is_training=is_training)
                net = tf.add(net, previous_block)

                print(net.get_shape().as_list())

        with tf.name_scope('Non-bt-1D-dilated4_6'):
            with tf.variable_scope('Non-bt-1D-dilated4_6', reuse=reuse):
                # print(scope)

                rate=4

                previous_block = net

                net = slim.conv2d(net, 128, kernel_size=[3, 1], scope='sep_conv_dilated_1')
                net = slim.conv2d(net, 128, kernel_size=[1, 3], scope='sep_conv_dilated_2')
                net = slim.batch_norm(net, is_training=is_training)
                net = slim.conv2d(net, 128, kernel_size=[3, 1], rate=rate, scope='sep_conv_dilated_3')
                net = slim.conv2d(net, 128, kernel_size=[1, 3], rate=rate, scope='sep_conv_dilated_4')
                net = slim.batch_norm(net, is_training=is_training)
                net = tf.add(net, previous_block)

                print(net.get_shape().as_list())

        with tf.name_scope('Non-bt-1D-dilated8_7'):
            with tf.variable_scope('Non-bt-1D-dilated8_7', reuse=reuse):
                # print(scope)

                rate=8

                previous_block = net

                net = slim.conv2d(net, 128, kernel_size=[3, 1], scope='sep_conv_dilated_1')
                net = slim.conv2d(net, 128, kernel_size=[1, 3], scope='sep_conv_dilated_2')
                net = slim.batch_norm(net, is_training=is_training)
                net = slim.conv2d(net, 128, kernel_size=[3, 1], rate=rate, scope='sep_conv_dilated_3')
                net = slim.conv2d(net, 128, kernel_size=[1, 3], rate=rate, scope='sep_conv_dilated_4')
                net = slim.batch_norm(net, is_training=is_training)
                net = tf.add(net, previous_block)

                print(net.get_shape().as_list())

        with tf.name_scope('Non-bt-1D-dilated16_8'):
            with tf.variable_scope('Non-bt-1D-dilated16_8', reuse=reuse):
                # print(scope)

                rate=16

                previous_block = net

                net = slim.conv2d(net, 128, kernel_size=[3, 1], rate=rate, scope='sep_conv_dilated_1')
                net = slim.conv2d(net, 128, kernel_size=[1, 3], rate=rate, scope='sep_conv_dilated_2')
                net = slim.batch_norm(net, is_training=is_training)
                net = slim.conv2d(net, 128, kernel_size=[3, 1], rate=rate, scope='sep_conv_dilated_3')
                net = slim.conv2d(net, 128, kernel_size=[1, 3], rate=rate, scope='sep_conv_dilated_4')
                net = slim.batch_norm(net, is_training=is_training)
                net = tf.add(net, previous_block)

                print(net.get_shape().as_list())


        with tf.name_scope('Deconvolution_1'):
            with tf.variable_scope('Deconvolution_1', reuse=reuse):
                # print(scope)

                net = slim.conv2d_transpose(net, num_outputs=64, kernel_size=[2,2], stride=2, padding='SAME')

                print(net.get_shape().as_list())

        with tf.name_scope('Non-bt-1D-2'):
            with tf.variable_scope('Non-bt-1D-2', reuse=reuse):
                # print(scope)

                for i in range(2):
                    previous_block = net

                    net = slim.conv2d(net, 64, kernel_size=[3, 1], scope='sep_conv_%d_1' % (i + 1))
                    net = slim.conv2d(net, 64, kernel_size=[1, 3], scope='sep_conv_%d_2' % (i + 1))
                    net = slim.batch_norm(net, is_training=is_training)
                    net = slim.conv2d(net, 64, kernel_size=[3, 1], scope='sep_conv_%d_3' % (i + 1))
                    net = slim.conv2d(net, 64, kernel_size=[1, 3], scope='sep_conv_%d_4' % (i + 1))
                    net = slim.batch_norm(net, is_training=is_training)
                    net = tf.add(net, previous_block)

                    print(net.get_shape().as_list())

        with tf.name_scope('Deconvolution_2'):
            with tf.variable_scope('Deconvolution_2', reuse=reuse):
                # print(scope)

                net = slim.conv2d_transpose(net, num_outputs=16, kernel_size=[2, 2], stride=2, padding='SAME')

                print(net.get_shape().as_list())

        with tf.name_scope('Non-bt-1D-3'):
            with tf.variable_scope('Non-bt-1D-3', reuse=reuse):
                # print(scope)

                for i in range(2):
                    previous_block = net

                    net = slim.conv2d(net, 16, kernel_size=[3, 1], scope='sep_conv_%d_1' % (i + 1))
                    net = slim.conv2d(net, 16, kernel_size=[1, 3], scope='sep_conv_%d_2' % (i + 1))
                    net = slim.batch_norm(net, is_training=is_training)
                    net = slim.conv2d(net, 16, kernel_size=[3, 1], scope='sep_conv_%d_3' % (i + 1))
                    net = slim.conv2d(net, 16, kernel_size=[1, 3], scope='sep_conv_%d_4' % (i + 1))
                    net = slim.batch_norm(net, is_training=is_training)
                    net = tf.add(net, previous_block)

                    print(net.get_shape().as_list())

        with tf.name_scope('Deconvolution_3'):
            with tf.variable_scope('Deconvolution_3', reuse=reuse):
                # print(scope)

                logits = slim.conv2d_transpose(net, num_outputs=num_classes, kernel_size=[2, 2], stride=2, padding='SAME')
                probabilities = tf.nn.softmax(logits, name='logits_to_softmax')

                print(logits.get_shape().as_list())

    return logits, probabilities









