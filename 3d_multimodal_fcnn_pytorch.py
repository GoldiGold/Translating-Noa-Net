import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms
import numpy as np
from config import CONF

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def calculate_rigid_transform(sin_a, sin_b, sin_c, cos_a, cos_b, cos_c, MrVista=False):
    # Calculate the rigid transformation according to transform parameters Tx,Ty,Tz,Ax.Ay,Az
    # According to: http://www.songho.ca/opengl/gl_anglestoaxes.html
    #               https://slideplayer.com/slide/9396372/

    # cos_a, sin_a correlates with Ry
    # cos_b, sin_b correlates with Rx
    # cos_c, sin_c correlates with Rz
    # Axis x switched with axis y
    if MrVista == False:
        a00 = cos_c * cos_b
        a01 = cos_c * sin_b * sin_a - sin_c * cos_a
        a02 = cos_c * sin_b * cos_a + sin_c * sin_a
        a10 = sin_c * cos_b
        a11 = sin_a * sin_b * sin_c + cos_c * cos_a
        a12 = sin_c * sin_b * cos_a - cos_c * sin_a
        a20 = -sin_b
        a21 = cos_b * sin_a
        a22 = cos_b * cos_a
    else:
        a00 = cos_c * cos_b
        a01 = -(-cos_b * sin_c)
        a02 = sin_b
        a10 = -(sin_a * sin_b * cos_c + cos_a * sin_c)
        a11 = -sin_a * sin_b * sin_c + cos_a * cos_c
        a12 = -(-sin_a * cos_b)
        a20 = -cos_a * sin_b * cos_c + sin_a * sin_c
        a21 = -(cos_a * sin_b * sin_c + sin_a * cos_c)
        a22 = cos_a * cos_b

    return a00, a01, a02, a10, a11, a12, a20, a21, a22


def transformer3D(U, theta, out_size, forward_mapping=False, **kwargs):
    """Spatial Transformer Layer

    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and edited by David Dao for Tensorflow. Transformed to Pytorch by Omri Leshem on March 2022

    Parameters
    ----------s
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, depth, num_channels].
    theta: float
        The output of the
        localisation network should be [num_batch, 12].
    out_size: tuple of three ints
        The size of the output of the network (height, width, depth)

    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py

    Notes
    -----
    To initialize the network to the identity transform init
    ``theta`` to :
        identity = np.array([[1., 0., 0., 0.],
                             [0., 1., 0., 0.]
                             [0., 0., 1., 0.]])
        identity = identity.flatten()
        theta = Variable(initial_value=identity)

    """

    data_type = 'float32'

    def _repeat(x, n_repeats):
        with torch.variable_scope('_repeat'):
            rep = torch.transpose(
                torch.expand_dims(torch.ones(shape=torch.stack([n_repeats, ]), dtype=data_type), 1), [1, 0])
            x = torch.cast(x, data_type)
            x = torch.matmul(torch.reshape(x, (-1, 1)), rep)
            x = torch.cast(x, 'int32')
            return torch.reshape(x, [-1])

    def _bilinear_interpolate(im, x, y, z):

            with torch.variable_scope('_interpolate'):
                # constants
                num_batch = torch.shape(im)[0]
                height = torch.shape(im)[1]
                width = torch.shape(im)[2]
                depth = torch.shape(im)[3]
                num_channels = torch.shape(im)[4]

                x = torch.cast(x, data_type)
                y = torch.cast(y, data_type)
                z = torch.cast(z, data_type)

                height_f = torch.cast(height, data_type)
                width_f = torch.cast(width, data_type)
                depth_f = torch.cast(depth, data_type)

                zero = torch.zeros([], dtype='int32')
                max_y = torch.cast(height - 1, 'int32')
                max_x = torch.cast(width - 1, 'int32')
                max_z = torch.cast(depth - 1, 'int32')

                # scale indices from [-1, 1] to [0, width/height]
                x = (x + 1.0) * (width_f) / 2.0
                y = (y + 1.0) * (height_f) / 2.0
                z = (z + 1.0) * (depth_f) / 2.0

                # do sampling
                x0 = torch.cast(torch.floor(x), 'int32')
                x1 = x0 + 1
                y0 = torch.cast(torch.floor(y), 'int32')
                y1 = y0 + 1
                z0 = torch.cast(torch.floor(z), 'int32')
                z1 = z0 + 1

                x0 = torch.clip_by_value(x0, zero, max_x)
                x1 = torch.clip_by_value(x1, zero, max_x)
                y0 = torch.clip_by_value(y0, zero, max_y)
                y1 = torch.clip_by_value(y1, zero, max_y)
                z0 = torch.clip_by_value(z0, zero, max_z)
                z1 = torch.clip_by_value(z1, zero, max_z)

                dim3 = depth
                dim2 = depth * width
                dim1 = depth * width * height

                base = _repeat(range(num_batch)*dim1, height*width*depth)

                # a, b, c, d - corners of upper box face
                base_y0 = base + y0*dim2
                base_x0 = base_y0 + x0*dim3
                base_x1 = base_y0 + x1*dim3
                idx_a = torch.expand_dims(base_x0 + z0, 1)
                idx_b = torch.expand_dims(base_x1 + z0, 1)
                idx_c = torch.expand_dims(base_x0 + z1, 1)
                idx_d = torch.expand_dims(base_x1 + z1, 1)

                # e, f, g, h - corners of lower box face
                base_y1 = base + y1*dim2
                base_x0 = base_y1 + x0*dim3
                base_x1 = base_y1 + x1*dim3
                idx_e = torch.expand_dims(base_x0 + z0, 1)
                idx_f = torch.expand_dims(base_x1 + z0, 1)
                idx_g = torch.expand_dims(base_x0 + z1, 1)
                idx_h = torch.expand_dims(base_x1 + z1, 1)

                # use indices to lookup pixels in the flat image and restore
                # channels dim
                im_flat = torch.reshape(im,torch.stack([-1, num_channels]))
                im_flat = torch.cast(im_flat, data_type)

                x0_f = torch.cast(x0, data_type)
                x1_f = torch.cast(x1, data_type)
                y0_f = torch.cast(y0, data_type)
                y1_f = torch.cast(y1, data_type)
                z0_f = torch.cast(z0, data_type)
                z1_f = torch.cast(z1, data_type)

                # choose only neighbour pixels
                dx1 = torch.abs(x1_f - x)
                dx1 = torch.cast(torch.less_equal(dx1, 1), data_type) * dx1
                dy1 = torch.abs(y1_f - y)
                dy1 = torch.cast(torch.less_equal(dy1, 1), data_type) * dy1
                dz1 = torch.abs(z1_f - z)
                dz1 = torch.cast(torch.less_equal(dz1, 1), data_type) * dz1
                dx0 = torch.abs(x - x0_f)
                dx0 = torch.cast(torch.less_equal(dx0, 1), data_type) * dx0
                dy0 = torch.abs(y - y0_f)
                dy0 = torch.cast(torch.less_equal(dy0, 1), data_type) * dy0
                dz0 = torch.bs(z - z0_f)
                dz0 = torch.cast(torch.less_equal(dz0, 1), data_type) * dz0

                Iwa = torch.scatter_nd(idx_a, torch.expand_dims(dx1 * dy1 * dz1, 1) * im_flat, [num_batch*height*width*depth, num_channels])
                Iwb = torch.scatter_nd(idx_b, torch.expand_dims(dx0 * dy1 * dz1, 1) * im_flat, [num_batch*height*width*depth, num_channels])
                Iwc = torch.scatter_nd(idx_c, torch.expand_dims(dx1 * dy1 * dz0, 1) * im_flat, [num_batch*height*width*depth, num_channels])
                Iwd = torch.scatter_nd(idx_d, torch.expand_dims(dx0 * dy1 * dz0, 1) * im_flat, [num_batch*height*width*depth, num_channels])
                Iwe = torch.scatter_nd(idx_e, torch.expand_dims(dx1 * dy0 * dz1, 1) * im_flat, [num_batch*height*width*depth, num_channels])
                Iwf = torch.scatter_nd(idx_f, torch.expand_dims(dx0 * dy0 * dz1, 1) * im_flat, [num_batch*height*width*depth, num_channels])
                Iwg = torch.scatter_nd(idx_g, torch.expand_dims(dx1 * dy0 * dz0, 1) * im_flat, [num_batch*height*width*depth, num_channels])
                Iwh = torch.scatter_nd(idx_h, torch.expand_dims(dx0 * dy0 * dz0, 1) * im_flat, [num_batch*height*width*depth, num_channels])

                wa = torch.scatter_nd(idx_a, torch.expand_dims(dx1 * dy1 * dz1, 1), [num_batch*height*width*depth, 1])
                wb = torch.scatter_nd(idx_b, torch.expand_dims(dx0 * dy1 * dz1, 1), [num_batch*height*width*depth, 1])
                wc = torch.scatter_nd(idx_c, torch.expand_dims(dx1 * dy1 * dz0, 1), [num_batch*height*width*depth, 1])
                wd = torch.scatter_nd(idx_d, torch.expand_dims(dx0 * dy1 * dz0, 1), [num_batch*height*width*depth, 1])
                we = torch.scatter_nd(idx_e, torch.expand_dims(dx1 * dy0 * dz1, 1), [num_batch*height*width*depth, 1])
                wf = torch.scatter_nd(idx_f, torch.expand_dims(dx0 * dy0 * dz1, 1), [num_batch*height*width*depth, 1])
                wg = torch.scatter_nd(idx_g, torch.expand_dims(dx1 * dy0 * dz0, 1), [num_batch*height*width*depth, 1])
                wh = torch.scatter_nd(idx_h, torch.expand_dims(dx0 * dy0 * dz0, 1), [num_batch*height*width*depth, 1])

                value_all = torch.add_n([Iwa, Iwb, Iwc, Iwd, Iwe, Iwf, Iwg, Iwh])
                weight_all = torch.clip_by_value(torch.add_n([wa, wb, wc, wd, we, wf, wg, wh]), 1e-5, 1e+10)

                output = torch.iv(value_all, weight_all)

                return output

    def _interpolate(im, x, y, z, out_size):

        with torch.variable_scope('_interpolate'):
            # constants
            num_batch = torch.shape(im)[0]
            height = torch.shape(im)[1]
            width = torch.shape(im)[2]
            depth = torch.shape(im)[3]
            channels = torch.shape(im)[4]

            x = torch.cast(x, data_type)
            y = torch.cast(y, data_type)
            z = torch.cast(z, data_type)
            height_f = torch.cast(height, data_type)
            width_f = torch.cast(width, data_type)
            depth_f = torch.cast(depth, data_type)
            out_height = out_size[0]
            out_width = out_size[1]
            out_depth = out_size[2]
            zero = torch.zeros([], dtype='int32')
            max_y = torch.cast(torch.shape(im)[1] - 1, 'int32')
            max_x = torch.cast(torch.shape(im)[2] - 1, 'int32')
            max_z =torch.cast(torch.shape(im)[3] - 1, 'int32')

            # scale indices from [-1, 1] to [0, width/height/depth]
            x = (x + 1.0) * (width_f) / 2.0
            y = (y + 1.0) * (height_f) / 2.0
            z = (z + 1.0) * (depth_f) / 2.0

            # do sampling
            x0 = torch.cast(torch.floor(x), 'int32')
            x1 = x0 + 1
            y0 = torch.cast(torch.floor(y), 'int32')
            y1 = y0 + 1
            z0 = torch.cast(torch.floor(z), 'int32')
            z1 = z0 + 1

            x0 = torch.clip_by_value(x0, zero, max_x)
            x1 = torch.clip_by_value(x1, zero, max_x)
            y0 = torch.clip_by_value(y0, zero, max_y)
            y1 = torch.clip_by_value(y1, zero, max_y)
            z0 = torch.clip_by_value(z0, zero, max_z)
            z1 = torch.clip_by_value(z1, zero, max_z)

            dim3 = depth
            dim2 = depth*width
            dim1 = depth*width*height

            base = _repeat(torch.range(num_batch)*dim1, out_height*out_width*out_depth)

            # a, b, c, d - corners of upper box face
            base_y0 = base + y0*dim2
            base_x0 = base_y0 + x0*dim3
            base_x1 = base_y0 + x1*dim3
            idx_a = base_x0 + z0
            idx_b = base_x1 + z0
            idx_c = base_x0 + z1
            idx_d = base_x1 + z1

            # e, f, g, h - corners of lower box face
            base_y1 = base + y1*dim2
            base_x0 = base_y1 + x0*dim3
            base_x1 = base_y1 + x1*dim3
            idx_e = base_x0 + z0
            idx_f = base_x1 + z0
            idx_g = base_x0 + z1
            idx_h = base_x1 + z1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = torch.eshape(im,torch.stack([-1, channels]))
            im_flat = torch.cast(im_flat, data_type)
            Ia = torch.gather(im_flat, idx_a)
            Ib = torch.gather(im_flat, idx_b)
            Ic = torch.gather(im_flat, idx_c)
            Id = torch.gather(im_flat, idx_d)
            Ie = torch.gather(im_flat, idx_e)
            If = torch.gather(im_flat, idx_f)
            Ig = torch.gather(im_flat, idx_g)
            Ih = torch.ather(im_flat, idx_h)

            # and finally calculate interpolated values
            x0_f = torch.cast(x0, data_type)
            x1_f = torch.st(x1, data_type)
            y0_f = torch.ast(y0, data_type)
            y1_f = torch.cast(y1, data_type)
            z0_f = torch.ast(z0, data_type)
            z1_f = torch.cast(z1, data_type)

            wa = torch.expand_dims(((x1_f-x) * (y1_f-y) * (z1_f-z)), 1)
            wb = torch.expand_dims(((x-x0_f) * (y1_f-y) * (z1_f-z)), 1)
            wc = torch.expand_dims(((x1_f-x) * (y1_f-y) * (z-z0_f)), 1)
            wd = torch.expand_dims(((x-x0_f) * (y1_f-y) * (z-z0_f)), 1)
            we = torch.expand_dims(((x1_f-x) * (y-y0_f) * (z1_f-z)), 1)
            wf = torch.expand_dims(((x-x0_f) * (y-y0_f) * (z1_f-z)), 1)
            wg = torch.expand_dims(((x1_f-x) * (y-y0_f) * (z-z0_f)), 1)
            wh = torch.expand_dims(((x-x0_f) * (y-y0_f) * (z-z0_f)), 1)

            output = torch.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id, we*Ie, wf*If, wg*Ig, wh*Ih])

            return output

    def _meshgrid(height, width, depth):

        with torch.variable_scope('_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t, z_t = np.meshgrid(np.linspace(-1, 1, width),
            #                              np.linspace(-1, 1, height),
            #                              np.linspace(-1, 1, depth))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), z_t.flatten(), ones])

            # x_t = matmul(ones(shape=stack([height, 1])),
            #                 transpose(expand_dims(linspace(-1.0, 1.0, width), 1), [1, 0]))
            # y_t = matmul(expand_dims(linspace(-1.0, 1.0, height), 1),
            #                 ones(shape=stack([1, width])))

            x_t, y_t, z_t = torch.meshgrid(torch.linspace(-1, 1, width, dtype=data_type),
                                        torch.linspace(-1, 1, height, dtype=data_type),
                                        torch.linspace(-1, 1, depth, dtype=data_type))
            x_t = torch.cast(x_t, data_type)
            y_t = torch.cast(y_t, data_type)
            z_t = torch.cast(z_t, data_type)

            x_t_flat = torch.reshape(x_t, (1, -1))
            y_t_flat = torch.reshape(y_t, (1, -1))
            z_t_flat = torch.reshape(z_t, (1, -1))

            ones = torch.ones_like(x_t_flat)
            grid = torch.concat([x_t_flat, y_t_flat, z_t_flat, ones], 0)
            return grid

    def _transform(theta, input_dim, out_size, forward_mapping):
        with torch.variable_scope('_transform'):
            num_batch = torch.shape(input_dim)[0]
            theta = torch.reshape(theta, (-1, 3, 4))
            theta = torch.cast(theta, data_type)

            # grid of (x_t, y_t, z_t, 1), eq (1) in ref [1]
            out_height = out_size[0]
            out_width = out_size[1]
            out_depth = out_size[2]
            num_channels = out_size[-1]
            grid = _meshgrid(out_height, out_width, out_depth)
            grid = torch.expand_dims(grid, 0)
            grid = torch.reshape(grid, [-1])  # flatten
            grid = torch.tile(grid, torch.stack([num_batch]))
            grid = torch.reshape(grid, torch.stack([num_batch, 4, -1]))

            # Transform A x (x_t, y_t, z_t, 1)^T -> (x_s, y_s, z_s)
            T_g = torch.matmul(theta, grid)
            x_s = slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = slice(T_g, [0, 1, 0], [-1, 1, -1])
            z_s = slice(T_g, [0, 2, 0], [-1, 1, -1])
            x_s_flat = torch.reshape(x_s, [-1])
            y_s_flat = torch.reshape(y_s, [-1])
            z_s_flat = torch.reshape(z_s, [-1])

            if forward_mapping:
                input_transformed = _bilinear_interpolate(input_dim, x_s_flat, y_s_flat, z_s_flat)
            else:
                input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat, z_s_flat, out_size)

            output = torch.reshape(
                input_transformed, torch.stack([num_batch, out_height, out_width, out_depth, num_channels]))

            return output

    with torch.variable_scope('SpatialTransformer'):
        output = _transform(theta, U, out_size, forward_mapping)
        return output

def stn_pytorch(rigid_body_affine):

    # Create rigid_body_affine with tensorflow
    # According to: http://www.songho.ca/opengl/gl_anglestoaxes.html
    #               https://slideplayer.com/slide/9396372/
    # Axis x switched with axis y

    ax = torch.identity(rigid_body_affine[0, 3], name='angle_a')  # X axis
    ay = torch.identity(rigid_body_affine[0, 4], name='angle_b')  # Y axis
    az = torch.identity(rigid_body_affine[0, 5], name='angle_c')  # Z axis

    tx = torch.identity(rigid_body_affine[0, 0], name='tx')
    ty = torch.identity(rigid_body_affine[0, 1], name='ty')
    tz = torch.identity(rigid_body_affine[0, 2], name='tz')

    sin_a = torch.math.sin(ay)
    sin_b = torch.math.sin(ax)
    sin_c = torch.math.sin(az)

    cos_a = torch.math.cos(ay)
    cos_b = torch.math.cos(ax)
    cos_c = torch.math.cos(az)

    a00,a01,a02,a10,a11,a12,a20,a21,a22 = calculate_rigid_transform(sin_a, sin_b, sin_c, cos_a, cos_b, cos_c)

    affine1 = torch.transpose(torch.expand_dims(torch.stack([a00, a01, a02, ty], 0), 1))
    affine2 = torch.transpose(torch.expand_dims(torch.stack([a10, a11, a12, tx], 0), 1))
    affine3 = torch.transpose(torch.expand_dims(torch.stack([a20, a21, a22, tz], 0), 1))

    theta = torch.concat([affine1, affine2, affine3], 0)

    return theta


def STN(data, theta, output_size, forward_mapping=False):

    rigid_body_trans = stn_pytorch(theta)

    return transformer3D(data, rigid_body_trans , output_size, forward_mapping=forward_mapping)

class STN(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        k = self.kernel_size
        #######
        # STN #
        #######
        (self.feed('data_T1', 'data_FA').locnet(trainable=CONF['use_STN'], theta_name='theta_prev', reuse=False))

        if CONF['use_STN']:
            (self.feed('data_FA', 'theta_prev').stn(output_size=(self.mri_dims[0], self.mri_dims[1], self.mri_dims[2], 1),
                                                forward_mapping=True).set_layer_name(name='data_FA_warped_prev'))

        def cond(i, stn_iterations):
            return torch.less(i, stn_iterations)

        def body(i, theta_prev, t1, fa, fa_warped, stn_iterations):
            concat = torch.concat(values=[fa_warped, t1], axis=-1, name='concat_in_stn_loop')
            theta_new = self.layers['locnet_model'](concat)
            theta_prev = torch.add(theta_new, theta_prev)
            fa_warped = STN(fa, theta_prev, output_size=(self.mri_dims[0], self.mri_dims[1], self.mri_dims[2], 1)
                        , forward_mapping=True)
            return [torch.add(i, 1), theta_prev, t1, fa, fa_warped, stn_iterations]


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        k = self.kernel_size




    def forward(self, x):
        pass