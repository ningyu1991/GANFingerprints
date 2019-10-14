from .model import MMD_GAN, tf, np
from .architecture import get_networks
from .ops import safer_norm

class Cramer_GAN(MMD_GAN):                   
    def build_model(self):
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.lr = tf.Variable(self.config.learning_rate, name='lr', 
                                  trainable=False, dtype=tf.float32)
        self.lr_decay_op = self.lr.assign(tf.maximum(self.lr * self.config.decay_rate, 1.e-6))
        with tf.variable_scope('loss'):
            if self.config.is_train and (self.config.gradient_penalty > 0):
                self.gp = tf.Variable(self.config.gradient_penalty,
                                      name='gradient_penalty',
                                      trainable=False, dtype=tf.float32)
                self.gp_decay_op = self.gp.assign(self.gp * self.config.gp_decay_rate)        

        self.set_pipeline()

        self.sample_z = tf.constant(np.random.uniform(-1, 1, size=(self.sample_size,
                                                      self.z_dim)).astype(np.float32),
                                    dtype=tf.float32, name='sample_z')

        Generator, Discriminator = get_networks(self.config.architecture)
        generator = Generator(self.gf_dim, self.c_dim, self.output_size, self.config.batch_norm)
        dbn = self.config.batch_norm & (self.config.gradient_penalty <= 0)
        self.discriminator = Discriminator(self.df_dim, self.dof_dim, dbn)


        self.G = generator(tf.random_uniform([self.batch_size, self.z_dim], minval=-1.,
                                                   maxval=1., dtype=tf.float32, name='z'),
                           self.batch_size)
        self.G2 = generator(tf.random_uniform([self.batch_size, self.z_dim], minval=-1.,
                                                    maxval=1., dtype=tf.float32, name='z2'),
                           self.batch_size)
        self.sampler = generator(self.sample_z, self.sample_size)
            
        self.d_images_layers = self.discriminator(self.images, self.real_batch_size, return_layers=True)
        self.d_G_layers = self.discriminator(self.G, self.batch_size, return_layers=True)
        self.d_images = self.d_images_layers['hF']
        self.d_G = self.d_G_layers['hF']
        G2 = self.discriminator(self.G2, self.batch_size)

        self.set_loss(self.d_G, G2, self.d_images)

        block = min(8, int(np.sqrt(self.real_batch_size)), int(np.sqrt(self.batch_size)))
        tf.summary.image("train/input image",
                         self.imageRearrange(tf.clip_by_value(self.images, 0, 1), block))
        tf.summary.image("train/gen image",
                         self.imageRearrange(tf.clip_by_value(self.G, 0, 1), block))

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=2)
        
        
    def set_loss(self, G, G2, images):
        bs = min([self.batch_size, self.real_batch_size])
        alpha = tf.random_uniform(shape=[bs])
        alpha = tf.reshape(alpha, [bs, 1, 1, 1])
        real_data = self.images[:bs] #before discirminator
        fake_data = self.G[:bs] #before discriminator
        x_hat_data = (1. - alpha) * real_data + alpha * fake_data
        x_hat = self.discriminator(x_hat_data, bs)
        
        critic = lambda x, x_ : safer_norm(x - x_, axis=1) - safer_norm(x, axis=1) 
        
        with tf.variable_scope('loss'):
            if self.config.model == 'cramer': # Cramer GAN paper
                self.g_loss = tf.reduce_mean(
                    - safer_norm(G - G2, axis=1) + safer_norm(G - images, axis=1) + safer_norm(G2 - images, axis=1))
                self.d_loss = -tf.reduce_mean(critic(images, G) - critic(G2, G))
                to_penalize = critic(x_hat, G)
            elif self.config.model == 'reddit_cramer':
                self.g_loss = tf.reduce_mean(critic(images, G) - critic(G, G2))
                self.d_loss = -self.g_loss
                to_penalize = critic(x_hat, G)
            else:
                raise(AttributeError('wrong model: %s' % self.config.model))
                
            gradients = tf.gradients(to_penalize, [x_hat_data])[0]
            
            penalty = tf.reduce_mean(tf.square(safer_norm(gradients, axis=1) - 1.0))#
        
            self.gp = tf.get_variable('gradient_penalty', dtype=tf.float32,
                                      initializer=self.config.gradient_penalty)
            self.d_loss += penalty * self.gp
            
            self.optim_name = '%s gp %.1f' % (self.config.model, self.config.gradient_penalty)
            tf.summary.scalar(self.optim_name + ' G', self.g_loss)
            tf.summary.scalar(self.optim_name + ' D', self.d_loss)
            tf.summary.scalar('dx_penalty', penalty)
