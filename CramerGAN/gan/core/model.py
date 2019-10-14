from __future__ import division, print_function
import os, sys, time, pprint, numpy as np
from . import  mmd
from .ops import safer_norm, tf
from .architecture import get_networks
from .pipeline import get_pipeline
from utils import timer, scorer, misc 

class MMD_GAN(object):
    def __init__(self, sess, config, 
                 batch_size=64, output_size=64,
                 z_dim=100, c_dim=3, data_dir='./data'):
        if config.learning_rate_D < 0:
            config.learning_rate_D = config.learning_rate
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [64]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.timer = timer.Timer()
        self.dataset = config.dataset
        if config.architecture == 'dc128':
            output_size = 128
        if config.architecture in ['dc64', 'dcgan64']:
            output_size = 64
            
        self.sess = sess
        if config.real_batch_size == -1:
            config.real_batch_size = config.batch_size
        self.config = config
        self.is_grayscale = (c_dim == 1)
        self.batch_size = batch_size
        self.real_batch_size = config.real_batch_size
        self.sample_size = 64 if self.config.is_train else batch_size
        self.output_size = output_size
        self.data_dir = data_dir
        self.z_dim = z_dim

        self.gf_dim = config.gf_dim
        self.df_dim = config.df_dim
        self.dof_dim = self.config.dof_dim

        self.c_dim = c_dim            
        
        discriminator_desc = '_dc'
        if self.config.learning_rate_D == self.config.learning_rate:
            lr = 'lr%.8f' % self.config.learning_rate
        else:
            lr = 'lr%.8fG%fD' % (self.config.learning_rate, self.config.learning_rate_D)
        arch = '%dx%d' % (self.config.gf_dim, self.config.df_dim)
        
        self.description = ("%s%s_%s%s_%sd%d-%d-%d_%s_%s_%s" % (
                    self.dataset, arch,
                    self.config.architecture, discriminator_desc,
                    self.config.kernel, self.config.dsteps,
                    self.config.start_dsteps, self.config.gsteps, self.batch_size,
                    self.output_size, lr))
        
        if self.config.batch_norm:
            self.description += '_bn'
        
        self._ensure_dirs()
        
        stdout = sys.stdout
        if self.config.log:
            self.old_stdout = sys.stdout
            self.old_stderr = sys.stderr
            self.log_file = open(os.path.join(self.sample_dir, 'log.txt'), 'w', buffering=1)
            print('Execution start time: %s' % time.ctime())
            print('Log file: %s' % self.log_file)
            stdout = self.log_file
            sys.stdout = self.log_file
            sys.stderr = self.log_file
        if config.compute_scores:
            self.scorer = scorer.Scorer(self.dataset, config.MMD_lr_scheduler, stdout=stdout)
        print('Execution start time: %s' % time.ctime())
        #pprint.PrettyPrinter().pprint(self.config.__dict__['__flags'])
        self.build_model()
        
        self.initialized_for_sampling = config.is_train

    def _ensure_dirs(self, folders=['sample', 'log', 'checkpoint']):
        if type(folders) == str:
            folders = [folders]
        for folder in folders:
            ff = folder + '_dir'
            if not os.path.exists(ff):
                os.makedirs(ff)
            self.__dict__[ff] = os.path.join(self.config.__getattr__(ff),
                                             self.config.name + self.config.suffix,
                                             self.description)
            if not os.path.exists(self.__dict__[ff]):
                os.makedirs(self.__dict__[ff])
            

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

        self.z = tf.random_uniform([self.batch_size, self.z_dim], minval=-1., 
                                   maxval=1., dtype=tf.float32, name='z')
        self.sample_z = tf.constant(np.random.uniform(-1, 1, size=(self.sample_size, 
                                                      self.z_dim)).astype(np.float32),
                                    dtype=tf.float32, name='sample_z')        

        Generator, Discriminator = get_networks(self.config.architecture)
        generator = Generator(self.gf_dim, self.c_dim, self.output_size, self.config.batch_norm)
        dbn = self.config.batch_norm & (self.config.gradient_penalty <= 0)
        self.discriminator = Discriminator(self.df_dim, self.dof_dim, dbn)
        # tf.summary.histogram("z", self.z)

        self.G = generator(self.z, self.batch_size)

        self.sampler = generator(self.sample_z, self.sample_size)
        
        self.d_images_layers = self.discriminator(self.images, 
            self.real_batch_size, return_layers=True)
        self.d_G_layers = self.discriminator(self.G, self.batch_size,
                                             return_layers=True)
        self.d_images = self.d_images_layers['hF']
        self.d_G = self.d_G_layers['hF']
            
        if self.config.is_train:
            self.set_loss(self.d_G, self.d_images)

        block = min(8, int(np.sqrt(self.real_batch_size)), int(np.sqrt(self.batch_size)))
        tf.summary.image("train/input image", 
                         self.imageRearrange(tf.clip_by_value(self.images, 0, 1), block))
        tf.summary.image("train/gen image", 
                         self.imageRearrange(tf.clip_by_value(self.G, 0, 1), block))
        
        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=2)
            
        print('[*] Model built.')


    def set_loss(self, G, images):
        kernel = getattr(mmd, '_%s_kernel' % self.config.kernel)
        kerGI = kernel(G, images)
            
        with tf.variable_scope('loss'):
            self.g_loss = mmd.mmd2(kerGI)
            self.d_loss = -self.g_loss 
            self.optim_name = 'kernel_loss'
            
        self.add_gradient_penalty(kernel, G, images)
        self.add_l2_penalty()
        
        print('[*] Loss set')


    def add_gradient_penalty(self, kernel, fake, real):
        bs = min([self.batch_size, self.real_batch_size])
        real, fake = real[:bs], fake[:bs]
        
        alpha = tf.random_uniform(shape=[bs, 1, 1, 1])
        real_data = self.images[:bs] # discirminator input level
        fake_data = self.G[:bs] # discriminator input level
        x_hat_data = (1. - alpha) * real_data + alpha * fake_data
        x_hat = self.discriminator(x_hat_data, bs)
        Ekx = lambda yy: tf.reduce_mean(kernel(x_hat, yy, K_XY_only=True), axis=1)
        Ekxr, Ekxf = Ekx(real), Ekx(fake)
        witness = Ekxr - Ekxf
        gradients = tf.gradients(witness, [x_hat_data])[0]
        
        penalty = tf.reduce_mean(tf.square(safer_norm(gradients, axis=1) - 1.0))

        with tf.variable_scope('loss'):
            if self.config.gradient_penalty > 0:
                self.d_loss += penalty * self.gp
                self.optim_name += ' (gp %.1f)' % self.config.gradient_penalty
                tf.summary.scalar('dx_penalty', penalty)
                print('[*] Gradient penalty added')
            tf.summary.scalar(self.optim_name + ' G', self.g_loss)
            tf.summary.scalar(self.optim_name + ' D', self.d_loss)
    
    
    def add_l2_penalty(self):
        if self.config.L2_discriminator_penalty > 0:
            penalty = 0.0
            for _, layer in self.d_G_layers.items():
                penalty += tf.reduce_mean(tf.reshape(tf.square(layer), [self.batch_size, -1]), axis=1)
            for _, layer in self.d_images_layers.items():
                penalty += tf.reduce_mean(tf.reshape(tf.square(layer), [self.batch_size, -1]), axis=1)
            self.d_L2_penalty = self.config.L2_discriminator_penalty * tf.reduce_mean(penalty)
            self.d_loss += self.d_L2_penalty
            self.optim_name += ' (L2 dp %.6f)' % self.config.L2_discriminator_penalty
            self.optim_name = self.optim_name.replace(') (', ', ')
            tf.summary.scalar('L2_disc_penalty', self.d_L2_penalty)
            print('[*] L2 discriminator penalty added')
        
        
    def set_grads(self):
        with tf.variable_scope("G_grads"):
            self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=self.config.beta1, beta2=0.9)
            self.g_gvs = self.g_optim.compute_gradients(
                loss=self.g_loss,
                var_list=self.g_vars
            )       
            self.g_gvs = [(tf.clip_by_norm(gg, 1.), vv) for gg, vv in self.g_gvs]
            self.g_grads = self.g_optim.apply_gradients(
                self.g_gvs, 
                global_step=self.global_step
            ) # minimizes self.g_loss <==> minimizes MMD

        with tf.variable_scope("D_grads"):
            self.d_optim = tf.train.AdamOptimizer(
                self.lr * self.config.learning_rate_D / self.config.learning_rate, 
                beta1=self.config.beta1, beta2=0.9
            )
            self.d_gvs = self.d_optim.compute_gradients(
                loss=self.d_loss, 
                var_list=self.d_vars
            )
            # negative gradients not needed - by definition d_loss = -optim_loss
            self.d_gvs = [(tf.clip_by_norm(gg, 1.), vv) for gg, vv in self.d_gvs]
            self.d_grads = self.d_optim.apply_gradients(self.d_gvs) # minimizes self.d_loss <==> max MMD    
        print('[*] Gradients set')
    
    def train_step(self, batch_images=None):
        step = self.sess.run(self.global_step)
        write_summary = ((np.mod(step, 50) == 0) and (step < 1000)) \
                or (np.mod(step, 1000) == 0) or (self.err_counter > 0)

        if (self.g_counter == 0) and (self.d_grads is not None):
            d_steps = self.config.dsteps
            if ((step % 500 == 0) or (step < 20)):
                d_steps = self.config.start_dsteps
            self.d_counter = (self.d_counter + 1) % (d_steps + 1)
        if self.d_counter == 0:
            self.g_counter = (self.g_counter + 1) % self.config.gsteps        

        eval_ops = [self.g_gvs, self.d_gvs, self.g_loss, self.d_loss]
        if self.config.is_demo:
            summary_str, g_grads, d_grads, g_loss, d_loss = self.sess.run(
                [self.TrainSummary] + eval_ops
            )
        else:
            if self.d_counter == 0:
                if write_summary:
                    _, summary_str, g_grads, d_grads, g_loss, d_loss = self.sess.run(
                        [self.g_grads, self.TrainSummary] + eval_ops
                    )
                else:
                    _, g_grads, d_grads, g_loss, d_loss = self.sess.run([self.g_grads] + eval_ops)
            else:
                _, g_grads, d_grads, g_loss, d_loss = self.sess.run([self.d_grads] + eval_ops)
            et = self.timer(step, "g step" if (self.d_counter == 0) else "d step", False)

        assert ~np.isnan(g_loss), et + "NaN g_loss, epoch: "
        assert ~np.isnan(d_loss), et + "NaN d_loss, epoch: "
        # if G STEP, after D steps
        if self.d_counter == 0:
            if step % 10000 == 0:
                try:
                    self.writer.add_summary(summary_str, step)
                    self.err_counter = 0
                except Exception as e:
                    print('Step %d summary exception. ' % step, e)
                    self.err_counter += 1
            if write_summary:
                self.timer(step, "%s, G: %.8f, D: %.8f" % (self.optim_name, g_loss, d_loss))
                if self.config.L2_discriminator_penalty > 0:
                    print(' ' * 22 + ('Discriminator L2 penalty: %.8f' % self.sess.run(self.d_L2_penalty)))
            if np.mod(step + 1, self.config.max_iteration//5) == 0:
                if not self.config.MMD_lr_scheduler:
#                    self.lr *= self.config.decay_rate
                    self.sess.run(self.lr_decay_op)
                    print('current learning rate: %f' % self.sess.run(self.lr))
                if (self.config.gp_decay_rate > 0) and (self.config.gradient_penalty > 0):
                    self.sess.run(self.gp_decay_op)
                    print('current gradient penalty: %f' % self.sess.run(self.gp))
        
            if self.config.compute_scores:
                self.scorer.compute(self, step)
        return g_loss, d_loss, step
      

    def train_init(self):
        self.set_grads()

        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())
        print('[*] Variables initialized.')
        
        self.TrainSummary = tf.summary.merge_all()
        
        self._ensure_dirs('log')
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        self.d_counter, self.g_counter, self.err_counter = 0, 0, 0
        
        if self.load_checkpoint():
            print(""" [*] Load SUCCESS, re-starting at epoch %d with learning
                  rate %.7f""" % (self.sess.run(self.global_step), 
                                  self.sess.run(self.lr)))
        else:
            print(" [!] Load failed...")
#        self.sess.run(self.lr.assign(self.config.learning_rate))
        if (not self.config.MMD_lr_scheduler) and (self.sess.run(self.gp) == self.config.gradient_penalty):
            step = self.sess.run(self.global_step)
            lr_decays_so_far = int((step * 5.)/self.config.max_iteration)
            self.lr *= self.config.decay_rate ** lr_decays_so_far
            if self.config.gp_decay_rate > 0:
                self.gp *= self.config.gp_decay_rate ** lr_decays_so_far
                print('current gradient penalty: %f' % self.sess.run(self.gp))
        print('current learning rate: %f' % self.sess.run(self.lr))    
        
        print('[*] Model initialized for training')
            
            
    def set_pipeline(self):
        Pipeline = get_pipeline(self.dataset, self.config.suffix)
        pipe = Pipeline(self.output_size, self.c_dim, self.real_batch_size, 
                        self.data_dir, 
                        timer=self.timer, sample_dir=self.sample_dir)
        self.images = pipe.connect()        

            
    def train(self):    
        self.train_init()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)        
        step = 0
        
        print('[ ] Training ... ')
        while step <= self.config.max_iteration:
            g_loss, d_loss, step = self.train_step()
            self.save_checkpoint_and_samples(step)
            if self.config.save_layer_outputs:
                self.save_layers(step)            
        coord.request_stop()
        coord.join(threads)


    def save_checkpoint(self, step=None):
        self._ensure_dirs('checkpoint')
        if step is None:
            self.saver.save(self.sess,
                            os.path.join(self.checkpoint_dir, "best.model"))
        else:
            self.saver.save(self.sess,
                            os.path.join(self.checkpoint_dir, "MMDGAN.model"),
                            global_step=step)


    def load_checkpoint(self):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
            return True
        else:
            return False


    def save_checkpoint_and_samples(self, step, freq=1000):
        if (np.mod(step, freq) == 0) and (self.d_counter == 0):
            self.save_checkpoint(step)
            samples = self.sess.run(self.sampler)
            self._ensure_dirs('sample')
            p = os.path.join(self.sample_dir, 'train_{:02d}.png'.format(step))
            misc.save_images(samples[:64, :, :, :], [8, 8], p)  
            
            
    def save_layers(self, step, freq=1000, n=256, layers=[-1, -2]):
        c = self.config.save_layer_outputs
        valid = list(freq * np.arange(self.config.max_iteration/freq + 1))
        if c > 1:
            valid += [int(k) for k in c**np.arange(np.log(freq)/np.log(c))]
        if (step in valid) and (self.d_counter == 0):
            if not (layers == 'all'):
                keys = [sorted(list(self.d_G_layers))[i] for i in layers]
            fake = [(key + '_fake', self.d_G_layers[key]) for key in keys] 
            real = [(key + '_real', self.d_images_layers[key]) for key in keys]
            
            values = self._evaluate_tensors(dict(real + fake), n=n)    
            path = os.path.join(self.sample_dir, 'layer_outputs_%d.npz' % step)
            np.savez(path, **values)
        
        
    def imageRearrange(self, image, block=4):
        image = tf.slice(image, [0, 0, 0, 0], [block * block, -1, -1, -1])
        x1 = tf.batch_to_space(image, [[0, 0], [0, 0]], block)
        image_r = tf.reshape(tf.transpose(tf.reshape(x1,
            [self.output_size, block, self.output_size, block, self.c_dim])
            , [1, 0, 3, 2, 4]),
            [1, self.output_size * block, self.output_size * block, self.c_dim])
        return image_r

        
    def _evaluate_tensors(self, variable_dict, n=None):
        if n is None:
            n = self.batch_size
        values = dict([(key, []) for key in variable_dict.keys()])
        sampled = 0
        while sampled < n:
            vv = self.sess.run(variable_dict)
            for key, val in vv.items():
                values[key].append(val)
            sampled += list(vv.items())[0][1].shape[0]
        for key, val in values.items():
            values[key] = np.concatenate(val, axis=0)[:n]        
        return values
        
    
    def get_samples(self, n=None, save=True, layers=[]):
        if not (self.initialized_for_sampling or self.config.is_train):
            print('[*] Loading from ' + self.checkpoint_dir + '...')
            self.sess.run(tf.local_variables_initializer())
            self.sess.run(tf.global_variables_initializer())
            if self.load_checkpoint():
                print(" [*] Load SUCCESS, model trained up to epoch %d" % \
                      self.sess.run(self.global_step))
            else:
                print(" [!] Load failed...")
                return
    
        if len(layers) > 0:
            outputs = dict([(key + '_features', val) for key, val in self.d_G_layers.items()])
            if not (layers == 'all'):
                keys = [sorted(list(outputs.keys()))[i] for i in layers]
                outputs = dict([(key, outputs[key]) for key in keys])
        else:
            outputs = {}
        outputs['samples'] = self.G

        values = self._evaluate_tensors(outputs, n=n)
        
        if not save:
            if len(layers) > 0:
                return values
            return values['samples']
        
        if not os.path.isdir(self.config.output_dir_of_test_samples):
            os.mkdir(self.config.output_dir_of_test_samples)
        for key, val in values.items(): 
            if key == 'samples':  
                for idx in range(val.shape[0]):
                    print('Generating png to %s: %d / %d...' % (self.config.output_dir_of_test_samples, idx, val.shape[0]), end='\r')
                    if self.config.model == 'mmd':
                        p = os.path.join(self.config.output_dir_of_test_samples, 'MMD_{:08d}.png'.format(idx))
                    elif self.config.model == 'wgan_gp':
                        p = os.path.join(self.config.output_dir_of_test_samples, 'WGAN-GP_{:08d}.png'.format(idx))
                    elif self.config.model == 'cramer':
                        p = os.path.join(self.config.output_dir_of_test_samples, 'CRAMER_{:08d}.png'.format(idx))
                    misc.save_images(val[idx:idx+1, :, :, :], [1, 1], p)         