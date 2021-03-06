import logging
import numpy as np
import tensorflow as tf
from collections import deque
import adda_doom
from tqdm import tqdm
import click
import os
from baselines.a2c.utils import fc
import copy
#import matplotlib.pyplot as plt
#batch_size = 100
#adversary_layers =[500, 500]
#lr = 0.0002

@click.command()
@click.argument('source') # source_dataset.npy
@click.argument('target') # target_dataset.npy
@click.argument('model') # nature_cnn
@click.argument('output') # adda_doom_DA
#@click.option('--iterations', default=20000) # 10k
@click.option('--epochs', default=10) # 10
@click.option('--batch_size', default=50) # 100
@click.option('--display', default=10) #100 
@click.option('--lr', default=1e-4) # 0.0002
@click.option('--stepsize', type=int)
@click.option('--snapshot', default=1) # 1  #####################Run for 10 epochs. Save after every 1############################
#@click.option('--load_fc/--no_load_fc', default=True)
@click.option('--adversary', 'adversary_layers', default=[512, 512],
              multiple=True)


# Step 1: Create base model for ADDA to run on
# Done in cnn.py

def main(source, target, model, output, epochs, batch_size, display, lr, stepsize, snapshot, adversary_layers):

    adda_doom.util.config_logging()

    # Step 2: Import source and target datasets
    source_array = np.load(source) # (100000, 84, 84, 1)
    target_array = np.load(target) # (100000, 84, 84, 1)


    x = tf.placeholder(tf.uint8, shape=[None, 84, 84, 1])
    seed = int.from_bytes(os.urandom(4), byteorder='big')
    # Step 3: Make dataset object using placeholder
    source_dataset = tf.data.Dataset.from_tensor_slices(x).shuffle(buffer_size=100000, seed=seed).batch(batch_size).repeat()
    target_dataset = tf.data.Dataset.from_tensor_slices(x).shuffle(buffer_size=100000, seed=seed).batch(batch_size).repeat()
    print('SEED: ', seed)
    # Step 4: Create generic iterator of correct shape and type
    src_iter = tf.data.Iterator.from_structure(source_dataset.output_types, source_dataset.output_shapes)
    tgt_iter = tf.data.Iterator.from_structure(target_dataset.output_types, target_dataset.output_shapes)
    
    src_imgs = src_iter.get_next()
    tgt_imgs = tgt_iter.get_next()

    # Step 5: Create initialisation operations
    source_iter_op = src_iter.make_initializer(source_dataset)
    target_iter_op = tgt_iter.make_initializer(target_dataset)

    # Step 6: Feed images from both domains into 2 seperate instances of ur base network
    model_fn = adda_doom.model.get_model_fn(model)
    source_ft = model_fn(src_imgs, scope='source')    
    target_ft = model_fn(tgt_imgs, scope='target')

    print('Encoded Source Output Size is:', source_ft.get_shape()) # (batch_size, 512)
    print('Encoded Target Output Size is:', target_ft.get_shape()) # (batch_size, 512)
    print('\n')

    # Step 7: Concat model o/p's to form feature i/p to discriminator()
    adversary_ft = tf.concat([source_ft, target_ft], 0) 

    print('Adversary_ft size is:', adversary_ft.get_shape()) # (2*batch_size, 512)
    print('\n')

    # Step 8: Create source and target adversary label of 0's and 1's resp
    source_adversary_label = tf.zeros([batch_size], tf.int32) 
    target_adversary_label = tf.ones([batch_size], tf.int32)

    print('Source Adversary Label size is:', source_adversary_label.get_shape()) # (batch_size, )
    print('Target Adversary Label size is:', target_adversary_label.get_shape()) # (batch_size, )
    print('\n')


    # Step 9: Concat them to form label i/p to discriminiator()
    adversary_label = tf.concat([source_adversary_label, target_adversary_label], 0)
    
    print('Adversary Label size is:', adversary_label.get_shape()) # (2*batch_size, )
    print('\n')


    # Step 10: Run discriminator() on adversary_ft and adversary_label
    adversary_logits = adda_doom.adversary.discriminator(net=adversary_ft, layers=adversary_layers)
    print('Discriminator output shape is:', adversary_logits.get_shape()) # (2*batch_size, 2)


    # Step 11: Compute losses
    # If logits and 1-label match, means mapping is wrong. Therefore loss increases. Eg, if label=0(source img) and logits=1(target img), in this case, it'll be
    # cross_entropy(1-0, 1)
    mapping_loss = tf.losses.sparse_softmax_cross_entropy( 
        1 - adversary_label, adversary_logits)
    adversary_loss = tf.losses.sparse_softmax_cross_entropy(
        adversary_label, adversary_logits)


    # Step 12: Variable collection. Returns an ordered dict with key as variable name from RL part and value as the variable for src and tgt encoder
    # var: <tf.Variable 'source/c1/w:0' shape=(8, 8, 1, 32) dtype=float32_ref> This wud be a value if run in a session
    # var.name: source/c1/w:0
    # var.op.name: source/c1/w

    '''
    OrderedDict([('model/c1/w', <tf.Variable 'source/c1/w:0' shape=(8, 8, 1, 32) dtype=float32_ref>)])
    '''
    source_vars = adda_doom.util.collect_vars('source', prepend_scope='model')
    target_vars = adda_doom.util.collect_vars('target', prepend_scope='model')

    #print('Before Modifying')
    #print(target_vars)
    #print(len(target_vars))    

    #print('\n')

    #print('After Modifying')
    print('########################')
    print(target_vars.values())
    print('########################')
    print('\n')
    #print(len(target_vars))
    
    adversary_vars = adda_doom.util.collect_vars('adversary')
    print('########################')
    print(adversary_vars.values())
    print('########################')
    #import sys; sys.exit()


    # Step 13: Set the optimizer
    lr_var = tf.Variable(lr, name='learning_rate', trainable=False)
    optimizer = tf.train.RMSPropOptimizer(lr_var)
    mapping_step = optimizer.minimize(
        mapping_loss, var_list=list(target_vars.values()))
    adversary_step = optimizer.minimize(
        adversary_loss, var_list=list(adversary_vars.values()))


    # Step 14: Set up session and initialize
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    # Step 15: Restore weights from RL part to source and target model (Read variable names from saved model first)
    #weights = tf.train.latest_checkpoint('./Snapshots_no_DA/Multiple Snapshots/hg_multiTexture_no_da/Seed 0/')
    weights = tf.train.latest_checkpoint('./Snapshots_no_DA/Multiple Snapshots/hg_normal_no_da/Seed 0/')
    #weights = '/misc/lmbraid18/raob/Snapshots_no_DA/Multiple Snapshots/hg_multiTexture_no_da/Seed 2/hg_multiTexture_no_da_model-48'
    #weights = '/misc/lmbraid18/raob/Snapshots_no_DA/Multiple Snapshots/hg_normal_no_da/Seed 2/hg_normal_no_da_model-28'
    print('WEIGHTS:', weights)
    logging.info('Restoring weights from {}:'.format(weights))

    logging.info('Restoring source model:')

    # Restore both conv layers and fc layer from RL Model
    source_restorer = tf.train.Saver(var_list=source_vars)
    for src, tgt in source_vars.items():
        logging.info('        {:30} -> {:30}'.format(src, tgt.name)) # model/c1/w -> source/c1/w:0 etc

    print('Source Weights before restoring',sess.run(source_vars['model/fc1/b']))
    source_restorer.restore(sess, weights)
    print('Source Weights after restoring',sess.run(source_vars['model/fc1/b']))

    

    logging.info('Restoring target model:')

    # Restore both conv layers and fc layer from RL Model
    target_restorer = tf.train.Saver(var_list=target_vars, max_to_keep=41)
    for src, tgt in target_vars.items():
        logging.info('        {:30} -> {:30}'.format(src, tgt.name)) # model/c1/w -> source/c1/w:0 etc
        
    print('Target Weights before restoring',sess.run(target_vars['model/fc1/b']))
    target_restorer.restore(sess, weights)
    print('Target Weights after restoring',sess.run(target_vars['model/fc1/b']))

    print('Weights Restored Successfully')

    #import sys; sys.exit()





    # Step 16: Setup Optimization Loop
    output_dir = os.path.join('snapshot', output)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    mapping_losses = deque(maxlen=10)
    adversary_losses = deque(maxlen=10)
    
    avg_mapping_losses = []
    avg_adversary_losses = []
    
    bar = tqdm(range(epochs))
    bar.set_description('{} (lr: {:.0e})'.format(output, lr))
    bar.refresh()
    
    # Initialize iterators
    sess.run(source_iter_op, feed_dict={x:source_array})
    sess.run(target_iter_op, feed_dict={x:target_array})
    print('################################################', sess.run(tf.shape(src_imgs)))
    print('################################################', sess.run(tf.shape(tgt_imgs)))  
    print('#################### SOURCE_FT SHAPE ############################', sess.run(tf.shape(source_ft)))
    print('#################### TARGET_FT SHAPE ############################', sess.run(tf.shape(target_ft)))
    print('#################### COMBINED SHAPE ############################', sess.run(tf.shape(adversary_ft)))
    print('#################### DISCRIMINTAOR OUTPUT SHAPE ############################', sess.run(tf.shape(adversary_logits)))
    print('################################## TRAINING BEGINS ##################################')
    for ep in bar:
        print('#######EPOCH {}#######'.format(ep+1))
        for i in range(100000//batch_size):
            mapping_loss_val, adversary_loss_val, _, _ = sess.run([mapping_loss, adversary_loss, mapping_step, adversary_step])
            #print('Adversary Loss: ', adversary_loss_val)
            #print(i)
            mapping_losses.append(mapping_loss_val)
            adversary_losses.append(adversary_loss_val)

            avg_mapping_losses.append(mapping_loss_val)
            avg_adversary_losses.append(adversary_loss_val)
            '''
            if i % display == 0:
                logging.info('{:20} Mapping: {:10.4f}     (avg: {:10.4f})'
                            '    Adversary: {:10.4f}     (avg: {:10.4f})'
                            .format('Iteration {}:'.format(i),
                                    mapping_loss_val,
                                    np.mean(mapping_losses),
                                    adversary_loss_val,
                                    np.mean(adversary_losses)))
            #print(np.mean(mapping_losses))
                #avg_mapping_losses.append(np.mean(mapping_losses))
                #avg_adversary_losses.append(np.mean(adversary_losses))
            '''
        print('EPOCH: {} -> Mapping Loss: {:4f} Adversary Loss: {:.4f}'.format(ep+1, np.mean(mapping_losses), np.mean(adversary_losses)))
        
        if stepsize is not None and (ep + 1) % stepsize == 0:
            lr = sess.run(lr_var.assign(lr * 0.1))
            logging.info('Changed learning rate to {:.0e}'.format(lr))
            bar.set_description('{} (lr: {:.0e})'.format(output, lr))
        
        if (ep + 1) % snapshot == 0:
            snapshot_path = target_restorer.save(
                sess, os.path.join(output_dir, output), global_step=ep + 1, write_meta_graph=False) 
            logging.info('Saved snapshot to {}'.format(snapshot_path))

    print('################################## TRAINING ENDS ##################################')
    sess.close()



if __name__ == '__main__':
    main()





