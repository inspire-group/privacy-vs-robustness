import os
import tensorflow as tf
import numpy as np
import math

def PGD_perturb(sess, model, gradient, x, y, num_step, step_size, max_perturb):
    perturb = np.zeros(x.shape)
    #perturb = np.random.uniform(-max_perturb, max_perturb, x.shape)
    for num in range(num_step):
        perturb += step_size*np.sign(sess.run(gradient,feed_dict={model.x_input:x+perturb,model.y_input:y}))        
        perturb = np.clip(perturb,-max_perturb,max_perturb)
        perturb = np.clip(x+perturb, 0, 1.0)-x
    return np.clip(x+perturb,0,1.0)

def softmax_by_row(logits, T = 1.0):
    mx = np.max(logits, axis=-1, keepdims=True)
    exp = np.exp((logits - mx)/T)
    denominator = np.sum(exp, axis=-1, keepdims=True)
    return exp/denominator


def classifier_performance(sess, model, train_data, train_label, test_data, test_label, batch_size, num_step, max_perturb, step_size):
    
    loss= tf.nn.sparse_softmax_cross_entropy_with_logits(logits= model.pre_softmax, labels= model.y_input)
    gradient = tf.gradients(loss, model.x_input)[0]
    
    output_train_benign = []
    output_train_adversarial = []
    for num in range(math.ceil(train_data.shape[0]/batch_size)):
        end_idx = num*batch_size+batch_size
        if end_idx > train_data.shape[0]:
            end_idx = train_data.shape[0]
        input_data = train_data[num*batch_size:end_idx]
        input_label = train_label[num*batch_size:end_idx]
        input_PGD = PGD_perturb(sess, model, gradient, input_data, input_label, num_step, step_size, max_perturb)
        output_train_benign.append(softmax_by_row(sess.run(model.pre_softmax,feed_dict={model.x_input:input_data})))
        output_train_adversarial.append(softmax_by_row(sess.run(model.pre_softmax,feed_dict={model.x_input:input_PGD})))
   
    output_train_benign=np.concatenate(output_train_benign)
    output_train_adversarial=np.concatenate(output_train_adversarial)


    output_test_benign = []
    output_test_adversarial = []
    for num in range(math.ceil(test_data.shape[0]/batch_size)):
        end_idx = num*batch_size+batch_size
        if end_idx > test_data.shape[0]:
            end_idx = test_data.shape[0]
        input_data = test_data[num*batch_size:end_idx]
        input_label = test_label[num*batch_size:end_idx]
        input_PGD = PGD_perturb(sess, model, gradient, input_data, input_label, num_step, step_size, max_perturb)
        output_test_benign.append(softmax_by_row(sess.run(model.pre_softmax,feed_dict={model.x_input:input_data})))
        output_test_adversarial.append(softmax_by_row(sess.run(model.pre_softmax,feed_dict={model.x_input:input_PGD})))
        
    output_test_benign=np.concatenate(output_test_benign)
    output_test_adversarial=np.concatenate(output_test_adversarial)

    train_acc1 = np.sum(np.argmax(output_train_benign,axis=1) == train_label.flatten())/len(train_label)
    train_acc2 = np.sum(np.argmax(output_train_adversarial,axis=1) == train_label.flatten())/len(train_label)
    test_acc1 = np.sum(np.argmax(output_test_benign,axis=1) == test_label.flatten())/len(test_label)
    test_acc2 = np.sum(np.argmax(output_test_adversarial,axis=1) == test_label.flatten())/len(test_label)
    print('Benign accuracy: ', (train_acc1, test_acc1), ' Adversarial accuracy: ', (train_acc2, test_acc2))
    return output_train_benign, output_train_adversarial, output_test_benign, output_test_adversarial