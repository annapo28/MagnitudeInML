import torch
import torch.nn.functional as F

def mag_loss(epsilon=1e-9):
    def loss_fn(y_true, y_pred):
        diff = y_true - y_pred

        zero_vec = torch.zeros_like(diff[:1, :])
        extended = torch.cat([diff, zero_vec], dim=0)

        distance_matrix = torch.cdist(extended, extended).clamp_min(1e-9)
        similarity_matrix = torch.exp(-distance_matrix)

        N = similarity_matrix.shape[0]
        eye = torch.eye(N, device=similarity_matrix.device, dtype=similarity_matrix.dtype)
        solve_me = torch.ones((N, 1), device=similarity_matrix.device, dtype=similarity_matrix.dtype)

        stabilized_similarity = similarity_matrix + epsilon * eye

        sol = torch.linalg.solve(stabilized_similarity, solve_me)
        magnitude = torch.sum(sol) - 1

        return magnitude
    return loss_fn

def mag_bal_loss(epsilon=1e-9):
    magloss = mag_loss(epsilon)
    
    def loss_fn(y_true, y_pred_logits):
        target_classes = torch.argmax(y_true, dim=-1)

        cce = F.cross_entropy(y_pred_logits, target_classes, reduction='none')
        cce = torch.mean(cce) 

        mag = magloss(y_true, F.softmax(y_pred_logits, dim=-1))

        mag = mag.to(dtype=torch.float32)
        cce = cce.to(dtype=torch.float32)

        return cce * mag
    
    return loss_fn

def mag_loss_tf(epsilon=1e-5):
    import tensorflow as tf

    def loss_fn(y_true, y_pred):
        diff = y_true - y_pred

        zero_vec = tf.zeros_like(diff[:1, :])
        extended = tf.concat([diff, zero_vec], axis=0)

        a = tf.expand_dims(extended, 1)  
        b = tf.expand_dims(extended, 0)  

        squared_diff = tf.reduce_sum(tf.square(a - b), axis=-1)
        distance_matrix = tf.sqrt(tf.maximum(squared_diff, 1e-9))

        similarity_matrix = tf.exp(-distance_matrix)

        N = tf.shape(similarity_matrix)[0]
        eye = tf.eye(N)
        solve_me = tf.ones([N,1])

        stabilized_similarity = similarity_matrix + epsilon * eye

        sol = tf.linalg.solve(stabilized_similarity,solve_me)
        magnitude = tf.reduce_sum(sol) - 1

        return magnitude

    return loss_fn


def spread_bal_loss():
    spreadloss = spread_loss() 
    
    def loss_fn(y_true, y_pred_logits):

        target_classes = torch.argmax(y_true, dim=-1)

        cce = F.cross_entropy(y_pred_logits, target_classes, reduction='none')
        cce = torch.mean(cce)

        spread = spreadloss(y_true, F.softmax(y_pred_logits, dim=-1))

        spread = spread.to(dtype=torch.float32)
        cce = cce.to(dtype=torch.float32)

        return cce * spread
    
    return loss_fn


def spread_loss():
    def loss_fn(y_true, y_pred):
        diff = y_true - y_pred
        zero_vec = torch.zeros_like(diff[:1, :])
        extended = torch.cat([diff, zero_vec], dim=0)

        a = extended.unsqueeze(1) 
        b = extended.unsqueeze(0)  

        squared_diff = torch.sum((a - b) ** 2, dim=-1)
        distance_matrix = torch.sqrt(torch.clamp(squared_diff, min=1e-9))
        similarity_matrix = torch.exp(-distance_matrix)

        crunched_sim = 1.0 / torch.sum(similarity_matrix, dim=1)
        spread = torch.sum(crunched_sim) - 1

        return spread
    return loss_fn

def spread_loss_tf():
    import tensorflow as tf
    def loss_fn(y_true, y_pred):
        diff = y_true-y_pred
        zero_vec = tf.zeros_like(diff[:1, :])
        extended = tf.concat([diff, zero_vec], axis=0)
        a = tf.expand_dims(extended, 1) 
        b = tf.expand_dims(extended, 0) 
        squared_diff = tf.reduce_sum(tf.square(a - b), axis=-1)
        distance_matrix = tf.sqrt(tf.maximum(squared_diff, 1e-9))
        similarity_matrix = tf.exp(-distance_matrix)
        crunched_sim = tf.math.reciprocal(tf.reduce_sum(similarity_matrix, axis=1))
        spread = tf.reduce_sum(crunched_sim)-1
        return spread
    return loss_fn
