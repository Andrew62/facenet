import tensorflow as tf


def load_partial_model(sess, graph, exclude_scopes, checkpoint):
    to_restore = {}
    to_init = []
    variables = graph.get_collection("variables")
    for variable in variables:
        v_name = variable.name.split(":")[0]
        if not any(map(lambda x: x in v_name, exclude_scopes)):
            to_restore[v_name] = variable
        else:
            to_init.append(variable)
    saver = tf.train.Saver(var_list=variables)
    model_restore = tf.train.Saver(var_list=to_restore)
    model_restore.restore(sess, checkpoint)
    sess.run(tf.variables_initializer(var_list=to_init))
    return saver
