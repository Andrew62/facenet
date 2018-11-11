import tensorflow as tf


def _check_scopes(variable_name, scopes):
    return any(map(lambda x: x in variable_name, scopes))


def _get_uninitialized(sess):
    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)
    return uninitialized_vars


def load_partial_model(sess, graph, checkpoint, exclude_scopes=[], include_scopes=[]):
    to_restore = {}
    to_init = []
    variables = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for variable in variables:
        v_name = variable.name.split(":")[0]
        if not _check_scopes(v_name, exclude_scopes) or _check_scopes(v_name, include_scopes):
            to_restore[v_name] = variable
        else:
            to_init.append(variable)
    saver = tf.train.Saver(var_list=variables)
    model_restore = tf.train.Saver(var_list=to_restore)
    model_restore.restore(sess, checkpoint)
    to_init.extend(_get_uninitialized(sess))
    sess.run(tf.variables_initializer(var_list=to_init))
    return saver
