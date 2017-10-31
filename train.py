
import os
import time
import numpy as np
import tensorflow as tf
from data import Dataset
from utils import helper
from facenet import FaceNet
from argparse import ArgumentParser
from transfer import load_partial_model


def process_all_images(dataset, network, sess, global_step, args):
    all_images, image_ids = dataset.get_all_files()
    all_images_np, image_ids_np = np.array(all_images), np.array(image_ids)
    all_embeddings = network.inference(sess,
                                       all_images_np,
                                       args.batch_size,
                                       False,
                                       global_step)
    np.savez(os.path.join(args.checkpoint_dir, "embeddings.npz"),
             embeddings=all_embeddings,
             class_codes=image_ids_np)
    return all_embeddings, image_ids_np


def model_train(args):
    image_shape = (160, 160, 3)
    thresholds = np.arange(0, 4, 0.1)
    checkpoint_exclude_scopes = ["InceptionResnetV2/Logits", "InceptionResnetV2/AuxLogits", "RMSProp"]
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("Building graph")
    graph = tf.Graph()
    with graph.as_default():
        image_paths_ph = tf.placeholder(tf.string, name="input_image_paths")
        global_step_ph = tf.placeholder(tf.int32, name="global_step")
        is_training_ph = tf.placeholder(tf.bool, name="is_training")
        network = FaceNet(image_paths_ph,
                          is_training_ph,
                          args.embedding_size,
                          global_step_ph,
                          args.learning_rate,
                          image_shape)

    print("Starting session")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config).as_default() as sess:
        summary_writer = tf.summary.FileWriter(args.checkpoint_dir,
                                               graph=graph)
        saver = load_partial_model(sess,
                                   graph,
                                   checkpoint_exclude_scopes,
                                   args.pretrained_base_model)

        global_step = 0
        start = time.time()
        dataset = Dataset(args.input_faces,
                          n_identities_per=args.identities_per_batch,
                          n_images_per=args.n_images_per_iden,
                          n_eval_pairs=args.n_validation)
        print("Starting loop")
        while global_step < args.train_steps:
            try:
                # embed and collect all current face weights
                image_paths, classes = dataset.get_train_batch()
                embeddings_np = network.inference(sess,
                                                  image_paths,
                                                  args.batch_size,
                                                  True,
                                                  global_step)
                triplets = network.get_triplets(image_paths, embeddings_np, classes)
                n_trips = triplets.shape[0]
                trip_step = args.batch_size - (args.batch_size % 3)
                for idx in range(0, n_trips, trip_step):
                    feed_dict = {
                        image_paths_ph: triplets[idx: idx + trip_step],
                        is_training_ph: True,
                        global_step_ph: global_step
                    }
                    global_step += 1
                    batch_per_sec = (time.time() - start) / global_step
                    if global_step % 100 == 0:
                        summary, _, loss = sess.run([network.merged_summaries, network.optimizer, network.total_loss],
                                                    feed_dict=feed_dict)
                        summary_writer.add_summary(summary, global_step)
                    else:
                        _, loss = sess.run([network.optimizer, network.total_loss], feed_dict=feed_dict)
                    print("global step: {0:,}\tloss: {1:0.5f}\tstep/sec: {2:0.2f}".format(global_step,
                                                                                          loss,
                                                                                          batch_per_sec))
                    if global_step % 1000 == 0:
                        saver.save(sess, args.checkpoint_dir + '/facenet', global_step=global_step)

                        print("Evaluating")
                        start = time.time()
                        evaluation_set = dataset.get_evaluation_batch()
                        (threshold, accuracy, precision, recall, f1) = network.evaluate(sess,
                                                                                        evaluation_set,
                                                                                        global_step,
                                                                                        batch_size=args.batch_size,
                                                                                        thresholds=thresholds)
                        elapsed = time.time() - start
                        print("Accuracy: {0:0.2f}\tThreshold: {1:0.2f}\t".format(accuracy, threshold),
                              "Precision: {0:0.2f}\tRecall: {1:0.2f}\tF-1: {2:0.2f}\t".format(precision, recall, f1),
                              "Elapsed time: {0:0.2f} secs".format(elapsed))

                        all_embeddings, image_ids = process_all_images(dataset, network, sess, global_step, args)

                        for name in ['andrew', 'erin']:
                            person_embed = all_embeddings[image_ids == dataset.name_to_idx[name], :]
                            sim = np.dot(all_embeddings, person_embed[0])
                            sorted_values = np.argsort(sim)[::-1]
                            print("Similar to {0}".format(name.title()))
                            for pls_make_functions in sorted_values[1:6]:
                                print("\t{0} ({1:0.5f})".format(dataset.idx_to_name[image_ids[pls_make_functions]],
                                                                sim[pls_make_functions]))

            except KeyboardInterrupt:
                print("Keyboard Interrupt. Exiting.")
                break
        saver.save(sess, os.path.join(args.checkpoint_dir, 'facenet'), global_step=global_step)
        helper.to_json(dataset.idx_to_name, os.path.join(args.checkpoint_dir, "idx_to_name.json"))
        process_all_images(dataset, network, sess, global_step, args)
        print("Saved to: {0}".format(args.checkpoint_dir))
    print("Done")


def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_faces", help="input faces json file",
                        default="fixtures/faces.json")
    parser.add_argument("-c", "--checkpoint_dir", help="location to write training checkpoints",
                        default="checkpoints/inception_resnet_v2/" + helper.get_current_timestamp())
    parser.add_argument("-b", "--batch_size", default=64, type=int)
    parser.add_argument("-e", "--embedding_size", default=128, type=int)
    parser.add_argument("-l", "--learning_rate", default=0.01, type=float)
    parser.add_argument("-d", "--identities_per_batch", default=100, type=int)
    parser.add_argument("-n", "--n_images_per_iden", default=25, type=int)
    parser.add_argument("-v", "--n_validation", default=3000, type=int)
    parser.add_argument("-p", "--pretrained_base_model",
                        default="checkpoints/pretrained/inception_resnet_v2_2016_08_30.ckpt")
    parser.add_argument("-s", "--train_steps", default=65000, type=int)

    args = parser.parse_args()
    model_train(args)


if __name__ == "__main__":
    main()
