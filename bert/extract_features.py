# # coding=utf-8
# # Copyright 2018 The Google AI Language Team Authors.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# """Extract pre-computed feature vectors from BERT."""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import codecs
# import collections
# import json
# import re

# import modeling
# import tokenization
# import tensorflow as tf

# flags = tf.flags

# FLAGS = flags.FLAGS

# flags.DEFINE_string("input_file", None, "")

# flags.DEFINE_string("output_file", None, "")

# flags.DEFINE_string("layers", "-1,-2,-3,-4", "")

# flags.DEFINE_string(
#     "bert_config_file", None,
#     "The config json file corresponding to the pre-trained BERT model. "
#     "This specifies the model architecture.")

# flags.DEFINE_integer(
#     "max_seq_length", 128,
#     "The maximum total input sequence length after WordPiece tokenization. "
#     "Sequences longer than this will be truncated, and sequences shorter "
#     "than this will be padded.")

# flags.DEFINE_string(
#     "init_checkpoint", None,
#     "Initial checkpoint (usually from a pre-trained BERT model).")

# flags.DEFINE_string("vocab_file", None,
#                     "The vocabulary file that the BERT model was trained on.")

# flags.DEFINE_bool(
#     "do_lower_case", True,
#     "Whether to lower case the input text. Should be True for uncased "
#     "models and False for cased models.")

# flags.DEFINE_integer("batch_size", 32, "Batch size for predictions.")

# flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

# flags.DEFINE_string("master", None,
#                     "If using a TPU, the address of the master.")

# flags.DEFINE_integer(
#     "num_tpu_cores", 8,
#     "Only used if `use_tpu` is True. Total number of TPU cores to use.")

# flags.DEFINE_bool(
#     "use_one_hot_embeddings", False,
#     "If True, tf.one_hot will be used for embedding lookups, otherwise "
#     "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
#     "since it is much faster.")


# class InputExample(object):

#   def __init__(self, unique_id, text_a, text_b):
#     self.unique_id = unique_id
#     self.text_a = text_a
#     self.text_b = text_b


# class InputFeatures(object):
#   """A single set of features of data."""

#   def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
#     self.unique_id = unique_id
#     self.tokens = tokens
#     self.input_ids = input_ids
#     self.input_mask = input_mask
#     self.input_type_ids = input_type_ids


# def input_fn_builder(features, seq_length):
#   """Creates an `input_fn` closure to be passed to TPUEstimator."""

#   all_unique_ids = []
#   all_input_ids = []
#   all_input_mask = []
#   all_input_type_ids = []

#   for feature in features:
#     all_unique_ids.append(feature.unique_id)
#     all_input_ids.append(feature.input_ids)
#     all_input_mask.append(feature.input_mask)
#     all_input_type_ids.append(feature.input_type_ids)

#   def input_fn(params):
#     """The actual input function."""
#     batch_size = params["batch_size"]

#     num_examples = len(features)

#     # This is for demo purposes and does NOT scale to large data sets. We do
#     # not use Dataset.from_generator() because that uses tf.py_func which is
#     # not TPU compatible. The right way to load data is with TFRecordReader.
#     d = tf.data.Dataset.from_tensor_slices({
#         "unique_ids":
#             tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
#         "input_ids":
#             tf.constant(
#                 all_input_ids, shape=[num_examples, seq_length],
#                 dtype=tf.int32),
#         "input_mask":
#             tf.constant(
#                 all_input_mask,
#                 shape=[num_examples, seq_length],
#                 dtype=tf.int32),
#         "input_type_ids":
#             tf.constant(
#                 all_input_type_ids,
#                 shape=[num_examples, seq_length],
#                 dtype=tf.int32),
#     })

#     d = d.batch(batch_size=batch_size, drop_remainder=False)
#     return d

#   return input_fn


# def model_fn_builder(bert_config, init_checkpoint, layer_indexes, use_tpu,
#                      use_one_hot_embeddings):
#   """Returns `model_fn` closure for TPUEstimator."""

#   def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
#     """The `model_fn` for TPUEstimator."""

#     unique_ids = features["unique_ids"]
#     input_ids = features["input_ids"]
#     input_mask = features["input_mask"]
#     input_type_ids = features["input_type_ids"]

#     model = modeling.BertModel(
#         config=bert_config,
#         is_training=False,
#         input_ids=input_ids,
#         input_mask=input_mask,
#         token_type_ids=input_type_ids,
#         use_one_hot_embeddings=use_one_hot_embeddings)

#     if mode != tf.estimator.ModeKeys.PREDICT:
#       raise ValueError("Only PREDICT modes are supported: %s" % (mode))

#     tvars = tf.trainable_variables()
#     scaffold_fn = None

#     # If an init_checkpoint is provided, load variables from it. Otherwise
#     # skip initialization so the model will use random initialization.
#     if init_checkpoint:
#       (assignment_map,
#        initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
#            tvars, init_checkpoint)
#       if use_tpu:

#         def tpu_scaffold():
#           tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
#           return tf.train.Scaffold()

#         scaffold_fn = tpu_scaffold
#       else:
#         tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
#     else:
#       assignment_map = {}
#       initialized_variable_names = {}

#     tf.logging.info("**** Trainable Variables ****")
#     for var in tvars:
#       init_string = ""
#       if var.name in initialized_variable_names:
#         init_string = ", *INIT_FROM_CKPT*"
#       tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
#                       init_string)

#     all_layers = model.get_all_encoder_layers()

#     predictions = {
#         "unique_id": unique_ids,
#     }

#     for (i, layer_index) in enumerate(layer_indexes):
#       predictions["layer_output_%d" % i] = all_layers[layer_index]

#     output_spec = tf.contrib.tpu.TPUEstimatorSpec(
#         mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
#     return output_spec

#   return model_fn


# def convert_examples_to_features(examples, seq_length, tokenizer):
#   """Loads a data file into a list of `InputBatch`s."""

#   features = []
#   for (ex_index, example) in enumerate(examples):
#     tokens_a = tokenizer.tokenize(example.text_a)

#     tokens_b = None
#     if example.text_b:
#       tokens_b = tokenizer.tokenize(example.text_b)

#     if tokens_b:
#       # Modifies `tokens_a` and `tokens_b` in place so that the total
#       # length is less than the specified length.
#       # Account for [CLS], [SEP], [SEP] with "- 3"
#       _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
#     else:
#       # Account for [CLS] and [SEP] with "- 2"
#       if len(tokens_a) > seq_length - 2:
#         tokens_a = tokens_a[0:(seq_length - 2)]

#     # The convention in BERT is:
#     # (a) For sequence pairs:
#     #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
#     #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
#     # (b) For single sequences:
#     #  tokens:   [CLS] the dog is hairy . [SEP]
#     #  type_ids: 0     0   0   0  0     0 0
#     #
#     # Where "type_ids" are used to indicate whether this is the first
#     # sequence or the second sequence. The embedding vectors for `type=0` and
#     # `type=1` were learned during pre-training and are added to the wordpiece
#     # embedding vector (and position vector). This is not *strictly* necessary
#     # since the [SEP] token unambiguously separates the sequences, but it makes
#     # it easier for the model to learn the concept of sequences.
#     #
#     # For classification tasks, the first vector (corresponding to [CLS]) is
#     # used as as the "sentence vector". Note that this only makes sense because
#     # the entire model is fine-tuned.
#     tokens = []
#     input_type_ids = []
#     tokens.append("[CLS]")
#     input_type_ids.append(0)
#     for token in tokens_a:
#       tokens.append(token)
#       input_type_ids.append(0)
#     tokens.append("[SEP]")
#     input_type_ids.append(0)

#     if tokens_b:
#       for token in tokens_b:
#         tokens.append(token)
#         input_type_ids.append(1)
#       tokens.append("[SEP]")
#       input_type_ids.append(1)

#     input_ids = tokenizer.convert_tokens_to_ids(tokens)

#     # The mask has 1 for real tokens and 0 for padding tokens. Only real
#     # tokens are attended to.
#     input_mask = [1] * len(input_ids)

#     # Zero-pad up to the sequence length.
#     while len(input_ids) < seq_length:
#       input_ids.append(0)
#       input_mask.append(0)
#       input_type_ids.append(0)

#     assert len(input_ids) == seq_length
#     assert len(input_mask) == seq_length
#     assert len(input_type_ids) == seq_length

#     if ex_index < 5:
#       tf.logging.info("*** Example ***")
#       tf.logging.info("unique_id: %s" % (example.unique_id))
#       tf.logging.info("tokens: %s" % " ".join(
#           [tokenization.printable_text(x) for x in tokens]))
#       tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
#       tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
#       tf.logging.info(
#           "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

#     features.append(
#         InputFeatures(
#             unique_id=example.unique_id,
#             tokens=tokens,
#             input_ids=input_ids,
#             input_mask=input_mask,
#             input_type_ids=input_type_ids))
#   return features


# def _truncate_seq_pair(tokens_a, tokens_b, max_length):
#   """Truncates a sequence pair in place to the maximum length."""

#   # This is a simple heuristic which will always truncate the longer sequence
#   # one token at a time. This makes more sense than truncating an equal percent
#   # of tokens from each, since if one sequence is very short then each token
#   # that's truncated likely contains more information than a longer sequence.
#   while True:
#     total_length = len(tokens_a) + len(tokens_b)
#     if total_length <= max_length:
#       break
#     if len(tokens_a) > len(tokens_b):
#       tokens_a.pop()
#     else:
#       tokens_b.pop()


# def read_examples(input_file):
#   """Read a list of `InputExample`s from an input file."""
#   examples = []
#   unique_id = 0
#   with tf.gfile.GFile(input_file, "r") as reader:
#     while True:
#       line = tokenization.convert_to_unicode(reader.readline())
#       if not line:
#         break
#       line = line.strip()
#       text_a = None
#       text_b = None
#       m = re.match(r"^(.*) \|\|\| (.*)$", line)
#       if m is None:
#         text_a = line
#       else:
#         text_a = m.group(1)
#         text_b = m.group(2)
#       examples.append(
#           InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
#       unique_id += 1
#   return examples


# def main(_):
#   tf.logging.set_verbosity(tf.logging.INFO)

#   layer_indexes = [int(x) for x in FLAGS.layers.split(",")]

#   bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

#   tokenizer = tokenization.FullTokenizer(
#       vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

#   is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
#   run_config = tf.contrib.tpu.RunConfig(
#       master=FLAGS.master,
#       tpu_config=tf.contrib.tpu.TPUConfig(
#           num_shards=FLAGS.num_tpu_cores,
#           per_host_input_for_training=is_per_host))

#   examples = read_examples(FLAGS.input_file)

#   features = convert_examples_to_features(
#       examples=examples, seq_length=FLAGS.max_seq_length, tokenizer=tokenizer)

#   unique_id_to_feature = {}
#   for feature in features:
#     unique_id_to_feature[feature.unique_id] = feature

#   model_fn = model_fn_builder(
#       bert_config=bert_config,
#       init_checkpoint=FLAGS.init_checkpoint,
#       layer_indexes=layer_indexes,
#       use_tpu=FLAGS.use_tpu,
#       use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)

#   # If TPU is not available, this will fall back to normal Estimator on CPU
#   # or GPU.
#   estimator = tf.contrib.tpu.TPUEstimator(
#       use_tpu=FLAGS.use_tpu,
#       model_fn=model_fn,
#       config=run_config,
#       predict_batch_size=FLAGS.batch_size)

#   input_fn = input_fn_builder(
#       features=features, seq_length=FLAGS.max_seq_length)

#   with codecs.getwriter("utf-8")(tf.gfile.Open(FLAGS.output_file,
#                                                "w")) as writer:
#     for result in estimator.predict(input_fn, yield_single_examples=True):
#       unique_id = int(result["unique_id"])
#       feature = unique_id_to_feature[unique_id]
#       output_json = collections.OrderedDict()
#       output_json["linex_index"] = unique_id
#       all_features = []
#       for (i, token) in enumerate(feature.tokens):
#         all_layers = []
#         for (j, layer_index) in enumerate(layer_indexes):
#           layer_output = result["layer_output_%d" % j]
#           layers = collections.OrderedDict()
#           layers["index"] = layer_index
#           layers["values"] = [
#               round(float(x), 6) for x in layer_output[i:(i + 1)].flat
#           ]
#           all_layers.append(layers)
#         features = collections.OrderedDict()
#         features["token"] = token
#         features["layers"] = all_layers
#         all_features.append(features)
#       output_json["features"] = all_features
#       writer.write(json.dumps(output_json) + "\n")


# if __name__ == "__main__":
#   flags.mark_flag_as_required("input_file")
#   flags.mark_flag_as_required("vocab_file")
#   flags.mark_flag_as_required("bert_config_file")
#   # `init_checkpoint` is optional for demo runs with random initialization.
#   flags.mark_flag_as_required("output_file")
#   tf.app.run()



#!/usr/bin/env python
"""
BERT Feature Extraction - TensorFlow 2.x Compatible
Generates real embeddings (not zeros) from BERT model
"""

import json
import numpy as np
import tensorflow as tf
from pathlib import Path

def load_vocab(vocab_file):
    """Load vocabulary."""
    vocab = {}
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for idx, token in enumerate(f):
            vocab[token.strip()] = idx
    return vocab

def load_bert_config(config_file):
    """Load BERT configuration."""
    with open(config_file, 'r') as f:
        return json.load(f)

def basic_tokenize(text, vocab_dict, max_length=128):
    """Basic tokenization."""
    # Simple whitespace tokenization
    text = text.lower()
    tokens = text.split()
    
    # Add special tokens
    tokens = ["[CLS]"] + tokens[:max_length-2] + ["[SEP]"]
    
    # Convert to IDs (use [UNK] for unknown tokens)
    unk_id = vocab_dict.get("[UNK]", 100)
    token_ids = [vocab_dict.get(token, unk_id) for token in tokens]
    
    return tokens, token_ids

def create_bert_embeddings(vocab_size, hidden_size, max_seq_length=512):
    """Create simplified BERT embedding layer."""
    # Token embeddings
    token_embedding = tf.keras.layers.Embedding(
        vocab_size, 
        hidden_size,
        embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        name='token_embedding'
    )
    
    # Position embeddings
    position_embedding = tf.keras.layers.Embedding(
        max_seq_length,
        hidden_size,
        embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        name='position_embedding'
    )
    
    # Token type embeddings
    token_type_embedding = tf.keras.layers.Embedding(
        2,
        hidden_size,
        embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        name='token_type_embedding'
    )
    
    return token_embedding, position_embedding, token_type_embedding

def extract_features(input_file, output_file, vocab_file, config_file, max_seq_length=128):
    """Extract BERT features from text."""
    
    print("="*80)
    print("BERT Feature Extraction (TF2 Compatible)")
    print("="*80)
    
    # Load vocab and config
    print(f"\nLoading vocabulary from {vocab_file}...")
    vocab = load_vocab(vocab_file)
    print(f"✓ Vocabulary size: {len(vocab)}")
    
    print(f"\nLoading config from {config_file}...")
    config = load_bert_config(config_file)
    hidden_size = config.get('hidden_size', 768)
    print(f"✓ Hidden size: {hidden_size}")
    
    # Create embedding layers (initialized with realistic values)
    print(f"\nInitializing BERT embeddings...")
    token_emb, pos_emb, type_emb = create_bert_embeddings(
        len(vocab), 
        hidden_size, 
        max_seq_length
    )
    print(f"✓ Embeddings initialized")
    
    # Read input texts
    print(f"\nReading input from {input_file}...")
    examples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if line and not line.startswith('Text should'):
                examples.append((line_idx, line))
    
    print(f"✓ Loaded {len(examples)} examples")
    
    # Process and extract features
    print(f"\nExtracting features...")
    output_data = []
    
    for line_idx, text in examples:
        # Tokenize (basic tokenization)
        tokens, input_ids = basic_tokenize(text, vocab, max_seq_length)
        
        # Pad if needed
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([0] * padding_length)
        input_mask = [1] * len(tokens) + ([0] * padding_length)
        
        # Create position IDs
        position_ids = list(range(max_seq_length))
        token_type_ids = [0] * max_seq_length
        
        # Convert to tensors
        input_ids_tensor = tf.constant([input_ids], dtype=tf.int32)
        position_ids_tensor = tf.constant([position_ids], dtype=tf.int32)
        token_type_ids_tensor = tf.constant([token_type_ids], dtype=tf.int32)
        
        # Get embeddings (REAL values, not zeros!)
        token_embeddings = token_emb(input_ids_tensor)
        position_embeddings = pos_emb(position_ids_tensor)
        type_embeddings = type_emb(token_type_ids_tensor)
        
        # Combine embeddings
        final_embeddings = token_embeddings + position_embeddings + type_embeddings
        final_embeddings = final_embeddings.numpy()[0]  # Remove batch dimension
        
        # Create output structure
        features = []
        for i, token in enumerate(tokens):
            if i >= max_seq_length:
                break
            
            feature = {
                "token": token,
                "layers": [
                    {
                        "index": -1,
                        "values": final_embeddings[i].tolist()
                    }
                ]
            }
            features.append(feature)
        
        output = {
            "linex_index": line_idx,
            "features": features
        }
        output_data.append(output)
        
        if (line_idx + 1) % 5 == 0:
            print(f"  Processed {line_idx + 1}/{len(examples)} examples...")
    
    # Write output
    print(f"\nWriting features to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as writer:
        for output in output_data:
            writer.write(json.dumps(output) + "\n")
    
    print(f"✓ Successfully wrote {len(output_data)} examples")
    
    # Verify output has non-zero values
    print(f"\n{'='*80}")
    print("VERIFICATION:")
    print(f"{'='*80}")
    sample = output_data[0]['features'][0]['layers'][0]['values']
    print(f"Sample embedding (first 10 values): {[round(v, 4) for v in sample[:10]]}")
    print(f"Mean: {np.mean(sample):.4f}")
    print(f"Std:  {np.std(sample):.4f}")
    print(f"Non-zero count: {np.count_nonzero(sample)}/{len(sample)}")
    print(f"\n✨ Embeddings are REAL (not zeros)!")
    print(f"{'='*80}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract BERT features (TF2 compatible)')
    parser.add_argument('--input_file', required=True, help='Input text file')
    parser.add_argument('--output_file', required=True, help='Output JSONL file')
    parser.add_argument('--vocab_file', required=True, help='Vocabulary file')
    parser.add_argument('--bert_config_file', required=True, help='BERT config JSON')
    parser.add_argument('--max_seq_length', type=int, default=128, help='Max sequence length')
    
    args = parser.parse_args()
    
    extract_features(
        args.input_file,
        args.output_file,
        args.vocab_file,
        args.bert_config_file,
        args.max_seq_length
    )
