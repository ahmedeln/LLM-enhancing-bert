import json
import numpy as np
import tensorflow as tf
from bert import modeling


def count_params_for_config(config_dict, seq_length=64, batch_size=1, scope_name='bert_test'):
    # Build in a fresh Graph so variables don't collide
    g = tf.Graph()
    with g.as_default():
        config = modeling.BertConfig.from_dict(config_dict)
        input_ids = tf.placeholder(tf.int32, shape=[batch_size, seq_length], name='input_ids')
        input_mask = tf.placeholder(tf.int32, shape=[batch_size, seq_length], name='input_mask')
        token_type_ids = tf.placeholder(tf.int32, shape=[batch_size, seq_length], name='token_type_ids')
        model = modeling.BertModel(config=config, is_training=False,
                                  input_ids=input_ids, input_mask=input_mask,
                                  token_type_ids=token_type_ids, scope=scope_name)
        tvars = tf.trainable_variables()
        total = 0
        details = []
        for v in tvars:
            shape = v.get_shape().as_list()
            if None in shape:
                # try to get shape from variable
                try:
                    shape = v.shape.as_list()
                except Exception:
                    shape = [s if s is not None else -1 for s in shape]
            size = 1
            for d in shape:
                size *= (d if d is not None else 1)
            details.append((v.name, shape, int(size)))
            total += int(size)
    return total, details


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: parameter_counts.py <bert_config.json>')
        sys.exit(1)
    cfg_path = sys.argv[1]
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    # Prepare baseline config (original BERT): no factorized embeddings, no share, full attention, no swiglu
    baseline_cfg = dict(cfg)
    baseline_cfg['embedding_size'] = baseline_cfg.get('hidden_size')
    baseline_cfg['share_parameters'] = False
    baseline_cfg['window_size'] = 0
    baseline_cfg['use_swiglu'] = False

    print('Counting parameters for BASELINE (original BERT) configuration...')
    base_total, base_details = count_params_for_config(baseline_cfg, scope_name='bert_baseline')
    print('Baseline total parameters (approx):', base_total)

    print('\nCounting parameters for ENHANCED configuration (from file)...')
    enhanced_cfg = dict(cfg)
    # Ensure embedding_size in config defaults to hidden_size if not present
    if 'embedding_size' not in enhanced_cfg or enhanced_cfg['embedding_size'] is None:
        enhanced_cfg['embedding_size'] = enhanced_cfg.get('hidden_size')
    enh_total, enh_details = count_params_for_config(enhanced_cfg, scope_name='bert_enhanced')
    print('Enhanced total parameters (approx):', enh_total)

    diff = base_total - enh_total
    print('\nParameter difference: baseline - enhanced =', diff)

    # Print a few large variables for each
    def print_top(details, topk=8):
        details_sorted = sorted(details, key=lambda x: x[2], reverse=True)
        for name, shape, size in details_sorted[:topk]:
            print(f"{name}: shape={shape}, params={size}")

    print('\nTop variables baseline:')
    print_top(base_details)
    print('\nTop variables enhanced:')
    print_top(enh_details)
