#!/bin/bash

DATA_DIR=$(pwd)/binary_cifar10_data/

# Evaluating (only evolving the setup) a hand designed Neural Network on
# projected binary tasks. Utility script to check whether the tasks are
# ready.
bazel run -c opt \
  --script_path run_stacking_nobuild.sh \
  --action_env=CC=/usr/bin/gcc \
  //:stack -- \
  --experiment_name="baseline" \
  --final_tasks="
    tasks { \
      projected_binary_classification_task { \
        dataset_name: 'cifar10' \
        path: '${DATA_DIR}' \
        held_out_pairs {positive_class: 0 negative_class: 1} \
        held_out_pairs {positive_class: 0 negative_class: 2} \
        held_out_pairs {positive_class: 0 negative_class: 3} \
        held_out_pairs {positive_class: 0 negative_class: 4} \
        held_out_pairs {positive_class: 0 negative_class: 6} \
        held_out_pairs {positive_class: 0 negative_class: 7} \
        held_out_pairs {positive_class: 0 negative_class: 8} \
        held_out_pairs {positive_class: 1 negative_class: 2} \
        held_out_pairs {positive_class: 1 negative_class: 3} \
        held_out_pairs {positive_class: 1 negative_class: 4} \
        held_out_pairs {positive_class: 1 negative_class: 5} \
        held_out_pairs {positive_class: 1 negative_class: 6} \
        held_out_pairs {positive_class: 1 negative_class: 7} \
        held_out_pairs {positive_class: 1 negative_class: 9} \
        held_out_pairs {positive_class: 2 negative_class: 3} \
        held_out_pairs {positive_class: 2 negative_class: 4} \
        held_out_pairs {positive_class: 2 negative_class: 5} \
        held_out_pairs {positive_class: 2 negative_class: 6} \
        held_out_pairs {positive_class: 2 negative_class: 7} \
        held_out_pairs {positive_class: 2 negative_class: 8} \
        held_out_pairs {positive_class: 3 negative_class: 4} \
        held_out_pairs {positive_class: 3 negative_class: 7} \
        held_out_pairs {positive_class: 3 negative_class: 9} \
        held_out_pairs {positive_class: 4 negative_class: 5} \
        held_out_pairs {positive_class: 4 negative_class: 7} \
        held_out_pairs {positive_class: 4 negative_class: 8} \
        held_out_pairs {positive_class: 4 negative_class: 9} \
        held_out_pairs {positive_class: 5 negative_class: 6} \
        held_out_pairs {positive_class: 5 negative_class: 7} \
        held_out_pairs {positive_class: 5 negative_class: 8} \
        held_out_pairs {positive_class: 5 negative_class: 9} \
        held_out_pairs {positive_class: 6 negative_class: 7} \
        held_out_pairs {positive_class: 6 negative_class: 8} \
        held_out_pairs {positive_class: 6 negative_class: 9} \
        held_out_pairs {positive_class: 7 negative_class: 8} \
        held_out_pairs {positive_class: 7 negative_class: 9} \
        max_supported_data_seed: 100 \
      } \
      features_size: 16 \
      num_train_examples: 8000 \
      num_valid_examples: 1000 \
      num_train_epochs: 1 \
      num_tasks: 100 \
      eval_type: ACCURACY \
    } \
    " \
  --random_seed=0 \
  && ./run_stacking_nobuild.sh

    #fec {num_train_examples: 10 num_valid_examples: 10} \
