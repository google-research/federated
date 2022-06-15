# Copyright 2022, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable

from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from personalization_benchmark.cross_device import constants
from personalization_benchmark.cross_device.datasets import ted_multi

_NUM_LOCAL_EPOCHS = 1
_TRAIN_BATCH_SIZE = 2


class TedMultiTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    model_fn, datasets, train_preprocess_fn, split_data_fn, accuracy_name = (
        ted_multi.create_model_and_data(
            num_local_epochs=_NUM_LOCAL_EPOCHS,
            train_batch_size=_TRAIN_BATCH_SIZE,
            use_synthetic_data=True))
    self._model_fn = model_fn
    self._datasets = datasets
    self._train_preprocess_fn = train_preprocess_fn
    self._split_data_fn = split_data_fn
    self._accuracy_name = accuracy_name

  def test_load_datasets(self):
    datasets = self._datasets
    self.assertLen(datasets, 3)
    train_clients = datasets[constants.TRAIN_CLIENTS_KEY]
    val_clients = datasets[constants.VALID_CLIENTS_KEY]
    test_clients = datasets[constants.TEST_CLIENTS_KEY]
    ted_multi._print_clients_number(train_clients, 'train')
    ted_multi._print_clients_number(val_clients, 'validation')
    ted_multi._print_clients_number(test_clients, 'test')

  def _test_dataset_train_preprocess(self, dataset: tf.data.Dataset,
                                     preprocess_fn: Callable[[tf.data.Dataset],
                                                             tf.data.Dataset]):
    total_samples = dataset.reduce(0, lambda x, y: x + 1).numpy()
    logging.info('samples in example dataset: %d', total_samples)
    new_ds = preprocess_fn(dataset)
    total_batches = new_ds.reduce(0, lambda x, y: x + 1).numpy()
    logging.info('batches after preprocessed: %d', total_batches)
    self.assertEqual(
        tf.math.ceil(total_samples / float(_TRAIN_BATCH_SIZE)) *
        _NUM_LOCAL_EPOCHS, total_batches)
    example_batch = next(iter(new_ds))
    self.assertLen(example_batch, 2)
    self.assertEqual(
        tf.shape(example_batch[0])[0],
        tf.shape(example_batch[1])[0])
    logging.info('example batch %s', example_batch)

  def test_tokenize_text(self):
    train_clients = self._datasets[constants.TRAIN_CLIENTS_KEY]
    en_cids = ted_multi._filter_client_ids_by_language(train_clients.client_ids,
                                                       'en')
    es_cids = ted_multi._filter_client_ids_by_language(train_clients.client_ids,
                                                       'es')
    en_ds = train_clients.create_tf_dataset_for_client(en_cids[0])
    tokenized_en_ds = en_ds.map(ted_multi._text_tokenize)
    logging.info('tokenized example EN client dataset:')
    for sample in tokenized_en_ds:
      logging.info('%s', sample)
    es_ds = train_clients.create_tf_dataset_for_client(es_cids[0])
    tokenized_es_ds = es_ds.map(ted_multi._text_tokenize)
    logging.info('tokenized example ES client dataset:')
    for sample in tokenized_es_ds:
      logging.info('%s', sample)

  def test_train_preprocess(self):
    train_clients = self._datasets[constants.TRAIN_CLIENTS_KEY]
    en_cids = ted_multi._filter_client_ids_by_language(train_clients.client_ids,
                                                       'en')
    es_cids = ted_multi._filter_client_ids_by_language(train_clients.client_ids,
                                                       'es')
    en_ds = train_clients.create_tf_dataset_for_client(en_cids[0])
    logging.info('example EN client dataset:')
    self._test_dataset_train_preprocess(en_ds, self._train_preprocess_fn)
    es_ds = train_clients.create_tf_dataset_for_client(es_cids[0])
    logging.info('example ES client dataset:')
    self._test_dataset_train_preprocess(es_ds, self._train_preprocess_fn)

  def test_model_foward(self):
    model = self._model_fn()
    self.assertIsInstance(model, tff.learning.Model)
    train_clients = self._datasets[constants.TRAIN_CLIENTS_KEY]
    example_ds = train_clients.create_tf_dataset_for_client(
        train_clients.client_ids[0])
    new_ds = self._train_preprocess_fn(example_ds)
    self.assertAllEqual(new_ds.element_spec, model.input_spec)
    example_batch = next(iter(new_ds))
    outputs = model.forward_pass(example_batch)
    self.assertAllEqual(
        tf.shape(outputs.predictions)[:2],
        [_TRAIN_BATCH_SIZE, ted_multi._MAX_SEQ_LEN])
    self.assertEqual(outputs.num_examples, _TRAIN_BATCH_SIZE)
    self.assertGreater(outputs.loss, 0)

  def test_fedavg(self):
    max_clients_per_round = 2
    it_process = tff.learning.build_federated_averaging_process(
        self._model_fn, client_optimizer_fn=tf.keras.optimizers.SGD)
    train_clients = self._datasets[constants.TRAIN_CLIENTS_KEY]
    new_train_clients = train_clients.preprocess(self._train_preprocess_fn)
    sampled_clients = [
        new_train_clients.create_tf_dataset_for_client(cid)
        for cid in new_train_clients.client_ids[:max_clients_per_round]
    ]
    state = it_process.initialize()
    _, metrics = it_process.next(state, sampled_clients)
    self.assertGreaterEqual(metrics['train'][self._accuracy_name], 0)

  def test_split_data_fn(self):
    valid_clients = self._datasets[constants.VALID_CLIENTS_KEY]
    example_ds = valid_clients.create_tf_dataset_for_client(
        valid_clients.client_ids[0])
    datasets = self._split_data_fn(example_ds)
    p13n_data = datasets[constants.PERSONALIZATION_DATA_KEY]
    example_sample = next(iter(p13n_data))
    self.assertEqual(tf.shape(example_sample[0]), [ted_multi._MAX_SEQ_LEN])
    self.assertEqual(tf.shape(example_sample[1]), [ted_multi._MAX_SEQ_LEN])
    test_data = datasets[constants.TEST_DATA_KEY]
    example_sample = next(iter(test_data))
    example_sample = next(iter(p13n_data))
    self.assertEqual(tf.shape(example_sample[0]), [ted_multi._MAX_SEQ_LEN])
    self.assertEqual(tf.shape(example_sample[1]), [ted_multi._MAX_SEQ_LEN])


if __name__ == '__main__':
  tf.test.main()
