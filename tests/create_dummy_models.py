import os


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
from pathlib import Path

from transformers.file_utils import is_tf_available, is_torch_available


if not is_torch_available():
    raise ValueError("Please install PyTorch.")

if not is_tf_available():
    raise ValueError("Please install TensorFlow.")

import copy
import importlib
import os
import tempfile
from collections import OrderedDict

import h5py
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from transformers import (
    CONFIG_MAPPING,
    FEATURE_EXTRACTOR_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
    MODEL_FOR_OBJECT_DETECTION_MAPPING,
    MODEL_FOR_PRETRAINING_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    MODEL_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    TF_MODEL_FOR_CAUSAL_LM_MAPPING,
    TF_MODEL_FOR_MASKED_LM_MAPPING,
    TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
    TF_MODEL_FOR_PRETRAINING_MAPPING,
    TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    TF_MODEL_MAPPING,
    TF_MODEL_WITH_LM_HEAD_MAPPING,
    TOKENIZER_MAPPING,
    AutoFeatureExtractor,
    AutoTokenizer,
    logging,
)
from transformers.models.auto.configuration_auto import AutoConfig, model_type_to_module_name


INVALID_ARCH = []
logging.set_verbosity_error()

tokenizer_checkpoint_overrides = {"byt5": "google/byt5-small"}
ds = load_dataset("wikitext", "wikitext-2-raw-v1")
training_ds = ds["train"]
testing_ds = ds["test"]

per_model_type_configuration_attributes = {
    "big_bird": {"num_labels": 1},
}

unexportable_model_architectures = [
    "RoFormerForMultipleChoice",
    "TFRoFormerForMultipleChoice",
    "TFMobileBertForMultipleChoice",
    "MobileBertForMultipleChoice",
    "TFDistilBertForMultipleChoice",
    "DistilBertForMultipleChoice",
    "TFAlbertForMultipleChoice",
    "AlbertForMultipleChoice",
    "TFMPNetForMultipleChoice",
    "MPNetForMultipleChoice",
    "TFLongformerForMultipleChoice",
    "LongformerForMultipleChoice",
    "TFRobertaForMultipleChoice",
    "RobertaForMultipleChoice",
    "SqueezeBertForMultipleChoice",
    "TFSqueezeBertForMultipleChoice",
    "BertForMultipleChoice",
    "TFBertForMultipleChoice",
    "XLNetForMultipleChoice",
    "TFXLNetForMultipleChoice",
    "ElectraForMultipleChoice",
    "TFElectraForMultipleChoice",
    "FunnelForMultipleChoice",
    "TFFunnelForMultipleChoice",
]


def get_checkpoint_from_configuration_class(config):
    checkpoint = None
    model_type = config.model_type
    module = model_type_to_module_name(model_type)
    to_import = f"transformers.models.{module}.modeling_{module}"

    # First try to retrieve the _CHECKPOINT_FOR_DOC from the modeling file
    try:
        module = importlib.import_module(to_import)
        checkpoint = module._CHECKPOINT_FOR_DOC
    except (ModuleNotFoundError, AttributeError):
        # If no _CHECKPOINT_FOR_DOC or no modeling file, retrieve the first checkpoint defined in the tokenization file
        to_import = f"transformers.models.{module}.tokenization_{module}"
        try:
            module = importlib.import_module(to_import)
            checkpoint = list(list(module.PRETRAINED_VOCAB_FILES_MAP.values())[0].keys())[0]
        except (AttributeError, ModuleNotFoundError):
            pass

    return checkpoint


def get_tiny_config_from_class(configuration_class):
    """
    Retrieve a tiny configuration from the configuration class. It uses each class' `ModelTester`.
    Args:
        configuration_class: Subclass of `PreTrainedConfig`.

    Returns:
        an instance of the configuration passed, with very small hyper-parameters

    """
    model_type = configuration_class.model_type
    camel_case_model_name = configuration_class.__name__.split("Config")[0]

    try:
        print("Importing", model_type_to_module_name(model_type))
        module = importlib.import_module(f".test_modeling_{model_type_to_module_name(model_type)}", package="tests")
        model_tester_class = getattr(module, f"{camel_case_model_name}ModelTester", None)
    except ModuleNotFoundError:
        print(f"Will not build {model_type}: no model tester or cannot find the testing module from the model name.")
        return

    if model_tester_class is None:
        return

    model_tester = model_tester_class(parent=None)

    if hasattr(model_tester, "get_pipeline_config"):
        return model_tester.get_pipeline_config()
    elif hasattr(model_tester, "get_config"):
        return model_tester.get_config()


def build_pytorch_weights_from_multiple_architectures(pytorch_architectures, weights_path, config_overrides=None):
    # Create the PyTorch tiny models
    for config, architectures in pytorch_architectures.items():
        base_tiny_config = get_tiny_config_from_class(config)

        if base_tiny_config is None:
            continue

        if config_overrides is not None:
            for k, v in config_overrides.items():
                setattr(base_tiny_config, k, v)

        base_tiny_config.num_labels = 2

        if config.model_type in per_model_type_configuration_attributes:
            for key, value in per_model_type_configuration_attributes[config.model_type].items():
                setattr(base_tiny_config, key, value)

        print(base_tiny_config)
        os.makedirs(f"{weights_path}/{config.model_type}", exist_ok=True)
        base_tiny_config.save_pretrained(f"{weights_path}/{config.model_type}")

        state_dict = {}
        flat_architectures = []

        per_model_configuration_attributes = {
            "ReformerModelWithLMHead": {"is_decoder": True},
            "ReformerModelForMaskedLM": {"is_decoder": False},
        }

        for architecture_tuple in architectures:
            if not isinstance(architecture_tuple, tuple):
                architecture_tuple = (architecture_tuple,)

            for architecture in architecture_tuple:
                tiny_config = copy.deepcopy(base_tiny_config)
                if architecture.__name__ in per_model_configuration_attributes:
                    for key, value in per_model_configuration_attributes[architecture.__name__].items():
                        setattr(tiny_config, key, value)
                flat_architectures.append(architecture)

                if "DPRQuestionEncoder" in architecture.__name__:
                    continue

                print(architecture)
                try:
                    model = architecture.from_pretrained(
                        None, config=tiny_config, state_dict=state_dict, no_check_corrupted=True
                    )
                    state_dict = {
                        **state_dict,
                        **model.state_dict()
                    }

                    # for key in state_dict.keys():
                    #     if key.startswith(f"{config.model_type}."):珞
                    #         del state_dict[key]
                except:
                    INVALID_ARCH.append(architecture.__name__)

        keys = list(state_dict.keys())

        for key in keys:
            print(key, key.startswith(f"{config.model_type}."))
            if not key.startswith(f"{config.model_type}.") and f"{config.model_type}.{key}" in keys:
                print("Removing", key, f"{config.model_type}.{key}", "present:", f"{config.model_type}.{key}" in keys)
                del state_dict[key]

        torch.save(
            OrderedDict(state_dict),
            f"{weights_path}/{config.model_type}/pytorch_model.bin",
        )


def build_tensorflow_weights_from_multiple_architectures(
    tensorflow_architectures, weights_path, config_overrides=None
):
    # Create the TensorFlow tiny models
    for config, architectures in tensorflow_architectures.items():
        if len(architectures) == 0:
            continue

        base_tiny_config = get_tiny_config_from_class(config)

        if base_tiny_config is None:
            continue

        if config_overrides is not None:
            for k, v in config_overrides.items():
                setattr(base_tiny_config, k, v)

        base_tiny_config.num_labels = 2

        if config.model_type in per_model_type_configuration_attributes:
            for key, value in per_model_type_configuration_attributes[config.model_type].items():
                setattr(base_tiny_config, key, value)

        os.makedirs(f"{weights_path}/{config.model_type}", exist_ok=True)
        print(base_tiny_config)
        base_tiny_config.save_pretrained(f"{weights_path}/{config.model_type}")

        flat_architectures = []

        per_model_configuration_attributes = {
            "ReformerModelWithLMHead": {"is_decoder": True},
            "ReformerModelForMaskedLM": {"is_decoder": False},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            for architecture_tuple in architectures:
                if not isinstance(architecture_tuple, tuple):
                    architecture_tuple = (architecture_tuple,)

                for architecture in architecture_tuple:
                    tiny_config = copy.deepcopy(base_tiny_config)
                    if architecture.__name__ in per_model_configuration_attributes:
                        for key, value in per_model_configuration_attributes[architecture.__name__]:
                            setattr(tiny_config, key, value)

                    if "DPRQuestionEncoder" in architecture.__name__:
                        continue

                    flat_architectures.append(architecture)

                    try:
                        model = architecture.from_pretrained(
                            f"{weights_path}/{config.model_type}",
                            config=tiny_config,
                            from_pt=True,
                        )
                    except ValueError as e:
                        INVALID_ARCH.append(architecture.__name__)
                    model.save_pretrained(f"{temp_dir}/{config.model_type}/{architecture.__name__}")

            h5_files = [
                h5py.File(
                    f"{temp_dir}/{config.model_type}/{architecture.__name__}/tf_model.h5",
                    "r",
                )
                for architecture in flat_architectures
            ]
            new_h5_file = h5py.File(f"{weights_path}/{config.model_type}/tf_model.h5", "w")

            # function to return a list of paths to each dataset
            def get_datasets(key, archive):
                if key[-1] != "/":
                    key += "/"
                out = [key]

                for name in archive[key]:
                    path = key + name

                    if isinstance(archive[path], h5py.Dataset):
                        out += [path]
                    else:
                        out += get_datasets(path, archive)

                return out

            # open HDF5-files
            for data in h5_files:
                # read as much datasets as possible from the old HDF5-file
                datasets = get_datasets("/", data)

                # get the group-names from the lists of datasets
                groups = list(set([i[::-1].split("/", 1)[1][::-1] for i in datasets]))
                groups = [i for i in groups if len(i) > 0]

                # sort groups based on depth
                idx = np.argsort(np.array([len(i.split("/")) for i in groups]))
                groups = [groups[i] for i in idx]

                # create all groups that contain dataset that will be copied
                for group in groups:
                    try:
                        new_h5_file.create_group(group)
                    except ValueError:
                        pass

                # copy datasets
                for path in datasets:
                    group = path[::-1].split("/", 1)[1][::-1]

                    if len(group) == 0:
                        group = "/"

                    # Copy the data from one archive to the other
                    try:
                        data.copy(path, new_h5_file[group])
                    except (ValueError, RuntimeError):
                        pass

                    # Manually set attributes, such as `keras_version`, `weight_names`
                    for key, value in data[path].attrs.items():
                        if key in new_h5_file[group].attrs:
                            existing_value = new_h5_file[group].attrs[key]

                            # The value already exists. If it is a list, then we append to it the current value.
                            # We convert the list to a set and back to remove duplicates
                            if isinstance(existing_value, np.ndarray):
                                updated_value = list(set(existing_value.tolist() + value.tolist()))
                                new_h5_file[group].attrs.create(key, updated_value)
                            else:
                                new_h5_file[group].attrs.modify(key, value)
                        else:
                            new_h5_file[group].attrs.create(key, value)

            new_h5_file.close()


def build_processor_files(tokenizer_mapping, output_folder):
    report = {"no_feature_extractor": [], "no_tokenizer": [], "identical_tokenizer": [], "vocab_size": {}}
    for config, tokenizers in tokenizer_mapping.items():
        model_type = config.model_type
        checkpoint = get_checkpoint_from_configuration_class(config)

        # Try to build a processor
        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(checkpoint)
            feature_extractor.save_pretrained(os.path.join(output_folder, model_type))
        except (OSError, KeyError):
            report["no_feature_extractor"].append(model_type)
        except RecursionError:
            report["no_feature_extractor"].append(model_type)
            print(f"Recursion error on {model_type}")
            raise
        except TypeError:
            print(f"TypeError on {model_type}, with {tokenizers}")
            raise

        try:
            tokenizer_fast = AutoTokenizer.from_pretrained(checkpoint)

            try:
                new_tokenizer = tokenizer_fast.train_new_from_iterator(training_ds["text"], 1000)
                tokenizer_fast(testing_ds["text"])
                # print(f"Saving! Converted from {len(tokenizer_fast)} vocab size to {len(new_tokenizer)}.")
                print("SAVING", len(new_tokenizer))
                new_tokenizer.save_pretrained(os.path.join(output_folder, model_type))
                new_tokenizer.save_pretrained(os.path.join(output_folder, model_type), legacy_format=True)

                report["vocab_size"] = len(new_tokenizer)
            except Exception as e:
                tokenizer_fast.save_pretrained(os.path.join(output_folder, model_type))
                tokenizer_fast.save_pretrained(os.path.join(output_folder, model_type), legacy_format=True)
                report["identical_tokenizer"].append(model_type)
                report["vocab_size"] = len(tokenizer_fast)

        except (OSError, KeyError):
            report["no_tokenizer"].append(model_type)

    return report


def check_architecture_validity(pytorch_architectures, tensorflow_architectures, weights_path, vocab_sizes):
    for config, architectures in tqdm(pytorch_architectures.items(), desc="Checking PyTorch weights validity"):
        for architecture_tuple in architectures:
            if not isinstance(architecture_tuple, tuple):
                architecture_tuple = (architecture_tuple,)

            for architecture in architecture_tuple:
                model, loading_info = architecture.from_pretrained(
                    f"{weights_path}/{config.model_type}", output_loading_info=True, no_check_corrupted=True
                )
                if loading_info["missing_keys"] == 0:
                    raise ValueError(
                        f"Missing weights when loading PyTorch checkpoints: {loading_info['missing_keys']}"
                    )

    print("Checked PyTorch weights")

    for config, architectures in tqdm(tensorflow_architectures.items(), desc="Checking TensorFlow weights validity"):
        for architecture_tuple in architectures:
            if not isinstance(architecture_tuple, tuple):
                architecture_tuple = (architecture_tuple,)

            for architecture in architecture_tuple:
                model, loading_info = architecture.from_pretrained(
                    f"{weights_path}/{config.model_type}",
                    output_loading_info=True,
                )

                if len(loading_info["missing_keys"]) != 0:
                    required_weights_missing = []
                    for missing_key in loading_info["missing_keys"]:
                        if "dropout" not in missing_key:
                            required_weights_missing.append(missing_key)

                    if len(required_weights_missing) > 0:
                        raise ValueError(f"Found missing weights in {architecture}: {required_weights_missing}")

    print("Checked TensorFlow weights")


# Define the PyTorch and TensorFlow mappings
pytorch_mappings = [
    MODEL_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_PRETRAINING_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
    MODEL_FOR_OBJECT_DETECTION_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
]
tensorflow_mappings = [
    TF_MODEL_MAPPING,
    TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    TF_MODEL_FOR_MASKED_LM_MAPPING,
    TF_MODEL_FOR_PRETRAINING_MAPPING,
    TF_MODEL_FOR_CAUSAL_LM_MAPPING,
    TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
    TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    TF_MODEL_WITH_LM_HEAD_MAPPING,
    TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
]

# CONFIG_MAPPING = {k: v for i, (k, v) in enumerate(CONFIG_MAPPING.items()) if i < 15}

# Reorder the mappings, so that a single configuration maps to an array of all possible architectures attributed
# to that configuration.
# Ex: {BertConfig: [BertModel, BertForMaskedLM, ..., BertForQuestionAnswering]}
def get_architectures_from_configuration_list(mappings, configuration_list):
    returned_mapping = {}
    for config in configuration_list:
        returned_mapping[config] = []
        for mapping in mappings:
            if config in mapping:
                models = mapping[config] if isinstance(mapping[config], tuple) else (mapping[config],)
                for model in models:
                    if model.__name__ not in unexportable_model_architectures:
                        returned_mapping[config].append(model)
    return returned_mapping


def get_processor_mapping_from_configuration_list(configuration_list):
    def retrieve_tokenizer(config):
        checkpoint = get_checkpoint_from_configuration_class(config)

        try:
            configuration = AutoConfig.from_pretrained(checkpoint)
        except OSError:
            print(f"Couldn't load {config}")
            return None

        return configuration.tokenizer_class

    processor_mapping = {}
    for configuration in configuration_list:
        if configuration in TOKENIZER_MAPPING:
            processor_mapping[configuration] = TOKENIZER_MAPPING[configuration]
        elif configuration in FEATURE_EXTRACTOR_MAPPING:
            processor_mapping[configuration] = FEATURE_EXTRACTOR_MAPPING[configuration]
        else:
            # Some configurations have no tokenizer; for example GPT-J is bound to the GPT-2 processor.
            processor_mapping[configuration] = retrieve_tokenizer(configuration)

    return processor_mapping


if __name__ == "__main__":

    def list_str(values):
        return values.split(",")

    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Will create all tiny models.")
    parser.add_argument(
        "--no_check",
        action="store_true",
        help="If set, will not check the validity of architectures. Use with caution.",
    )
    parser.add_argument(
        "-m",
        "--model_types",
        type=list_str,
        help="Comma-separated list of model type(s) from which the tiny models will be created.",
    )
    parser.add_argument("--black_list", type=list_str, help="Comma-separated list of model type(s) to ignore.",default='convbert,blenderbot-small,rag,dpr,retribert,layoutlmv2')
    parser.add_argument("output_path", type=Path, help="Path indicating where to store generated ONNX model.")
    args = parser.parse_args()

    if not args.all and not args.model_types:
        raise ValueError("Please provide at least one model type or pass `--all` to export all architectures.")

    if args.all:
        pytorch_architectures = get_architectures_from_configuration_list(pytorch_mappings, CONFIG_MAPPING.values())
        tensorflow_architectures = get_architectures_from_configuration_list(
            tensorflow_mappings, CONFIG_MAPPING.values()
        )
        processors = get_processor_mapping_from_configuration_list(CONFIG_MAPPING.values())
        configurations = CONFIG_MAPPING.values()
    else:
        configurations = [CONFIG_MAPPING[model_type] for model_type in args.model_types]
        pytorch_architectures = get_architectures_from_configuration_list(pytorch_mappings, configurations)
        tensorflow_architectures = get_architectures_from_configuration_list(tensorflow_mappings, configurations)
        processors = get_processor_mapping_from_configuration_list(configurations)

    processors_config_without_mapping = [k for k, v in processors.items() if v is None]
    configurations = [c for c in configurations if c not in processors_config_without_mapping]

    if args.black_list:
        configurations = [c for c in configurations if c.model_type not in args.black_list]

    to_export = {
        configuration: {
            "tensorflow": tensorflow_architectures[configuration],
            "pytorch": pytorch_architectures[configuration],
            "processors": processors[configuration],
        }
        for configuration in configurations
    }

    report = {"no_feature_extractor": [], "no_tokenizer": [], "identical_tokenizer": [], "vocab_sizes": {}}

    for config, architectures in tqdm(to_export.items()):
        processor_report = build_processor_files({config: architectures["processors"]}, args.output_path)
        tokenizer_length = processor_report.pop("vocab_size", None)
        report["vocab_sizes"][config.model_type] = tokenizer_length
        [report[k].extend(v) for k, v in processor_report.items()]

        build_pytorch_weights_from_multiple_architectures(
            {config: architectures["pytorch"]}, args.output_path, config_overrides={"vocab_size": tokenizer_length}
        )
        build_tensorflow_weights_from_multiple_architectures(
            {config: architectures["tensorflow"]}, args.output_path, config_overrides={"vocab_size": tokenizer_length}
        )

    print("--- Report ---")
    print("INVALID", INVALID_ARCH)

    if len(processors_config_without_mapping):
        print(
            f"Some models could not be exported due to a lack of processor: {[c.model_type for c in processors_config_without_mapping]}"
        )

    if len(report["no_feature_extractor"]):
        print(
            "The following models have no feature extractor or it couldn't be built:", report["no_feature_extractor"]
        )

    if len(report["no_tokenizer"]):
        print("The following models have no tokenizer or it couldn't be built:", report["no_tokenizer"])

    if len(report["identical_tokenizer"]):
        print(
            "The following models leverage the checkpoint's tokenizer as a smaller copy couldn't be made:",
            report["identical_tokenizer"],
        )

    print("Vocabulary sizes:", report["vocab_sizes"])

    if not args.no_check:
        check_architecture_validity(
            pytorch_architectures, tensorflow_architectures, args.output_path, report["vocab_sizes"]
        )