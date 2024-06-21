#!/usr/bin/env python3
"""
A testbench for running neural network tests with Second Order Optimizers
Copyright 2021 Nanmiao Wu, Eric Silk

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from copy import deepcopy
import os
import random
import argparse
from pathlib import Path
from time import perf_counter
import json
import math
import hashlib
from types import SimpleNamespace
from typing import Optional, Union, Tuple
import warnings

# import pynvml
import torch
from torch import nn
from torch import optim
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import torch.multiprocessing as mp
import transformers
# from datasets import load_dataset
import pandas as pd
import peft

#QLoRA stuff
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model


from pytorch_soo import optim_hfcr_newton

from pytorch_soo import quasi_newton

from pytorch_soo import nonlinear_conjugate_gradient as nlcg
from pytorch_soo.line_search_spec import LineSearchSpec
from pytorch_soo.trust_region import BadTrustRegionSpec, TrustRegionSpec

from yahoo_answers_dataset import Yahoo_Answers_Dataset

import time

HASH_LENGTH = 10
print("WE WILL BE USING GPU") if torch.cuda.is_available() else print("CPU!!")


def args_to_fname(arg_dict_, ext):
    """
    Converts the argument dictionary into a filename with a hash
    """
    # Don't want the output directory influencing the hash
    # Old one did this as boolean, so let's add that back in
    outdir = deepcopy(arg_dict_["record"])
    arg_dict = deepcopy(arg_dict_)
    print("+" * 80)
    print("\tOutdir:", outdir)
    print("+" * 80)
    print("outdir:", outdir)
    del arg_dict["record"]
    arg_dict["record"] = outdir is not None
    sorted_args = sorted(arg_dict.items())
    sorted_arg_values = [i[1] for i in sorted_args]
    num_args = len(sorted_arg_values)

    hasher = hashlib.sha256()

    fname = "TEST" + ("_{}" * num_args)
    fname = fname.format(*sorted_arg_values)
    hasher.update(bytes(fname, "UTF-8"))
    digest = hasher.hexdigest()[:HASH_LENGTH]
    fname = f"TEST_{arg_dict['opt']}_{digest}.{ext}"
    if outdir is not None:
        print("outdir:", outdir)
        fname = os.path.join(outdir, fname)

    return fname


def get_args():
    """
    Get the arguments to the script for the dataset, optimizers, etc.
    """
    parser = argparse.ArgumentParser(description="PyTorch")
    # Batch Sizes
    parser.add_argument(
        "--batch_size_train",
        type=int,
        default=60000,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--batch_size_test",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )

    # # Dataset / implicitly model as well  Always Yahoo Answers
    # parser.add_argument(
    #     "--dataset", type=str, required=True, help="the dataset option: mnist / cifar10"
    # )

    # Optimizer
    parser.add_argument(
        "--opt",
        type=str,
        required=True,
        help="the optimizer option: sgd / lbfgs / kn / kn2/ sr1 /sr1d / dfp / dfpi",
    )

    parser.add_argument(
        "--memory",
        type=int,
        default=None,
        help="Size of quasi-newton memory, leave blank for unlimited",
    )

    parser.add_argument(
        "--momentum", type=float, default=0.9, help="momentum for SGD algorithm"
    )

    # Network (if not Resnet)
    parser.add_argument("--hidden", type=int, default=15, help="size of hidden layer")

    # SOO Options
    parser.add_argument(
        "--max_newton", type=int, default=10, help="max number of newton iterations"
    )
    parser.add_argument(
        "--abs_newton_tol",
        type=float,
        default=1.0e-5,
        help="Absolute tolerance for Newton iteration convergence",
    )
    parser.add_argument(
        "--rel_newton_tol",
        type=float,
        default=1.0e-8,
        help="Relative tolerance for Newton iteration convergence",
    )
    parser.add_argument(
        "--max_cr", type=int, default=10, help="max number of conjugate residual"
    )
    parser.add_argument(
        "--cr_tol",
        type=float,
        default=1.0e-3,
        help="tolerance for conjugate residual iteration" "convergence",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="training update weight (default: 0.01)",
    )

    # Line Search Params
    parser.add_argument(
        "--sufficient_decrease",
        type=float,
        default=None,
        help="The Armijo rule coefficient",
    )
    parser.add_argument(
        "--curvature_condition",
        type=float,
        default=None,
        help="The curvature coeff for the Wolfe conditions",
    )
    parser.add_argument(
        "--extrapolation_factor",
        type=float,
        default=None,
        help="The factor of decrease for the line search extrapolation",
    )
    parser.add_argument(
        "--max_searches",
        type=int,
        default=None,
        help="The maximum number of line searches that can be attempted before accepting failure",
    )

    # Trust Region Params
    parser.add_argument(
        "--initial_radius",
        type=float,
        default=None,
        help="the initial radius of the trust region model",
    )
    parser.add_argument(
        "--max_radius",
        type=float,
        default=None,
        help="The maximum radius for the trust region model",
    )
    parser.add_argument(
        "--nabla0",
        type=float,
        default=None,
        help=(
            "The minimum acceptable step size for the trust region model. Must be >=0, but should "
            "be a very small value. Values lower than this will outright reject the step."
        ),
    )
    parser.add_argument(
        "--nabla1",
        type=float,
        default=None,
        help=(
            'The minimum value for the trust region model to be "good enough" and not prompt a '
            "decrease in trust region radius"
        ),
    )
    parser.add_argument(
        "--nabla2",
        type=float,
        default=None,
        help=(
            'The minimum value for the trust region model to be "better than good" and prompt an '
            "increase in trust region radius"
        ),
    )
    parser.add_argument(
        "--shrink_factor",
        type=float,
        default=None,
        help=(
            "The multiplicative factor by which the trust region will be reduced if needed. Must "
            "be in (0.0, 1.0)"
        ),
    )
    parser.add_argument(
        "--growth_factor",
        type=float,
        default=None,
        help=(
            "The multiplicative factor by which the trust region will be grown if needed. Must "
            "be >1.0"
        ),
    )
    parser.add_argument(
        "--max_subproblem_iter",
        type=int,
        default=None,
        help=(
            "The maximum number of iterations the trust region subproblem algorithm may take to "
            "to solve before accepting an imperfect answer"
        ),
    )

    # Training options
    parser.add_argument(
        "--num_epoch", type=int, default=12, help="max number of epochs"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--read_nn", help="deserialize nn from directory before training"
    )
    parser.add_argument(
        "--write_nn",
        action="store_true",
        help="serialize nn to directory after training",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training" "status",
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        help="Enables storing results, specifies where",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device to train on: cpu / cuda (default: cuda)",
    )

    return parser.parse_args()

INPUT_LENGTH = 256
NUM_OUTPUTS = 8
# def load_data(batch_size_train:int, batch_size_test:int, device) -> Tuple:
#     """
#     Get the data based upon the relevant arguments
#     """
#     print("Loading up the data", flush = True)
#     ya = Yahoo_Answers_Dataset(dict_size = 10000, valid_ratio = 0.1, max_len = INPUT_LENGTH, test_path = "/rcfs/projects/sml2024/Yahoo-Answers-Topic-Classification-Dataset/dataset/yahoo_answers_test.pkl.gz",
#                            train_path= "/rcfs/projects/sml2024/Yahoo-Answers-Topic-Classification-Dataset/dataset/yahoo_answers_csv/train_sixty_thousand.pkl.gz", dict_path="/rcfs/projects/sml2024/Yahoo-Answers-Topic-Classification-Dataset/dataset/yahoo_answers_dict.pkl.gz")
    
#     # tokenizer = transformers.AutoTokenizer.from_pretrained("openai-community/gpt2", max_length=INPUT_LENGTH)
#     tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert/distilgpt2", max_length=INPUT_LENGTH)
#     # tokenizer.padding_side = "left"
#     # Define PAD Token = EOS Token
#     tokenizer.pad_token = tokenizer.eos_token
#     # tokenizer = transformers.GPT2Tokenizer() #What's the difference betgween this and the one above?


#     train_tokenizer_output = tokenizer(ya.train['seq'], return_tensors='pt', padding='max_length',
#                                        max_length=INPUT_LENGTH, truncation=True).to('cpu')

#     #New!
#     train_tokenized_ids = train_tokenizer_output['input_ids'].to('cpu')
#     train_masks = train_tokenizer_output['attention_mask'].to('cpu')

#     #Concatenate the tokenized ids and the masks horizontally to decompose later
#     train_ids_and_mask = torch.cat(tensors=(train_tokenized_ids, train_masks), dim = 1).to('cpu')

#     #one hot encode the targets
#     train_targets = [target - 1 for target in ya.train['label']] #Range from 0 to 9 inclusive
#     train_targets = torch.tensor(train_targets).to('cpu')
#     # print(train_targets, flush = True)
#     train_targets = torch.nn.functional.one_hot(train_targets).to('cpu')
#     # train_targets.to(device)
#     print("The training data has been loaded in", flush = True)


#     train = torch.utils.data.TensorDataset(train_ids_and_mask, train_targets)

#     train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size_train, shuffle=True)

#     ###############################
#     ###########Test Data###########
#     ###############################
#     test_tokenizer_output = tokenizer(ya.test['seq'], return_tensors='pt', padding='max_length',
#                                        max_length=INPUT_LENGTH, truncation=True).to('cpu')
#     # test_tokenized_inputs.to(device)
#     test_tokenized_ids = test_tokenizer_output['input_ids']
#     test_masks = test_tokenizer_output['attention_mask']

#     #Concatenate the tokenized ids and the masks horizontally to decompose later
#     test_ids_and_mask = torch.cat(tensors=(test_tokenized_ids, test_masks), dim = 1).to('cpu')

#     #One hot encode the targets
#     test_targets = [target - 1 for target in ya.test['label']] #Range from 0 to 9 inclusive
#     test_targets = torch.tensor(test_targets).to('cpu')
#     test_targets = torch.nn.functional.one_hot(test_targets).to('cpu')
#     # test_targets.to(device)

#     test = torch.utils.data.TensorDataset(test_ids_and_mask, test_targets)
#     print("The test data has been loaded in", flush = True)


#     test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size_test, shuffle=False)


#     return train_loader, test_loader
# world_size = torch.cuda.device_count()
# print(f"We are using {world_size} devices (GPUs)")
# port = random.randint(49152,65535)

def load_data(batch_size_train:int, batch_size_test:int, device) -> Tuple:
    """
    Get the data based upon the relevant arguments
    """

    train = pd.read_csv("/rcfs/projects/sml2024/train_clean_news_articles_categorization.csv")

    train_text = train.iloc[:,-1].to_list()

    # train_targets = torch.tensor(train.iloc[:,:-1].to_numpy()).to('cpu')

    train_targets = torch.argmax(torch.tensor(train.iloc[:,:-1].to_numpy()), dim=1).type(torch.int64).to('cpu')
    # global NUM_OUTPUTS
    # NUM_OUTPUTS = train_targets.shape[1]


    tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased", max_length=INPUT_LENGTH)
    
    # tokenizer.padding_side = "left"
    # Define PAD Token = EOS Token
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.eos_token = tokenizer.pad_token  #THIS WORKS WITH BERT
    tokenizer.padding_side = 'right'
    # tokenizer = transformers.GPT2Tokenizer() #What's the difference betgween this and the one above?


    train_tokenizer_output = tokenizer(train_text, return_tensors='pt', padding='max_length',
                                       max_length=INPUT_LENGTH, truncation=True).to('cpu')

    #New!
    train_tokenized_ids = train_tokenizer_output['input_ids']
    train_masks = train_tokenizer_output['attention_mask']

    #Concatenate the tokenized ids and the masks horizontally to decompose later
    train_ids_and_mask = torch.cat(tensors=(train_tokenized_ids, train_masks), dim = 1)

    
    # print("The training data has been loaded in", flush = True)


    train = torch.utils.data.TensorDataset(train_ids_and_mask, train_targets)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size_train, shuffle=True)

    ###############################
    ###########Test Data###########
    ###############################

    test_csv = pd.read_csv("/rcfs/projects/sml2024/test_clean_news_articles_categorization.csv")
    test_text = test_csv.iloc[:,-1].to_list()

    test_tokenizer_output = tokenizer(test_text, return_tensors='pt', padding='max_length',
                                       max_length=INPUT_LENGTH, truncation=True)
    # test_tokenized_inputs.to(device)
    test_tokenized_ids = test_tokenizer_output['input_ids']
    test_masks = test_tokenizer_output['attention_mask']

    #Concatenate the tokenized ids and the masks horizontally to decompose later
    test_ids_and_mask = torch.cat(tensors=(test_tokenized_ids, test_masks), dim = 1)

    #One hot encode the targets
    # test_targets = torch.tensor(test_csv.iloc[:,:-1].to_numpy())

    #The line below extracts the label of the target
    test_targets = torch.argmax(torch.tensor(test_csv.iloc[:,:-1].to_numpy()), dim=1).type(torch.int64).to('cpu')

    test = torch.utils.data.TensorDataset(test_ids_and_mask, test_targets)
    print("The test data has been loaded in")


    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size_test, shuffle=False)


    return train_loader, test_loader

# #FIXME: Put model here
# test = load_data(512,512,'cpu')
# model = transformers.AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path='gpt2', max_length=INPUT_LENGTH, num_labels=10)


#Done.
def get_model_and_loss(args, device, qlora = False, parallel = False):
    """
    Does what it says on the tin.
    """
    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    if qlora:

        model = transformers.AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path='distilbert/distilbert-base-uncased',
                                                                            max_length=INPUT_LENGTH,
                                                                            num_labels=NUM_OUTPUTS,
                                                                            quantization_config=bnb_config,
                                                                            device_map = device)    
        model.config.pad_token_id = model.config.eos_token_id
        model = prepare_model_for_kbit_training(model)
        config = LoraConfig(
            r=32, #Higher = more expressivity. Controls number of parameters used i.e. memory
            lora_alpha=32,
            target_modules = ["score"],# "c_attn", "c_proj", "c_fc"], #target_modules = ["c_attn", "c_proj", "c_fc"]
            bias = "none",
            lora_dropout=0.1, #FIXME: Check with SOO paper and see if this needs to be set to 0?
            task_type = "SEQ_CLS"
        )

        # lora_config = LoraConfig(
        #     r=16,
        #     lora_alpha=32,
        #     target_modules=target_modules,
        #     lora_dropout=0.05,
        #     bias="none",
        #     task_type="SEQ_CLS"
        #     )
        # model.gradient_checkpointing_enable()
        model = get_peft_model(model, config)
        # print(peft.print_number_of_trainable_model_parameters(model))
        # model.to('cpu')
        model.print_trainable_parameters()
    else: #no QLoRA
        model = transformers.AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path="distilbert/distilbert-base-uncased",
                                                                                max_length=INPUT_LENGTH,
                                                                                num_labels=NUM_OUTPUTS,
                                                                                device_map=device)
        model.config.pad_token_id = model.config.eos_token_id

        if True:
        #Freeze everything
            for name,param in model.named_parameters():
                param.requires_grad = False
                if name == 'classifier.weight' or name == 'pre_classifier.weight':
                    param.requires_grad = True
                if name == 'classifier.bias' or name == 'pre_classifier.bias':
                    param.requires_grad = True
            print("Using linear probe!")
        if parallel:
            model = nn.DataParallel(model)
        else:
            pass
        # ##########model = nn.DataParallel(model)  #DO NOT USE THIS!
        if True:
        #Freeze everything
            for name,param in model.named_parameters():
                param.requires_grad = False
                if name == 'classifier.weight' or name == 'pre_classifier.weight':
                    param.requires_grad = True
                if name == 'classifier.bias' or name == 'pre_classifier.bias':
                    param.requires_grad = True
            print("Using linear probe!")

    loss_calc = torch.nn.CrossEntropyLoss() #Categorical cross entropy doesn't support one hot encoding  MSELoss
    #torch.nn.CrossEntropyLoss() accepts only integer labels, no one hot encoded 

    if args.read_nn:
        print("Reading: ", args.read_nn)
        model.load_state_dict(torch.load(args.read_nn))

    return model, loss_calc


#Done.
def get_line_search(args) -> Optional[LineSearchSpec]:
    """
    Get the line search spec from the arguments
    """
    lss = LineSearchSpec(
        extrapolation_factor=args.extrapolation_factor,
        sufficient_decrease=args.sufficient_decrease,
        curvature_constant=args.curvature_condition,
        max_searches=args.max_searches,
    )

    if None in (lss.max_searches, lss.sufficient_decrease):
        print("Using no line search")
        lss = None

    return lss

#Done.
def get_trust_region(args) -> TrustRegionSpec:
    """
    Get the trust region spec from the arguments
    """
    try:
        # We specifically constrain the other params to narrow the space
        tregion = TrustRegionSpec(
            initial_radius=args.initial_radius,
            max_radius=args.max_radius,
            nabla0=args.nabla0,
            nabla1=args.nabla1,
            nabla2=args.nabla2,
            trust_region_subproblem_iter=args.max_subproblem_iter,
        )
    except BadTrustRegionSpec:
        tregion = None

    return tregion

#Done.
def get_trust_or_search(args) -> Tuple[LineSearchSpec, TrustRegionSpec]:
    """Return the line search, trust region, or no spec"""
    lss = get_line_search(args)
    trs = get_trust_region(args)
    if lss is None:
        print("Not using Line Search!")
    if trs is None:
        print("Not using trust-region!")
    if lss is not None and trs is not None:
        warnings.warn(
            "Both Line search and trust region specified, defaulting to None!"
        )
        lss = None
        trs = None

    return (lss, trs)

#Done..
def get_optimizer(args, model, linear_probe:bool = True):
    """Get the optimizer, configured as desired."""
    lss, trs = get_trust_or_search(args)
    if linear_probe:
        #Freeze everything
        for name,param in model.named_parameters(): #TODO: You can do this loop once for the whole function when done outside of the huge if block
            param.requires_grad = False
            if name == 'classifier.weight' or name == 'pre_classifier.weight':
                param.requires_grad = True
            if name == 'classifier.bias' or name == 'pre_classifier.bias':
                param.requires_grad = True
        print("Using linear probe!")


    if args.opt == "sgd":
        if linear_probe:
            #Freeze everything
            for name,param in model.named_parameters(): #TODO: You can do this loop once for the whole function when done outside of the huge if block
                param.requires_grad = False
                if name == 'classifier.weight' or name == 'pre_classifier.weight':
                    param.requires_grad = True
                if name == 'classifier.bias' or name == 'pre_classifier.bias':
                    param.requires_grad = True
            print("Using linear probe!")
            optimizer = optim.SGD(model.parameters(),#[{"params" : [p for parameter_name, p in model.named_parameters() if 'score' in parameter_name][0]}],
                                  lr = args.learning_rate, 
                                  momentum = args.momentum)

            # print(optimizer.__getstate__()["param_groups"])
        else:
            optimizer = optim.SGD(
                model.parameters(), lr=args.learning_rate, momentum=args.momentum
            )
        print(f"optimizer set up with learning rate {args.learning_rate}\nand {args.momentum} momentum")

    elif args.opt == "kn":
        if trs is not None:
            raise ValueError("kn does not support trust region!")
        optimizer = optim_hfcr_newton.HFCR_Newton(
            model.parameters(),
            lr=args.learning_rate,
            max_newton=args.max_newton,
            abs_newton_tol=args.abs_newton_tol,
            rel_newton_tol=args.rel_newton_tol,
            max_cr=args.max_cr,
            cr_tol=args.cr_tol,
            line_search_spec=lss,
        )

    elif args.opt == "bfgs":
        if trs is None:
            optimizer = quasi_newton.BFGS(
                model.parameters(),
                lr=args.learning_rate,
                max_newton=args.max_newton,
                abs_newton_tol=args.abs_newton_tol,
                rel_newton_tol=args.rel_newton_tol,
                max_krylov=args.max_cr,
                krylov_tol=args.cr_tol,
                matrix_free_memory=args.memory,
                line_search=lss,
            )
        else:
            optimizer = quasi_newton.BFGSTrust(
                model.parameters(),
                lr=args.learning_rate,
                max_newton=args.max_newton,
                abs_newton_tol=args.abs_newton_tol,
                rel_newton_tol=args.rel_newton_tol,
                max_krylov=args.max_cr,
                krylov_tol=args.cr_tol,
                matrix_free_memory=args.memory,
                trust_region=trs,
            )

    elif args.opt == "bfgsi":
        if trs is None:
            optimizer = quasi_newton.BFGSInverse(
                model.parameters(),
                lr=args.learning_rate,
                max_newton=args.max_newton,
                abs_newton_tol=args.abs_newton_tol,
                rel_newton_tol=args.rel_newton_tol,
                matrix_free_memory=args.memory,
                line_search=lss,
            )
        else:
            # TODO
            pass

    elif args.opt == "sr1":
        if trs is None:
            optimizer = quasi_newton.SymmetricRankOne(
                model.parameters(),
                lr=args.learning_rate,
                max_newton=args.max_newton,
                abs_newton_tol=args.abs_newton_tol,
                rel_newton_tol=args.rel_newton_tol,
                max_krylov=args.max_cr,
                krylov_tol=args.cr_tol,
                matrix_free_memory=args.memory,
                line_search=lss,
            )
        else:
            optimizer = quasi_newton.SymmetricRankOneTrust(
                model.parameters(),
                lr=args.learning_rate,
                max_newton=args.max_newton,
                abs_newton_tol=args.abs_newton_tol,
                rel_newton_tol=args.rel_newton_tol,
                matrix_free_memory=args.memory,
                trust_region=trs,
            )

    elif args.opt == "sr1d":
        if trs is None:
            optimizer = quasi_newton.SymmetricRankOneInverse(
                model.parameters(),
                lr=args.learning_rate,
                max_newton=args.max_newton,
                abs_newton_tol=args.abs_newton_tol,
                rel_newton_tol=args.rel_newton_tol,
                matrix_free_memory=args.memory,
                line_search=lss,
            )
        else:
            optimizer = quasi_newton.SymmetricRankOneDualTrust(
                model.parameters(),
                lr=args.learning_rate,
                max_newton=args.max_newton,
                abs_newton_tol=args.abs_newton_tol,
                rel_newton_tol=args.rel_newton_tol,
                matrix_free_memory=args.memory,
                trust_region=trs,
            )

    elif args.opt == "dfp":
        if trs is None:
            optimizer = quasi_newton.DavidonFletcherPowell(
                model.parameters(),
                lr=args.learning_rate,
                max_newton=args.max_newton,
                abs_newton_tol=args.abs_newton_tol,
                rel_newton_tol=args.rel_newton_tol,
                max_krylov=args.max_cr,
                krylov_tol=args.cr_tol,
                matrix_free_memory=args.memory,
                line_search=lss,
            )
        else:
            optimizer = quasi_newton.DavidonFletcherPowellTrust(
                model.parameters(),
                lr=args.learning_rate,
                max_newton=args.max_newton,
                abs_newton_tol=args.abs_newton_tol,
                rel_newton_tol=args.rel_newton_tol,
                max_krylov=args.max_cr,
                krylov_tol=args.cr_tol,
                matrix_free_memory=args.memory,
                trust_region=trs,
            )

    elif args.opt == "dfpi":
        if trs is None:
            optimizer = quasi_newton.DavidonFletcherPowellInverse(
                model.parameters(),
                lr=args.learning_rate,
                max_newton=args.max_newton,
                abs_newton_tol=args.abs_newton_tol,
                rel_newton_tol=args.rel_newton_tol,
                matrix_free_memory=args.memory,
                line_search=lss,
            )
        else:
            # TODO
            pass

    elif args.opt == "broy":
        if trs is None:
            optimizer = quasi_newton.Broyden(
                model.parameters(),
                lr=args.learning_rate,
                max_newton=args.max_newton,
                abs_newton_tol=args.abs_newton_tol,
                rel_newton_tol=args.rel_newton_tol,
                max_krylov=args.max_cr,
                krylov_tol=args.cr_tol,
                matrix_free_memory=args.memory,
                line_search=lss,
            )
        else:
            # TODO
            pass
            optimizer = quasi_newton.BroydenTrust(
                model.parameters(),
                lr=args.learning_rate,
                max_newton=args.max_newton,
                abs_newton_tol=args.abs_newton_tol,
                rel_newton_tol=args.rel_newton_tol,
                max_krylov=args.max_cr,
                krylov_tol=args.cr_tol,
                matrix_free_memory=args.memory,
                trust_region=trs,
            )

    elif args.opt == "broyi":
        if trs is None:
            optimizer = quasi_newton.BrodyenInverse(
                model.parameters(),
                lr=args.learning_rate,
                max_newton=args.max_newton,
                abs_newton_tol=args.abs_newton_tol,
                rel_newton_tol=args.rel_newton_tol,
                matrix_free_memory=args.memory,
                line_search=lss,
            )
        else:
            # TODO
            pass

    elif args.opt == "fr":
        if trs is not None:
            warnings.warn(
                "NLCG methods don't support Trust Regions, defaulting to no line search!"
            )
        optimizer = nlcg.FletcherReeves(
            model.parameters(),
            lr=args.learning_rate,
            max_newton=args.max_newton,
            abs_newton_tol=args.abs_newton_tol,
            rel_newton_tol=args.rel_newton_tol,
            line_search_spec=lss,
        )

    elif args.opt == "pr":
        if trs is not None:
            warnings.warn(
                "NLCG methods don't support Trust Regions, defaulting to no line search!"
            )
        optimizer = nlcg.PolakRibiere(
            model.parameters(),
            lr=args.learning_rate,
            max_newton=args.max_newton,
            abs_newton_tol=args.abs_newton_tol,
            rel_newton_tol=args.rel_newton_tol,
            line_search_spec=lss,
        )

    elif args.opt == "hs":
        if trs is not None:
            warnings.warn(
                "NLCG methods don't support Trust Regions, defaulting to no line search!"
            )
        optimizer = nlcg.HestenesStiefel(
            model.parameters(),
            lr=args.learning_rate,
            max_newton=args.max_newton,
            abs_newton_tol=args.abs_newton_tol,
            rel_newton_tol=args.rel_newton_tol,
            line_search_spec=lss,
        )

    elif args.opt == "dy":
        if trs is not None:
            warnings.warn(
                "NLCG methods don't support Trust Regions, defaulting to no line search!"
            )
        optimizer = nlcg.DaiYuan(
            model.parameters(),
            lr=args.learning_rate,
            max_newton=args.max_newton,
            abs_newton_tol=args.abs_newton_tol,
            rel_newton_tol=args.rel_newton_tol,
            line_search_spec=lss,
        )

    else:
        valid_opts = (
            "sgd",
            "lbfgs",
            "kn",
            "sr1",
            "sr1d",
            "dfp",
            "dfpi",
            "bfgs",
            "bfgsi",
            "broy",
            "broyi",
            "fr",
            "pr",
            "hs",
            "dy",
        )
        raise ValueError(
            f"Invalid optimizer specified: {args.opt}, must be one of {valid_opts}"
        )

    print("Optimizer:", type(optimizer))
    return optimizer


def train(args, model, device, train_loader, optimizer, epoch, loss_calc, linear_probe: bool = True):
    """Perform a training epoch using a first order optimizer"""
    # print("Using normal train function")
    model.to(device)
    model.train()

    if linear_probe:
        #Freeze everything
        for name,param in model.named_parameters():
            param.requires_grad = False
            if name == 'classifier.weight' or name == 'pre_classifier.weight':
                param.requires_grad = True
            if name == 'classifier.bias' or name == 'pre_classifier.bias':
                param.requires_grad = True
        print("Using linear probe!")

    correct = 0
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):

        #Split data into the actual inputs and the mask
        n = data.shape[1]
        n = n//2
        data, attention_mask =  data[:,:n].to(device), data[:,n:].to(device)
        target.type(torch.LongTensor)
        target.to(device)

        optimizer.zero_grad()

        #The output of the model is a "SequenceClassifierOutputWithPast" ?
        output = model(input_ids=data, attention_mask=attention_mask,output_attentions=False,output_hidden_states=False)
        softmax_out = torch.nn.functional.softmax(output.logits, dim = 1) #OLD

        #Next two lines solved a strange device mismatch error.
        softmax_out = softmax_out.to('cuda')
        target = target.to('cuda')
        # target = target.to(torch.float32)

        loss = loss_calc(softmax_out, target)  #loss_calc is MSE loss
        # loss.requires_grad = True
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if math.isnan(train_loss):
            accuracy = float("nan")
            break
        
        #Returns a tensor, where each entry is the index of the largest element in a row. 
        model_class_prediction = torch.argmax(softmax_out, dim = 1)
        # target_index = torch.argmax(target, dim = 1) OLD
        

        #Calculate number correct in this batch
        # batch_correct = torch.sum(model_class_prediction == target_index).item()gpt
        batch_correct = torch.sum(model_class_prediction == target).item()

        correct += batch_correct

        #Compare `score` layer before and after update

        # old_score_layer = [p for parameter_name, p in model.named_parameters() if 'classifier' in parameter_name][0][0].clone().detach() #this is selecting the pre classifier weight, which is frozen..
        # old_score_layer = [p for parameter_name, p in model.named_parameters() if 'classifier.weight' == parameter_name or 'classifier.bias' == parameter_name][0][0].clone().detach() #this is selecting the pre classifier, which is frozen..
        # print(list(model.named_parameters())[-1][1].grad)
        # new_score_layer = [p for parameter_name, p in model.named_parameters() if 'classifier.weight' == parameter_name or 'classifier.bias' == parameter_name][0][0].clone().detach()

        if batch_idx % args.log_interval == 0:
            #Print the parameters that are passed to the optimizer
            # print(optimizer.__getstate__()["param_groups"])
            # print("The score layer difference is zero:")
            # diff = new_score_layer - old_score_layer
            # print(torch.all(diff == 0))
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                ))
    print(
        "\nTrain set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            train_loss,
            correct,
            len(train_loader.dataset),
            100.0 * correct / len(train_loader.dataset),
        )
    )
    accuracy = 100.0 * correct / len(train_loader.dataset)

    return accuracy, train_loss


def train_sso(args, model, device, train_loader, optimizer, epoch, loss_calc, linear_probe: bool = True):
    """Perform a training epoch using a Second Order Optimizer"""
    model.to(device)
    model.train()

    if linear_probe:
        print("Using linear probe!",)
        #Freeze everything
        for name,param in model.named_parameters():
            param.requires_grad = False
            #Unfreeze last layer
            if name == 'classifier.weight' or name == 'pre_classifier.weight':
                param.requires_grad = True
            if name == 'classifier.bias' or name == 'pre_classifier.bias':
                param.requires_grad = True

    train_loss = 0.0
    correct = 0

    print(f"Beginning training loop for epoch {epoch}")
    for batch_idx, (data, target) in enumerate(train_loader):

        torch.cuda.empty_cache()
        # if batch_idx % 10 == 0:
        #     print(torch.cuda.memory_allocated())
        #     print(torch.cuda.memory_summary(device=None, abbreviated=False))


        #Split data into the actual inputs and the mask
        n = data.shape[1]
        n = n//2

        data, attention_mask =  data[:,:n].to(device), data[:,n:].to(device)
        target.type(torch.LongTensor)
        target.to(device)
        # print(f"Data and target are now on {device} device. Memory usage is {torch.cuda.memory_allocated()/(10e9)} GB", flush=True)

        def closure():
            #nonlocal means the varaible that comes after should not belond to inner function. In this case the closure() function
            nonlocal data
            nonlocal target
            optimizer.zero_grad()
            # output = checkpoint(model, data)
            time.sleep(1)
            output = model(input_ids=data, attention_mask=attention_mask,output_attentions=False,output_hidden_states=False)
            softmax_out = torch.nn.functional.softmax(output.logits, dim = 1)

            #Next two lines solved a strange device mismatch error.
            softmax_out = softmax_out.to(device)
            target = target.to(device)
            # target = target.to(torch.float32)
            # print(f"softmax out is {softmax_out}")
            # print(f"target is {target}")
            loss = loss_calc(softmax_out, target)  #loss_calc is MSE loss
            try:
                # print(f"The memory usage before backprop is {torch.cuda.memory_allocated()/(10e9)} GB")
                loss.backward()
                # print(f"The memory usage after backprop is {torch.cuda.memory_allocated()/(10e9)} GB")
            except RuntimeError:
                # we're in a scope that has disabled grad, "probably" on purpose
                pass
            return loss
            #nonlocal means the varaible that comes after should not belond to inner function. In this case the closure() function
            # output =  model(input_ids=prompt,attention_mask=mask,output_attentions=False,output_hidden_states=False)

        # old_score_layer = [p for parameter_name, p in model.named_parameters() if 'classifier.weight' == parameter_name or 'classifier.bias' == parameter_name][0][0].clone().detach()
        loss = optimizer.step(closure = closure)
        # new_score_layer = [p for parameter_name, p in model.named_parameters() if 'classifier.weight' == parameter_name or 'classifier.bias' == parameter_name][0][0].clone().detach()
        

        train_loss += loss.item()
        if math.isnan(train_loss):
            break

        with torch.no_grad():
            output = model(input_ids = data, attention_mask = attention_mask, output_attentions=False,output_hidden_states=False)
            probs = torch.nn.functional.softmax(output.logits, dim = 1)
            
            #Returns a tensor, where each entry is the index of the largest element in a row. 
            model_class_prediction = torch.argmax(probs, dim = 1)
            # target_index = torch.argmax(target, dim = 1)

            #Calculate number correct in this batch
            batch_correct = torch.sum(model_class_prediction == target).item()
            correct += batch_correct


        accuracy = 100.0 * correct / len(train_loader.dataset)
        time.sleep(1)
        if batch_idx % args.log_interval == 0:
            # print(optimizer.__getstate__()["param_groups"])

            # print("The score layer difference zero:")
            # diff = new_score_layer - old_score_layer
            # print(torch.all(diff == 0))        # print("Calculating loss. Via optimizer.step")
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                ))

    accuracy = 100.0 * correct / len(train_loader.dataset)
    print(
        "\nTrain set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            train_loss,
            correct,
            len(train_loader.dataset),
            100.0 * correct / len(train_loader.dataset),
        ))

    return accuracy, train_loss


def test(args, model, device, test_loader, loss_calc):
    """Evaluate the model using a test set."""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            #Split data into the actual inputs and the mask
            n = data.shape[1]
            n = n//2
            data, attention_mask = data[:,:n].to(device), data[:,n:].to(device)
            target = target.type(torch.LongTensor)
            target = target.to(device)

            output = model(input_ids=data, attention_mask=attention_mask,output_attentions=False,output_hidden_states=False)

            softmax_out = torch.nn.functional.softmax(output.logits, dim = 1)

            #Next two lines solved a strange device mismatch error.
            softmax_out = softmax_out.to(device)
            target = target.to(device)
            # target = target.to(torch.float32)

            test_loss += loss_calc(softmax_out, target).item()
            probs = torch.nn.functional.softmax(output.logits, dim = 1)

            #Returns a tensor, where each entry is the index of the largest element in a row. 
            model_class_prediction = torch.argmax(probs, dim = 1)
            # target_index = torch.argmax(target, dim = 1)

            #Calculate number correct in this batch
            # batch_correct = torch.sum(model_class_prediction == target_index).item()
            batch_correct = torch.sum(model_class_prediction == target).item()

            correct += batch_correct


    test_loss /= len(test_loader.dataset)#TODO: Remove me

    # time.sleep(1)
    print(
        "Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    time.sleep(1)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    return accuracy, test_loss


def _main(
    opt: str,
    dataset: str,
    batch_size_train: int = 60000,
    batch_size_test: int = 1000,
    momentum: float = 0.9,
    hidden: int = 15,
    max_newton: int = 10,
    abs_newton_tol: float = 1e-5,
    rel_newton_tol: float = 1e-8,
    max_cr: int = 10,
    cr_tol: float = 1e-3,
    learning_rate: float = 0.01,
    sufficient_decrease: Optional[float] = None,
    curvature_condition: Optional[float] = None,
    extrapolation_factor: Optional[float] = None,
    max_searches: Optional[float] = None,
    initial_radius: Optional[float] = None,
    max_radius: Optional[float] = None,
    nabla0: Optional[float] = None,
    nabla1: Optional[float] = None,
    nabla2: Optional[float] = None,
    shrink_factor: Optional[float] = None,
    growth_factor: Optional[float] = None,
    max_subproblem_iter: Optional[int] = None,
    num_epoch: int = 12,
    seed: int = 1,
    read_nn: Optional[str] = None,
    write_nn: bool = True,
    log_interval: int = 10,
    device: str = "cuda",
    record: Optional[Union[Path, str]] = None,
    memory: Optional[int] = None,
    parallel = False
):
    """Do the things."""
    args = SimpleNamespace(
        opt=opt,
        dataset=dataset, #Unused in this rendition. 
        batch_size_train=batch_size_train,
        batch_size_test=batch_size_test,
        momentum=momentum,
        hidden=hidden,
        max_newton=max_newton,
        abs_newton_tol=abs_newton_tol,
        rel_newton_tol=rel_newton_tol,
        max_cr=max_cr,
        cr_tol=cr_tol,
        learning_rate=learning_rate,
        sufficient_decrease=sufficient_decrease,
        curvature_condition=curvature_condition,
        extrapolation_factor=extrapolation_factor,
        max_searches=max_searches,
        initial_radius=initial_radius,
        max_radius=max_radius,
        nabla0=nabla0,
        nabla1=nabla1,
        nabla2=nabla2,
        shrink_factor=shrink_factor,
        growth_factor=growth_factor,
        max_subproblem_iter=max_subproblem_iter,
        num_epoch=num_epoch,
        seed=seed,
        read_nn=read_nn,
        write_nn=write_nn,
        log_interval=log_interval,
        device=device,
        record=record,
        memory=memory
    )
    if os.path.isfile(args_to_fname(vars(args), "json")):
        print(f"File already exists {vars(args)}.json, not re-running experiment!")
        return

    torch.manual_seed(args.seed)
    device_name = args.device

    device = torch.device(device_name)

    model, loss_calc = get_model_and_loss(args, device)

    train_loader, test_loader = load_data(
        args.batch_size_train, args.batch_size_test, device_name
    )

    # print("putting linear probe in main!")
    if False:
        print("Using linear probe!", flush = True)
        #Freeze everything
        for name,param in model.named_parameters():
            param.requires_grad = False
            #Unfreeze last layer
            if name == 'score.weight':
                param.requires_grad = True
            if name == 'score.bias':
                param.requires_grad = True

    optimizer = get_optimizer(args, model)

    times = []
    train_loss_list = []
    test_loss_list = []

    train_accuracy = []
    test_accuracy = []

    first_order_optimizers = ("sgd",)
    second_order_optimizers = (
        "lbfgs",
        "kn",
        "kn2",
        "sr1",
        "sr1d",
        "bfgs",
        "bfgsi",
        "dfp",
        "dfpi",
        "broy",
        "broyi",
        "fr",
        "pr",
        "hs",
        "dy",
    )

    for epoch in range(1, args.num_epoch + 1):
        if args.opt in first_order_optimizers:
            t_start = perf_counter()
            train_acc, train_loss = train(
                args, model, device, train_loader, optimizer, epoch, loss_calc
            )
            t_stop = perf_counter()
        elif args.opt in second_order_optimizers:
            t_start = perf_counter()
            train_acc, train_loss = train_sso(
                args, model, device, train_loader, optimizer, epoch, loss_calc
            )
            t_stop = perf_counter()
        else:
            raise ValueError(f'Invalid optimizer "{args.opt}" requested!')

        t_elaps = t_stop - t_start
        times.append(t_elaps)
        print(f"Epoch {epoch} took {t_elaps}")
        train_accuracy.append(train_acc)
        train_loss_list.append(train_loss)
        test_acc, test_loss = test(args, model, device, test_loader, loss_calc)
        test_accuracy.append(test_acc)
        test_loss_list.append(test_loss)
        if math.isnan(train_loss):
            print("=" * 80)
            print(f"NaN training loss, its borked at epoch {epoch}!")
            print("=" * 80)
            break

    total_time = sum(times)
    print("The train loss list is: ", train_loss_list)
    print("The average test loss list is: ", test_loss_list)
    print("The train accuracy is: ", train_accuracy)
    print("The test accuracy is: ", test_accuracy)
    print("The time list is: ", times)
    print("The total training time is: ", total_time)

    if args.write_nn:
        fname = args_to_fname(vars(args), "pkl")
        torch.save(model.state_dict(), fname)
        print(f"Model saved {fname}")

    if args.record is not None:
        rslts = {}
        rslts["specs"] = vars(args)
        rslts["time"] = times
        rslts["train_loss_list"] = train_loss_list
        rslts["test_loss_list"] = test_loss_list
        rslts["test_accuracy_list"] = test_accuracy
        rslts["train_accuracy_list"] = train_accuracy

        fname = args_to_fname(vars(args), "json")

        with open(fname, "w", encoding="UTF-8") as outfile:
            json.dump(rslts, outfile, indent=4)


def main(
    opt: str = "sgd",
    dataset: str = "yahoo_answers",
    batch_size_train: int = 60000,
    batch_size_test: int = 1000,
    momentum: float = 0.9,
    hidden: int = 15,
    max_newton: int = 10,
    abs_newton_tol: float = 1e-5,
    rel_newton_tol: float = 1e-8,
    max_cr: int = 10,
    cr_tol: float = 1e-3,
    learning_rate: float = 0.01,
    sufficient_decrease: Optional[float] = None,
    curvature_condition: Optional[float] = None,
    extrapolation_factor: Optional[float] = None,
    max_searches: Optional[float] = None,
    initial_radius: Optional[float] = None,
    max_radius: Optional[float] = None,
    nabla0: Optional[float] = None,
    nabla1: Optional[float] = None,
    nabla2: Optional[float] = None,
    shrink_factor: Optional[float] = None,
    growth_factor: Optional[float] = None,
    max_subproblem_iter: Optional[int] = None,
    num_epoch: int = 12,
    seed: int = 1,
    read_nn: Optional[str] = None,
    write_nn: bool = True,
    log_interval: int = 10,
    device: str = "cuda",
    record: Optional[Union[Path, str]] = None,
    memory: Optional[int] = None,
    parallel = False
):
    """Wrapper for internal main to catch a CUDA runtime error"""
    try:
        _main(
            opt,
            dataset,
            batch_size_train,
            batch_size_test,
            momentum,
            hidden,
            max_newton,
            abs_newton_tol,
            rel_newton_tol,
            max_cr,
            cr_tol,
            learning_rate,
            sufficient_decrease,
            curvature_condition,
            extrapolation_factor,
            max_searches,
            initial_radius,
            max_radius,
            nabla0,
            nabla1,
            nabla2,
            shrink_factor,
            growth_factor,
            max_subproblem_iter,
            num_epoch,
            seed,
            read_nn,
            write_nn,
            log_interval,
            device,
            record,
            memory,
        )
    except RuntimeError as runtime_error:
        if "CUDA out of memory" in runtime_error.args[0]:
            print("=" * 80)
            print("Need more vram!")
            arg_dict = dict(
                opt=opt,
                dataset=dataset,
                batch_size_train=batch_size_train,
                batch_size_test=batch_size_test,
                momentum=momentum,
                hidden=hidden,
                max_newton=max_newton,
                abs_newton_tol=abs_newton_tol,
                rel_newton_tol=rel_newton_tol,
                max_cr=max_cr,
                cr_tol=cr_tol,
                learning_rate=learning_rate,
                sufficient_decrease=sufficient_decrease,
                curvature_condtion=curvature_condition,
                extrapolation_factor=extrapolation_factor,
                max_searches=max_searches,
                initial_radius=initial_radius,
                max_radius=max_radius,
                nabla0=nabla0,
                nabla1=nabla1,
                nabla2=nabla2,
                shrink_factor=shrink_factor,
                growth_factor=growth_factor,
                max_subproblem_iter=max_subproblem_iter,
                num_epoch=num_epoch,
                seed=seed,
                read_nn=read_nn,
                write_nn=write_nn,
                log_interval=log_interval,
                device=device,
                record=record,
                memory=memory,
            )
            print(args_to_fname(arg_dict, ""))
            print("=" * 80)
            print(runtime_error)
            raise runtime_error
        else:
            raise runtime_error


if __name__ == "__main__":
    args_ = get_args()
    
    main(
        opt=args_.opt,
        dataset=args_.dataset,
        batch_size_train=args_.batch_size_train,
        batch_size_test=args_.batch_size_test,
        momentum=args_.momentum,
        hidden=args_.hidden,
        max_newton=args_.max_newton,
        abs_newton_tol=args_.abs_newton_tol,
        rel_newton_tol=args_.rel_newton_tol,
        max_cr=args_.max_cr,
        cr_tol=args_.cr_tol,
        learning_rate=args_.learning_rate,
        sufficient_decrease=args_.sufficient_decrease,
        curvature_condition=args_.curvature_condition,
        extrapolation_factor=args_.extrapolation_factor,
        max_searches=args_.max_searches,
        initial_radius=args_.initial_radius,
        max_radius=args_.max_radius,
        nabla0=args_.nabla0,
        nabla1=args_.nabla1,
        nabla2=args_.nabla2,
        shrink_factor=args_.shrink_factor,
        growth_factor=args_.growth_factor,
        max_subproblem_iter=args_.max_subproblem_iter,
        num_epoch=args_.num_epoch,
        seed=args_.seed,
        read_nn=args_.read_nn,
        write_nn=args_.write_nn,
        log_interval=args_.log_interval,
        device=args_.device,
        record=args_.record,
        memory=args_.memory,
    )
