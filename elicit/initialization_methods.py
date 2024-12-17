# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import tensorflow_probability as tfp

from elicit.helper_functions import save_as_pkl
from elicit.prior_simulation import Priors

tfd = tfp.distributions


def init_method(hyppar, n_warm_up, method, mean, radius):
    """
    Initialize multivariate normal prior over hyperparameter values

    Parameters
    ----------
    n_hypparam : int
        Number of hyperparameters.
    n_warm_up : int
        number of warmup iterations.

    Returns
    -------
    mvdist : tf.tensor
        samples from the multivariate prior
        (shape=(n_warm_up, n_hyperparameters).

    """

    assert method in ["random", "sobol"], "The initialization method must be one of the following: 'sobol', 'random'"  # noqa
    
    n_hypparam = len(hyppar)
    
    if method == "random":
        uniform_samples = tfd.Uniform(tf.subtract(mean,radius), 
                                      tf.add(mean,radius)).sample(n_warm_up)
    elif method == "sobol":
        # Sobol samples
        sobol_samples = tf.math.sobol_sample(n_hypparam, n_warm_up)
        # Inverse transform to get samples from normal
        uniform_samples = tfd.Uniform(tf.subtract(mean,radius), 
                                  tf.add(mean,radius)).quantile(sobol_samples)

    res_dict = {f"{hyppar[i]}": uniform_samples[:,i] for i in range(n_hypparam)
                }

    return res_dict


def initialization_phase(
    expert_elicited_statistics, one_forward_simulation, compute_loss,
    global_dict,
):
    """
    For the method "parametric_prior" it might be helpful to run different
    initializations before the actual training starts in order to find a
    'good' set of initial values. For this purpose the burnin phase can be
    used. It rans multiple initializations and computes for each the
    respective loss value. At the end that set of initial values is chosen
    which leads to the smallest loss.

    Parameters
    ----------
    expert_elicited_statistics : dict
        dictionary with expert elicited statistics.
    one_forward_simulation : callable
        one forward simulation from prior samples to model-simulated elicited
        statistics.
    compute_loss : callable
        wrapper for loss computation from loss components to (weighted) total
        loss.
    global_dict : dict
        global dictionary with all user input specifications.

    Returns
    -------
    loss_list : list
        list containing the loss values for each set of initial values.
    init_var_list : list
        set of initial values for each run.

    """

    loss_list = []
    init_var_list = []
    save_prior = []
    dict_copy = dict(global_dict)

    # get number of hyperparameters
    n_hypparam = 0

    for i in range(len(global_dict["model_parameters"])):
        n_hypparam += len(
            global_dict["model_parameters"][i]["hyperparams"]
        )

    # create initializations
    init_matrix = init_method(
        global_dict["initialization_settings"]["hyppar"],
        dict_copy["initialization_settings"]["number_of_iterations"],
        global_dict["initialization_settings"]["method"],
        global_dict["initialization_settings"]["mean"],
        global_dict["initialization_settings"]["radius"]
        )

    path = dict_copy["training_settings"][
        "output_path"] + "/initialization_matrix.pkl"
    save_as_pkl(init_matrix, path)

    for i in range(dict_copy["initialization_settings"][
            "number_of_iterations"]):
        dict_copy["training_settings"]["seed"] = (
            dict_copy["training_settings"]["seed"] + i
        )
        # create init-matrix-slice
        init_matrix_slice = {f"{key}": init_matrix[key][i] for key in init_matrix}

        # prepare generative model
        prior_model = Priors(global_dict=dict_copy,
                             ground_truth=False,
                             init_matrix_slice=init_matrix_slice)

        # generate simulations from model
        training_elicited_statistics = one_forward_simulation(prior_model,
                                                              dict_copy)

        # compute loss for each set of initial values
        weighted_total_loss = compute_loss(
            training_elicited_statistics,
            expert_elicited_statistics,
            dict_copy,
            epoch=0,
        )
        print(f"({i}) {weighted_total_loss.numpy():.1f} ", end="")

        init_var_list.append(prior_model)
        save_prior.append(prior_model.trainable_variables)
        loss_list.append(weighted_total_loss.numpy())

    path = dict_copy["training_settings"][
        "output_path"] + "/pre_training_results.pkl"
    save_as_pkl((loss_list, save_prior), path)

    print(" ")
    return loss_list, init_var_list
