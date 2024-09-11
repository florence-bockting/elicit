import tensorflow as tf
import pandas as pd

from functions.prior_simulation import Priors
from functions.model_simulation import simulate_from_generator
from functions.targets_elicits_computation import (
    computation_target_quantities,
    computation_elicited_statistics,
)
from functions.loss_computation import (
    compute_loss_components,
    compute_discrepancy,
    dynamic_weight_averaging,
)
from functions.training import training_loop
from functions.helper_functions import save_as_pkl


def prior_elicitation_dag(seed, global_dict: dict):
    """
    Defines the prior elicitation workflow and runs the workflow for the
    current setup as indicated by the user.

    Parameters
    ----------
    global_dict : dict
        generated through user inputs including all information about target
        quantities, elicitation techniques, optimization settings etc.

    """

    # set seed
    tf.random.set_seed(seed)
    regularizer_term = []

    def one_forward_simulation(prior_model, global_dict, ground_truth=False):
        """
        One forward simulation from prior samples to elicited statistics.

        Parameters
        ----------
        prior_model : instance of Priors class objects
            initialized prior distributions which can be used for sampling.
        global_dict : dict
            global dictionary with all user input specifications.
        ground_truth : bool, optional
            Is true if model should be learned with simulated data that
            represent a pre-defined ground truth. The default is False.

        Returns
        -------
        elicited_statistics : dict
            dictionary containing the elicited statistics that can be used to
            compute the loss components

        """
        prior_samples = prior_model()
        model_simulations = simulate_from_generator(
            prior_samples, ground_truth, global_dict
        )
        target_quantities = computation_target_quantities(
            model_simulations, ground_truth, global_dict
        )
        elicited_statistics = computation_elicited_statistics(
            target_quantities, ground_truth, global_dict
        )
        return elicited_statistics

    def load_expert_data(global_dict, path_to_expert_data=None):
        """
        Wrapper for loading the training data which can be expert data or
        data simulations using a pre-defined ground truth.

        Parameters
        ----------
        global_dict : dict
            global dictionary with all user input specifications.
        path_to_expert_data : str, optional
            path to file location where expert data has been saved

        Returns
        -------
        expert_data : dict
            dictionary containing the training data. Must have same form as the
            model-simulated elicited statistics.

        """
        if global_dict["expert_input"]["simulate_data"]:
            prior_model = Priors(global_dict=global_dict, ground_truth=True)
            expert_data = one_forward_simulation(
                prior_model, global_dict, ground_truth=True
            )
        else:
            # TODO: Checking: Must have the same shape/form as elicited statistics from model
            assert (
                global_dict["expert_input"]["data"] is not None
            ), "path to expert data has to provided"
            # load expert data from file
            expert_data = global_dict["expert_input"]["data"]
        return expert_data

    def compute_loss(
        training_elicited_statistics, expert_elicited_statistics, global_dict, epoch
    ):
        """
        Wrapper around the loss computation from elicited statistics to final
        loss value.

        Parameters
        ----------
        training_elicited_statistics : dict
            dictionary containing the expert elicited statistics.
        expert_elicited_statistics : TYPE
            dictionary containing the model-simulated elicited statistics.
        global_dict : dict
            global dictionary with all user input specifications.
        epoch : int
            epoch .

        Returns
        -------
        total_loss : float
            total loss value.

        """

        # regularization term for preventing degenerated solutions in var collapse to zero
        # used from Manderson and Goudie (2024)
        def regulariser(prior_samples):
            log_sd = tf.math.log(tf.math.reduce_std(prior_samples, 1))
            mean_log_sd = tf.reduce_mean(log_sd)
            return -mean_log_sd

        def compute_total_loss(epoch, loss_per_component, global_dict):
            """
            applies dynamic weight averaging for multi-objective loss function
            if specified. If loss_weighting has been set to None, all weights
            get an equal weight of 1.

            Parameters
            ----------
            epoch : int
                curernt epoch.
            loss_per_component : list of floats
                list of loss values per loss component.
            global_dict : dict
                global dictionary with all user input specifications.

            Returns
            -------
            total_loss : float
                total loss value (either weighted or unweighted).

            """
            loss_per_component_current = loss_per_component
            # TODO: check whether indicated loss weighting input is implemented

            if global_dict["loss_function"]["loss_weighting"] is None:
                total_loss = tf.math.reduce_sum(loss_per_component)
                priorsamples = pd.read_pickle(
                    global_dict["output_path"]["data"] + "/model_simulations.pkl"
                )["prior_samples"]
                regul_term = regulariser(priorsamples)
                regularizer_term.append(regul_term)

                path = global_dict["output_path"]["data"] + "/regularizer.pkl"
                save_as_pkl(regularizer_term, path)

                total_loss = total_loss + regul_term
            else:
                # apply selected loss weighting method
                if global_dict["loss_function"]["loss_weighting"]["method"] == "dwa":
                    # dwa needs information about the initial loss per component
                    if epoch == 0:
                        global_dict["loss_function"]["loss_weighting"]["method_specs"][
                            "loss_per_component_initial"
                        ] = loss_per_component
                    # apply method
                    total_loss = dynamic_weight_averaging(
                        epoch,
                        loss_per_component_current,
                        global_dict["loss_function"]["loss_weighting"]["method_specs"][
                            "loss_per_component_initial"
                        ],
                        global_dict["loss_function"]["loss_weighting"]["method_specs"][
                            "temperature"
                        ],
                        global_dict["output_path"]["data"],
                    )

                if global_dict["loss_function"]["loss_weighting"]["method"] == "custom":
                    total_loss = tf.math.reduce_sum(
                        tf.multiply(
                            loss_per_component,
                            global_dict["loss_function"]["loss_weighting"]["weights"],
                        )
                    )

            return total_loss

        loss_components_expert = compute_loss_components(
            expert_elicited_statistics, global_dict, expert=True
        )
        loss_components_training = compute_loss_components(
            training_elicited_statistics, global_dict, expert=False
        )
        loss_per_component = compute_discrepancy(
            loss_components_expert, loss_components_training, global_dict
        )
        weighted_total_loss = compute_total_loss(epoch, loss_per_component, global_dict)

        return weighted_total_loss

    def burnin_phase(
        expert_elicited_statistics, one_forward_simulation, compute_loss, global_dict
    ):
        """
        For the method "parametric_prior" it might be helpful to run different initializations
        before the actual training starts in order to find a 'good' set of initial values.
        For this purpose the burnin phase can be used. It rans multiple initializations and computes
        for each the respective loss value. At the end that set of initial values is chosen which
        leads to the smallest loss.

        Parameters
        ----------
        expert_elicited_statistics : dict
            dictionary with expert elicited statistics.
        one_forward_simulation : callable
            one forward simulation from prior samples to model-simulated elicited
            statistics.
        compute_loss : callable
            wrapper for loss computation from loss components to (weighted) total loss.
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
        for i in range(global_dict["burnin"]):
            print("|", end="")
            # prepare generative model
            prior_model = Priors(global_dict=global_dict, ground_truth=False)
            # generate simulations from model
            training_elicited_statistics = one_forward_simulation(
                prior_model, global_dict
            )
            # comput loss
            weighted_total_loss = compute_loss(
                training_elicited_statistics,
                expert_elicited_statistics,
                global_dict,
                epoch=0,
            )

            init_var_list.append(prior_model)
            save_prior.append(prior_model.trainable_variables)
            loss_list.append(weighted_total_loss.numpy())

        path = global_dict["output_path"]["data"] + "/burnin_phase.pkl"
        save_as_pkl((loss_list, save_prior), path)

        print(" ")
        return loss_list, init_var_list

    # get expert data
    expert_elicited_statistics = load_expert_data(global_dict)
    
    # compute loss for each set of initial values
    loss_list, init_prior = burnin_phase(
        expert_elicited_statistics, one_forward_simulation, compute_loss, global_dict
    )

    # extract minimum loss out of all runs and corresponding set of initial values
    min_index = tf.argmin(loss_list)
    init_prior_model = init_prior[min_index]

    # run training with optimal set of initial values
    training_loop(
                expert_elicited_statistics,
                init_prior_model,
                one_forward_simulation,
                compute_loss,
                global_dict,
                seed
            )
