import tensorflow_probability as tfp
from typing import TypedDict, Callable, Optional, Union, Tuple


class Hyper(TypedDict):
    name: str
    constraint: Callable
    constraint_name: str
    vtype: Callable
    dim: int
    shared: bool


class Parameter(TypedDict, total=False):
    name: str
    family: tfp.distributions.Distribution
    hyperparams: Optional[dict[str, Hyper]]
    constraint_name: str
    constraint: Callable


class QueriesDict(TypedDict, total=False):
    name: str
    value: Optional[Union[Callable, Tuple]]
    func_name: str


class Target(TypedDict):
    name: str
    query: QueriesDict
    target_method: Optional[Callable]
    loss: Callable
    weight: float


class ExpertDict(TypedDict, total=False):
    ground_truth: dict
    num_samples: int
    data: dict[str, list]


class Uniform(TypedDict):
    radius: Union[float,list]
    mean: Union[float,list]
    hyper: Optional[list]


class Initializer(TypedDict):
    method: Optional[str]
    distribution: Optional[Uniform]
    loss_quantile: Optional[float]
    iterations: Optional[int]
    hyperparams: Optional[dict]


class Trainer(TypedDict, total=False):
    method: str
    seed: int
    B: int
    num_samples: int
    epochs: int
    seed_chain: int


class NFDict(TypedDict):
    inference_network: Callable
    network_specs: dict
    base_distribution: Callable


class SaveHist(TypedDict):
    loss: bool
    time: bool
    loss_component: bool
    hyperparameter: bool
    hyperparameter_gradient: bool


class SaveResults(TypedDict):
    target_quantities: bool
    elicited_statistics: bool
    prior_samples: bool
    model_samples: bool
    expert_elicited_statistics: bool
    expert_prior_samples: bool
    init_loss_list: bool
    init_prior: bool
    init_matrix: bool
    loss_tensor_expert: bool
    loss_tensor_model: bool


class Parallel(TypedDict):
    runs: int
    cores: int
    seeds: Optional[list]
