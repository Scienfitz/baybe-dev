"""Test for the serialization of naive hybrid recommenders."""

import pytest

from baybe.core import BayBE
from baybe.searchspace import SearchSpaceType
from baybe.strategies.bayesian import (
    BayesianRecommender,
    NaiveHybridRecommender,
    SequentialGreedyRecommender,
)
from baybe.strategies.recommender import NonPredictiveRecommender
from baybe.utils import get_subclasses

valid_discrete_non_predictive_recommenders = [
    cls()
    for cls in get_subclasses(NonPredictiveRecommender)
    if cls.compatibility
    in [SearchSpaceType.DISCRETE, SearchSpaceType.EITHER, SearchSpaceType.HYBRID]
]
valid_discrete_bayesian_recommenders = [
    cls()
    for cls in get_subclasses(BayesianRecommender)
    if cls.compatibility
    in [SearchSpaceType.DISCRETE, SearchSpaceType.EITHER, SearchSpaceType.HYBRID]
]
valid_naive_hybrid_recommenders = [
    NaiveHybridRecommender(
        disc_recommender=disc, cont_recommender=SequentialGreedyRecommender()
    )
    for disc in [
        *valid_discrete_non_predictive_recommenders,
        *valid_discrete_bayesian_recommenders,
    ]
]


@pytest.mark.parametrize("recommender", valid_naive_hybrid_recommenders)
@pytest.mark.parametrize(
    "parameter_names",
    [["Categorical_1", "SomeSetting", "Num_disc_1", "Conti_finite1", "Conti_finite2"]],
)
def test_serialization_without_recommendation(baybe):
    """Serialize all possible hybrid recommender objects and test for equality"""
    baybe_orig_string = baybe.to_json()
    baybe_recreate = BayBE.from_json(baybe_orig_string)
    assert baybe == baybe_recreate