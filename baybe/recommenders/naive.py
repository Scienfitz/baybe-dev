"""Naive hybrid space recommender."""

import warnings
from typing import ClassVar, Optional

import pandas as pd
from attrs import define, evolve, field, fields

from baybe.acquisition import PartialAcquisitionFunction
from baybe.recommenders.pure.base import PureRecommender
from baybe.recommenders.pure.bayesian.base import BayesianRecommender
from baybe.recommenders.pure.bayesian.sequential_greedy import (
    SequentialGreedyRecommender,
)
from baybe.recommenders.pure.nonpredictive.base import NonPredictiveRecommender
from baybe.searchspace import SearchSpace, SearchSpaceType
from baybe.utils.dataframe import to_tensor


@define
class NaiveHybridSpaceRecommender(PureRecommender):
    """Recommend points by independent optimization of subspaces.

    This recommender splits the hybrid search space in the discrete and continuous
    subspace. Each of the subspaces is optimized on its own, and the recommenders for
    those subspaces can be chosen upon initialization. If this recommender is used on
    a non-hybrid space, it uses the corresponding recommender.
    """

    # Class variables
    compatibility: ClassVar[SearchSpaceType] = SearchSpaceType.HYBRID
    # See base class.

    # Object variables
    # TODO This used to be a Union of BayesianRecommender and NonPredictiveRecommender.
    # Due to serialization issues, this was changed to PureRecommender in general.
    # As we currently do not have other subclasses of PureRecommender, this solution
    # works for now. Still, we manually check whether the disc_recommender belongs to
    # one of these two subclasses such that we might be able to easily spot a potential
    # problem that might come up when implementing new subclasses of PureRecommender
    disc_recommender: PureRecommender = field(factory=SequentialGreedyRecommender)
    """The recommender used for the discrete subspace. Default:
    :class:`baybe.recommenders.pure.bayesian.sequential_greedy.SequentialGreedyRecommender`"""

    cont_recommender: BayesianRecommender = field(factory=SequentialGreedyRecommender)
    """The recommender used for the continuous subspace. Default:
    :class:`baybe.recommenders.pure.bayesian.sequential_greedy.SequentialGreedyRecommender`"""

    def __attrs_post_init__(self):
        """Validate if flags are synchronized and overrides them otherwise."""
        if (
            flag := self.allow_recommending_already_measured
        ) != self.disc_recommender.allow_recommending_already_measured:
            warnings.warn(
                f"The value of "
                f"'{fields(self.__class__).allow_recommending_already_measured.name}' "
                f"differs from what is specified in the discrete recommender. "
                f"The value of the discrete recommender will be ignored."
            )
            self.disc_recommender = evolve(
                self.disc_recommender,
                allow_recommending_already_measured=flag,
            )

        if (
            flag := self.allow_repeated_recommendations
        ) != self.disc_recommender.allow_repeated_recommendations:
            warnings.warn(
                f"The value of "
                f"'{fields(self.__class__).allow_repeated_recommendations.name}' "
                f"differs from what is specified in the discrete recommender. "
                f"The value of the discrete recommender will be ignored."
            )
            self.disc_recommender = evolve(
                self.disc_recommender,
                allow_repeated_recommendations=flag,
            )

    def recommend(  # noqa: D102
        self,
        searchspace: SearchSpace,
        batch_size: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        # See base class.

        # First check whether the disc_recommender is either bayesian or non-predictive
        is_bayesian_recommender = isinstance(self.disc_recommender, BayesianRecommender)
        is_np_recommender = isinstance(self.disc_recommender, NonPredictiveRecommender)

        if (not is_bayesian_recommender) and (not is_np_recommender):
            raise NotImplementedError(
                """The discrete recommender should be either a Bayesian or a
                NonPredictiveRecommender."""
            )

        # Check if the space is a pure continuous or discrete space first and just use
        # the corresponding recommendation function in that case
        degenerate_recommender = None
        if searchspace.type == SearchSpaceType.DISCRETE:
            degenerate_recommender = self.disc_recommender
        elif searchspace.type == SearchSpaceType.CONTINUOUS:
            degenerate_recommender = self.cont_recommender
        if degenerate_recommender is not None:
            return degenerate_recommender.recommend(
                searchspace=searchspace,
                batch_size=batch_size,
                train_x=train_x,
                train_y=train_y,
            )

        # We are in a hybrid setting now

        # We will attach continuous parts to discrete parts and the other way round.
        # To make things simple, we sample a single point in the continuous space which
        # will then be attached to every discrete point when the acquisition function
        # is evaluated.
        cont_part = searchspace.continuous.samples_random(1)
        cont_part = to_tensor(cont_part).unsqueeze(-2)

        # Get discrete candidates. The metadata flags are ignored since the search space
        # is hybrid
        # TODO Slight BOILERPLATE CODE, see recommender.py, ll. 47+
        _, candidates_comp = searchspace.discrete.get_candidates(
            allow_repeated_recommendations=True,
            allow_recommending_already_measured=True,
        )

        # We now check whether the discrete recommender is bayesian.
        if is_bayesian_recommender:
            # Get access to the recommenders acquisition function
            self.disc_recommender.setup_acquisition_function(
                searchspace, train_x, train_y
            )

            # Construct the partial acquisition function that attaches cont_part
            # whenever evaluating the acquisition function
            disc_acqf_part = PartialAcquisitionFunction(
                acqf=self.disc_recommender._acquisition_function,
                pinned_part=cont_part,
                pin_discrete=False,
            )

            self.disc_recommender._acquisition_function = disc_acqf_part

        # Call the private function of the discrete recommender and get the indices
        disc_rec_idx = self.disc_recommender._recommend_discrete(
            subspace_discrete=searchspace.discrete,
            candidates_comp=candidates_comp,
            batch_size=batch_size,
        )

        # Get one random discrete point that will be attached when evaluating the
        # acquisition function in the discrete space.
        disc_part = searchspace.discrete.comp_rep.loc[disc_rec_idx].sample(1)
        disc_part = to_tensor(disc_part).unsqueeze(-2)

        # Setup a fresh acquisition function for the continuous recommender
        self.cont_recommender.setup_acquisition_function(searchspace, train_x, train_y)

        # Construct the continuous space as a standalone space
        cont_acqf_part = PartialAcquisitionFunction(
            acqf=self.cont_recommender._acquisition_function,
            pinned_part=disc_part,
            pin_discrete=True,
        )
        self.cont_recommender._acquisition_function = cont_acqf_part

        # Call the private function of the continuous recommender
        rec_cont = self.cont_recommender._recommend_continuous(
            searchspace.continuous, batch_size
        )

        # Glue the solutions together and return them
        rec_disc_exp = searchspace.discrete.exp_rep.loc[disc_rec_idx]
        rec_cont.index = rec_disc_exp.index
        rec_exp = pd.concat([rec_disc_exp, rec_cont], axis=1)
        return rec_exp
