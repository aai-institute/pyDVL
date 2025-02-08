r"""
!!! Fixme "Move this to the docs"


Owen sampling schemes are a family of sampling schemes that are used to estimate
Shapley values. They are based on a multilinear extension technique from game theory,
and were introduced in (Okhrati and Lipani, 2021)[^1]. The core idea is to use different
probabilities of including indices into samples.

In order to compute values it is enough to use any of the Owen samplers together with a
[ShapleyValuation][pydvl.valuation.methods.ShapleyValuation] object.

!!! Example "Finite Owen Sampler"
    [FiniteOwenSampler][pydvl.valuation.samplers.owen.FiniteOwenSampler] is the most
    basic Owen sampler. It uses a deterministic grid of probability values between 0
    and 1 for the inner sampling. The number of samples drawn is therefore constant and
    equal to `n_samples_outer * n_samples_inner`. It follows the idea of the original
    paper and should be instantiated with
    [NoStopping][pydvl.valuation.stopping.NoStopping] as stopping criterion. Note that
    because the criterion never checks for convergence, the status of the valuation will
    always be `Status.Pending`.

    ```python
    from pydvl.valuation import FiniteOwenSampler, ShapleyValuation, NoStopping
    ...

    sampler = FiniteOwenSampler(n_samples_outer=200, n_samples_inner=4)
    valuation = ShapleyValuation(utility, sampler, NoStopping())
    valuation.fit(dataset)
    shapley_values = valuation.values()
    ```

!!! Example "Owen Sampler"
    [OwenSampler][pydvl.valuation.samplers.owen.OwenSampler] follows the same principle
    as [FiniteOwenSampler][pydvl.valuation.samplers.owen.FiniteOwenSampler], but samples
    probability values between 0 and 1 at random indefinitely. It requires a stopping
    criterion to be used with the valuation method, and thus follows more closely the
    general pattern of the valuation methods. This makes it more adequate for actual use
    since it is no longer required to estimate a number of outer samples required.

    ```python
    from pydvl.valuation import OwenSampler, ShapleyValuation, RankCorrelation
    ...

    sampler = OwenSampler()
    valuation = ShapleyValuation(utility, sampler, RankCorrelation(rtol=1e-3))
    valuation.fit(dataset)
    ```

!!! Example "Antithetic Owen Sampler"
    [AntitheticOwenSampler][pydvl.valuation.samplers.owen.AntitheticOwenSampler] is a
    variant of the [OwenSampler][pydvl.valuation.samplers.owen.OwenSampler] that draws
    probability values $q$ between 0 and 0.5 at random and then generates two samples
    for each index, one using the probability $q$ for index draws, and another with
    probability $1-q$.

    ```python
    from pydvl.valuation import AntitheticOwenSampler, ShapleyValuation, RankCorrelation
    ...

    sampler = AntitheticOwenSampler()
    valuation = ShapleyValuation(utility, sampler, RankCorrelation(rtol=1e-3))
    valuation.fit(dataset)
    ```


## References

[^1]: <a name="okhrati_multilinear_2021"></a>Okhrati, R., Lipani, A., 2021.
    [A Multilinear Sampling Algorithm to Estimate Shapley
    Values](https://ieeexplore.ieee.org/abstract/document/9412511). In: 2020 25th
    International Conference on Pattern Recognition (ICPR), pp. 7992–7999. IEEE.
"""
