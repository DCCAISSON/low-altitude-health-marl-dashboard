from __future__ import annotations

from dataclasses import dataclass
from math import exp, log


EPS = 1e-9


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _normalize(values: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    cleaned = [max(float(value), 0.0) for value in values]
    total = sum(cleaned)
    if total <= EPS:
        size = len(cleaned)
        return tuple(1.0 / size for _ in range(size))
    return tuple(value / total for value in cleaned)


def _softmax(scores: list[float]) -> tuple[float, ...]:
    anchor = max(scores)
    exps = [exp(score - anchor) for score in scores]
    total = sum(exps)
    return tuple(value / total for value in exps)


@dataclass(slots=True)
class MarkovConfig:
    state_names: tuple[str, ...]
    state_names_cn: tuple[str, ...]
    state_weights: tuple[float, ...]
    base_transition: tuple[tuple[float, ...], ...]
    service_effect: float
    unmet_need_effect: float
    shock_effect: float
    trust_effect: float


class HealthMarkovModel:
    def __init__(self, config: MarkovConfig) -> None:
        self.config = config
        self.n_states = len(config.state_weights)
        self._validate()
        self.state_weights = tuple(float(value) for value in config.state_weights)
        self.base_transition = tuple(_normalize(row) for row in config.base_transition)

    def _validate(self) -> None:
        if self.n_states < 2:
            raise ValueError("Markov states must contain at least two states")
        if len(self.config.state_names) != self.n_states:
            raise ValueError("state_names length does not match state_weights")
        if len(self.config.state_names_cn) != self.n_states:
            raise ValueError("state_names_cn length does not match state_weights")
        if tuple(sorted(self.config.state_weights, reverse=True)) != tuple(self.config.state_weights):
            raise ValueError("state_weights should be ordered from healthy to severe")
        if len(self.config.base_transition) != self.n_states:
            raise ValueError("base_transition row count does not match state size")
        for row in self.config.base_transition:
            if len(row) != self.n_states:
                raise ValueError("base_transition must be a square matrix")

    def health_from_distribution(self, distribution: tuple[float, ...] | list[float]) -> float:
        dist = _normalize(distribution)
        return sum(weight * share for weight, share in zip(self.state_weights, dist))

    def distribution_from_health(self, health: float) -> tuple[float, ...]:
        health = _clip(float(health), self.state_weights[-1], self.state_weights[0])
        if health >= self.state_weights[0] - EPS:
            return tuple(1.0 if idx == 0 else 0.0 for idx in range(self.n_states))
        if health <= self.state_weights[-1] + EPS:
            return tuple(1.0 if idx == self.n_states - 1 else 0.0 for idx in range(self.n_states))
        distribution = [0.0] * self.n_states
        for idx in range(self.n_states - 1):
            upper = self.state_weights[idx]
            lower = self.state_weights[idx + 1]
            if upper >= health >= lower:
                if abs(upper - lower) <= EPS:
                    distribution[idx] = 1.0
                else:
                    upper_share = (health - lower) / (upper - lower)
                    distribution[idx] = upper_share
                    distribution[idx + 1] = 1.0 - upper_share
                return tuple(distribution)
        distribution[-1] = 1.0
        return tuple(distribution)

    def average_distribution(
        self,
        distributions: tuple[tuple[float, ...], ...] | list[tuple[float, ...]],
        weights: tuple[float, ...] | list[float],
    ) -> tuple[float, ...]:
        weighted = [0.0] * self.n_states
        total = 0.0
        for distribution, weight in zip(distributions, weights):
            total += float(weight)
            dist = _normalize(distribution)
            for idx, share in enumerate(dist):
                weighted[idx] += share * float(weight)
        if total <= EPS:
            return tuple(1.0 / self.n_states for _ in range(self.n_states))
        return tuple(value / total for value in weighted)

    def _adjust_row(
        self,
        from_idx: int,
        service_signal: float,
        unmet_signal: float,
        shock_signal: float,
        trust_signal: float,
    ) -> tuple[float, ...]:
        service_push = self.config.service_effect * max(service_signal, 0.0)
        service_push += self.config.trust_effect * max(trust_signal, 0.0)
        deterioration_push = self.config.unmet_need_effect * max(unmet_signal, 0.0)
        deterioration_push += self.config.shock_effect * max(shock_signal, 0.0)
        scores = []
        for to_idx, base_prob in enumerate(self.base_transition[from_idx]):
            score = log(max(base_prob, EPS))
            if to_idx < from_idx:
                score += service_push * (from_idx - to_idx)
            elif to_idx > from_idx:
                score += deterioration_push * (to_idx - from_idx)
            scores.append(score)
        return _softmax(scores)

    def transition_distribution(
        self,
        current_distribution: tuple[float, ...] | list[float],
        service_signal: float,
        unmet_signal: float,
        shock_signal: float,
        trust_signal: float,
    ) -> tuple[float, ...]:
        current = _normalize(current_distribution)
        adjusted_rows = [
            self._adjust_row(
                from_idx=idx,
                service_signal=service_signal,
                unmet_signal=unmet_signal,
                shock_signal=shock_signal,
                trust_signal=trust_signal,
            )
            for idx in range(self.n_states)
        ]
        next_distribution = [0.0] * self.n_states
        for from_idx, mass in enumerate(current):
            for to_idx, transition_prob in enumerate(adjusted_rows[from_idx]):
                next_distribution[to_idx] += mass * transition_prob
        return _normalize(next_distribution)

    def state_share_pairs(self, distribution: tuple[float, ...] | list[float]) -> tuple[tuple[str, float], ...]:
        dist = _normalize(distribution)
        return tuple((name, share) for name, share in zip(self.config.state_names_cn, dist))
