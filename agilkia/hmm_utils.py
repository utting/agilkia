# -*- coding:utf8
import random
import subprocess
import hmmlearn.hmm
import platform
import os
import json
import numpy as np
from . utils import *
from typing import List, NoReturn, NewType, Optional, Tuple


Probability = float
LogProbability = float
InternalSymbol = NewType('InternalSymbol', int)
ExternalSymbol = NewType('ExternalSymbol', str)
State = NewType('State', int)


def makeRandomVector(l: int):
    """
    Create a vector of length l with pseudo-random values between 0 and 1. The sum of all values in the vector is 1.
    """
    v = []
    sum = 0
    for i in range(l - 1):
        p = random.random() * 2 * (1 - sum) / (l - i)
        v.append(p)
        sum += p
    v.append(1 - sum)
    return v


class HMM:
    """
    A base class to manipulate HMM. This class can be implemented by using an existing HMM library.
    """

    def train(self, sequences: List[List[ExternalSymbol]]) -> NoReturn:
        """
        Refine the model to fit best to the given sequences
        """
        raise NotImplementedError

    def getMostProbableStates(self, sequences: List[List[ExternalSymbol]]) -> List[List[State]]:
        """
        get the most probable path in this model for the given sequences
        """
        raise NotImplementedError

    def statesNumber(self) -> int:
        """
        get the number of states of this model.
        """
        raise NotImplementedError

    def symbolsNumber(self) -> int:
        """
        get the number of symbols of this model.
        """
        raise NotImplementedError

    def getSymbol(self, i: InternalSymbol) -> ExternalSymbol:
        """
        get symbol at position i in the model.
        """
        raise NotImplementedError

    def getSymbols(self) -> List[ExternalSymbol]:
        """
        get the list of symbols, in the same order than in the model.
        """
        raise NotImplementedError

    def getEmission(self, state: State, symbol: InternalSymbol) -> Probability:
        """
        get the emission probability of the given symbol from the given state.
        """
        raise NotImplementedError

    def getInitial(self, state: State) -> Probability:
        """
        get the probability of starting from the given state.
        """
        raise NotImplementedError

    def getTransition(self, state_i: State, state_j: State) -> Probability:
        """
        get the probability of going to `state_j` when current state is `state_i`.
        """
        raise NotImplementedError

    def setInitial(self, state: State, p: Probability) -> NoReturn:
        """
        set the probability of starting from the given state.
        It is the user responsibility to modify others probability to keep the sum equals to 1.
        """
        raise NotImplementedError

    def setTransition(self, state_i: State, state_j: State, p: Probability) -> NoReturn:
        """
        set the probability of going to state `state_j` when current state is `state_i`.
        It is the user responsibility to modify others probability to keep the sum equals to 1.
        """
        raise NotImplementedError

    def setEmission(self, state: State, symbol: InternalSymbol, p: Probability) -> NoReturn:
        """
        set the probability of emitting the given symbol when being in given state.
        It is the user responsibility to modify others probability to keep the sum equals to 1.
        """
        raise NotImplementedError

    def logLikelihoods(self, sequences: List[List[ExternalSymbol]]) -> List[LogProbability]:
        """
        get the log of the probability of each sequence to be emitted by this model.
        """
        raise NotImplementedError

    def _copy(self) -> 'HMM':
        """
        this method is used by the method `copy` to create the new model.
        The copy of the probability will be handled later by `copy()`, but the returned model must have the same symbols in the same order.
        """
        raise NotImplementedError

    def copy(self) -> 'HMM':
        """
        create a copy of this model.
        """
        other = self._copy()
        assert self.getSymbols() == other.getSymbols()
        for i in range(self.statesNumber()):
            other.setInitial(i, self.getInitial(i))
            for j in range(self.statesNumber()):
                other.setTransition(i, j, self.getTransition(i, j))
            for sym in range(self.symbolsNumber()):
                other.setEmission(i, sym, self.getEmission(i, sym))
        return other

    def save(self, fileName: str) -> NoReturn:
        """
        save this model into a file.
        """
        A = []
        B = []
        pi = []
        for i in range(self.statesNumber()):
            a = []
            for j in range(self.statesNumber()):
                a.append(self.getTransition(i, j))
            A.append(a)
            b = []
            for sym in range(self.symbolsNumber()):
                b.append(self.getEmission(i, sym))
            B.append(b)
            pi.append(self.getInitial(i))
        with open(fileName, "w") as f:
            json.dump({"A": A, "B": B, "pi": pi,
                       "symbols": self.getSymbols()}, f)

    @staticmethod
    def load(fileName: str) -> 'HMM':
        """
        create a new model from a file.
        """
        dict_ = None
        with open(fileName, "r")as f:
            dict_ = json.load(f)
        A = dict_["A"]
        B = dict_["B"]
        pi = dict_["pi"]
        symbols = dict_["symbols"]
        model = HMM.createModel(len(A), symbols)
        for i in range(len(A)):
            a = A[i]
            for j in range(len(A)):
                model.setTransition(i, j, a[j])
            b = B[i]
            for sym in range(len(b)):
                model.setEmission(i, sym, b[sym])
            model.setInitial(i, pi[i])
        return model

    def average(self, other: 'HMM', weight: float) -> NoReturn:
        """
        balance probabilities of this model with probabilities of `other` model with given weight.
        A weight of 0 means no change, a weight of 1 means copy the other.
        The two models are assumed to have the same size and the same symbols.
        """
        assert self.symbolsNumber() == other.symbolsNumber()
        assert self.statesNumber() == other.statesNumber()
        selfWeight = 1 - weight
        for i in range(self.statesNumber()):
            self.setInitial(i, self.getInitial(i) * selfWeight +
                            other.getInitial(i) * weight)
            for j in range(self.statesNumber()):
                self.setTransition(i, j, self.getTransition(
                    i, j) * selfWeight + other.getTransition(i, j) * weight)
            for sym in range(self.symbolsNumber()):
                self.setEmission(i, sym, self.getEmission(
                    i, sym) * selfWeight + other.getEmission(i, sym) * weight)

    def randomize(self, noise: float) -> NoReturn:
        """
        Add noise in the probabilities of this model. Useful to get away of null probabilities…
        """
        other = HMM.random(self.statesNumber(), self.getSymbols())
        self.average(other, weight=noise)

    def forceGaucheDroite(self, up_to: int = None) -> NoReturn:
        """remove return paths for the first states of the model

        :param up_to: the number of states to which should have a single direction transitions
        """
        N = self.statesNumber()
        for i in range(N):
            s = 0
            max = i
            if up_to is not None and i > up_to:
                max = up_to
            for j in range(N):
                if j < max:
                    s += self.getTransition(i, j)
                    self.setTransition(i, j, 0)
                else:
                    d = s / (N - j)
                    self.setTransition(i, j, d + self.getTransition(i, j))
                    s -= d
            assert s < 0.01

    def isNormalized(self) -> bool:
        """
        Indicate whether the probabilities are corrects and sums to 1 where they must.
        """
        normalized = True
        s = 0
        for i in range(self.statesNumber()):
            s += self.getInitial(i)
        if not (s < 1.01 and s > 0.99):
            normalized = False
            print("initial probabilities are not correct")
        for i in range(self.statesNumber()):
            s = 0
            for j in range(self.statesNumber()):
                s += self.getTransition(i, j)
            if not (s < 1.01 and s > 0.99):
                normalized = False
                print("Transitions are not normalized for state {}.".format(i))
            s = 0
            for sym in range(self.symbolsNumber()):
                s += self.getEmission(i, sym)
            if not (s < 1.01 and s > 0.99):
                normalized = False
                print("Emissions are not normalized for state {}.".format(i))
        return normalized

    @staticmethod
    def createModel(states, symbols: List[ExternalSymbol] = None) -> 'HMM':
        """
        create an empty model. Currently it instantiate a HMM based on hmmlearn library.
        """
        return HMM_hmmlearn(states, symbols)

    @staticmethod
    def random(states, symbols: List[ExternalSymbol]) -> 'HMM':
        """build a random HMM

        :param states: the number of states to create
        :param symbols: the symbols used by the model
        :returns: the HMM models as a tuple of matrice (A,B,pi)
        """
        assert symbols is not None
        model = HMM.createModel(states, symbols=symbols)
        init = makeRandomVector(states)
        for i in range(states):
            model.setInitial(i, init[i])
            t = makeRandomVector(states)
            for j in range(states):
                model.setTransition(i, j, t[j])
            e = makeRandomVector(len(symbols))
            for sym in range(len(symbols)):
                model.setEmission(i, sym, e[sym])
        return model

    @staticmethod
    def assembleModels(models: List['HMM']) -> 'HMM':
        """
        Groups several models into one big model. The grouping enable to have initial probabilities which balance the use of each individual model.
        """
        assert len(models) > 0
        symbols = models[0].getSymbols()
        states = 0
        for model in models:
            states += model.statesNumber()
            assert model.getSymbols() == symbols
        newModel = HMM.createModel(states, symbols)
        pos = 0
        for model in models:
            for i in range(model.statesNumber()):
                for j in range(states):
                    t = 0
                    if j >= pos and j < pos + model.statesNumber():
                        t = model.getTransition(i, j - pos)
                    newModel.setTransition(i + pos, j, t)
                for sym in range(len(symbols)):
                    newModel.setEmission(
                        i + pos, sym, model.getEmission(i, sym))
                newModel.setInitial(
                    i + pos, model.getInitial(i) * 1. / len(models))
            pos += model.statesNumber()
        return newModel


class HMM_hmmlearn(HMM):
    def __init__(self, states: int, symbols: Optional[List[ExternalSymbol]] = None):
        self._model = hmmlearn.hmm.MultinomialHMM(n_components=states)
        self._symbolsDict = symbols
        if symbols is not None:
            X = hmmlearn.base.check_array([[i] for i in range(len(symbols))])
            self._model._init(X)

    def _assembleSequences(self, traces: List[List[ExternalSymbol]],
                           addSymbols: bool = False) -> Tuple[List[List[InternalSymbol]], List[int]]:
        """
        for library hmmlearn, assemble a sequence of sequences into only one sequence and a list of length
        """
        if addSymbols:
            assert self._symbolsDict is None
            self._symbolsDict = []
        seq = []
        lengths = []
        for trace in traces:
            l = len(seq)
            for event in trace:
                p = None
                try:
                    p = self._symbolsDict.index(event)
                except ValueError:
                    assert addSymbols
                    p = len(self._symbolsDict)
                    self._symbolsDict.append(event)
                seq.append([p])
            lengths.append(len(seq) - l)
        return seq, lengths

    def _disassemble(self, seq: List[State], lengths: List[int]) -> List[List[State]]:
        seqs = []
        pos = 0
        for length in lengths:
            seqs.append(seq[pos:pos + length])
            pos += length
        return seqs

    def _permuteSymbols(self, sym1: InternalSymbol, sym2: InternalSymbol,
                        seq: Optional[List[List[InternalSymbol]]] = None) -> Optional[List[List[InternalSymbol]]]:
        """
        hmmlearn does not train a model if the symbols of the sequence are not in a simple sequence
        This method can be used to change the symbols order
        """
        for i in range(self.statesNumber()):
            t = self.getEmission(i, sym1)
            self.setEmission(i, sym1, self.getEmission(i, sym2))
            self.setEmission(i, sym2, t)
        t = self._symbolsDict[sym1]
        self._symbolsDict[sym1] = self._symbolsDict[sym2]
        self._symbolsDict[sym2] = t
        if seq is not None:
            newSeq = []
            for e in seq:
                e = e[0]
                if e == sym1:
                    e = sym2
                elif e == sym2:
                    e = sym1
                newSeq.append([e])
            return newSeq

    def train(self, sequences: List[List[ExternalSymbol]]) -> NoReturn:
        assert len(sequences) > 0
        if hasattr(self._model, 'n_features'):
            self._model.init_params = ''
        seq, lengths = self._assembleSequences(
            sequences, addSymbols=(self._symbolsDict is None))
        permut = []
        symbols = set()
        symbols.update(map(lambda x: x[0], seq))
        symbols = sorted(symbols)
        for i in range(len(symbols)):
            sym = symbols[i]
            if sym != i:
                permut.append((i, sym))
                seq = self._permuteSymbols(i, sym, seq)
        if len(seq) > 1:
            self._model.fit(seq, lengths)
        elif len(seq) == 1:
            # hmmlearn does not support training a model with only one symbol
            sym = seq[0][0]
            for i in range(self.statesNumber()):
                for j in range(self.statesNumber()):
                    self.setTransition(i, j, 0)
                self.setTransition(i, i, 1)
                for j in range(self.symbolsNumber()):
                    self.setEmission(i, j, 0)
                self.setEmission(i, sym, 1)
                self.setInitial(i, 0)
            self.setInitial(0, 1)  # state 0 will be the initial state
        else:
            # no data to train the model, do nothing
            pass
        for i in range(self.statesNumber()):
            row = self._model.transmat_[i]
            if row.sum() == 0:
                print(
                    "warning : state with no output transition ! Adding a loop transition")
                row[i] = 1
            if not (row.sum() > 0.99 and row.sum() < 1.01):
                print(
                    "warning :  state with invalid transitions (sum={}) ! Using one loop transition.".format(row.sum()))
                for j in range(self.statesNumber()):
                    row[j] = 0
                row[i] = 1.

            emissions = self._model.emissionprob_[i]
            if not (emissions.sum() > 0.99 and emissions.sum() < 1.01):
                print("warning : invalid emissions (sum={}) ! Using homogeneous emissions.".format(emissions.sum()))
                for j in range(self.symbolsNumber()):
                    emissions[j] = 1. / self.symbolsNumber()
        if not (self._model.startprob_.sum() >
                0.99 and self._model.startprob_.sum() < 1.01):
            print(
                "warning : model has invalid initial probabilities (sum={}). Using homogeneous initial probability.".format(
                    self._model.startprob_.sum()))
            for i in range(self.statesNumber()):
                self._model.startprob_[i] = 1. / self.statesNumber()
        permut.reverse()
        for (i, j)in permut:
            self._permuteSymbols(i, j)

    def getMostProbableStates(self, sequences: List[List[ExternalSymbol]]) -> List[List[State]]:
        if len(sequences) == 0:
            # hmmlearn decode() do not support empty sequence
            return []
        seq, lengths = self._assembleSequences(sequences)
        prob, states = self._model.decode(seq, lengths)
        return self._disassemble(states, lengths)

    def statesNumber(self) -> int:
        return self._model.n_components

    def symbolsNumber(self) -> int:
        return self._model.n_features

    def getSymbol(self, i: InternalSymbol) -> ExternalSymbol:
        return self._symbolsDict[i]

    def getSymbols(self) -> List[ExternalSymbol]:
        return self._symbolsDict

    def getEmission(self, state: State, symbol: InternalSymbol) -> Probability:
        return self._model.emissionprob_[state][symbol]

    def getInitial(self, state: State) -> Probability:
        return self._model.startprob_[state]

    def getTransition(self, state_i: State, state_j: State) -> Probability:
        return self._model.transmat_[state_i][state_j]

    def setInitial(self, state: State, p: Probability) -> NoReturn:
        self._model.startprob_[state] = p

    def setTransition(self, state_i: State, state_j: State, p: Probability) -> NoReturn:
        self._model.transmat_[state_i][state_j] = p

    def setEmission(self, state: State, symbol: InternalSymbol, p: Probability) -> NoReturn:
        self._model.emissionprob_[state][symbol] = p

    def logLikelihoods(self, sequences: List[List[ExternalSymbol]]) -> List[LogProbability]:
        ll = []
        for sequence in sequences:
            seq, length = self._assembleSequences([sequence])
            ll.append(self._model.score(seq, length))
        return ll

    def _copy(self) -> 'HMM_hmmlearn':
        other = HMM_hmmlearn(self.statesNumber(), self.getSymbols())
        return other


def simplify(
        hmm: HMM,
        remove: bool = False,
        emissionThresold: Probability = 0.1,
        reachedTolerance: float = 0.1,
        outputTransThresold: Probability = 0.1,
        reachedThresold: Probability = 0.001) -> Optional[HMM]:
    """ try to simplify a model by merging similar states.


    Merging is alowed when :
     - two states have proportional incoming transitions (including initial) and equivalent emissions
     - two states have proportional incoming transitions (including initial) and equivalent output transitions
     - two states have equivalent emissions and equivalent outputs transitions

    :param hmm: the Markov model to simplify
    :param emissionThresold: the maximum difference of probability of emission for two states seen as similar
    :type emissionThresold: float[0:1]
    :param reachedTolerance: a ratio to declare two state as «reachable with the same probability». Used with reachedThresold.
    :type reachedTolerance: float [0:infinity]
    :param outputTransThresold: the maximum probability of output transition for two states seen as similar
    :type outputTransThresold: float[0:1]
    :type reachedThresold: float[0:1]
    :return a model with merged states removed if `remove` is set to `True` (There is currently no method to remove state from a model, we need to build a new one)
    """

    assert isinstance(hmm, HMM)
    N = hmm.statesNumber()
    M = hmm.symbolsNumber()
    """merge state j into state i
    """
    def merge(i, j):
        sum_i = 0
        sum_j = 0
        for k in range(N):
            sum_i += hmm.getTransition(k, i)
            sum_j += hmm.getTransition(k, j)
        sum_i += hmm.getInitial(i)
        sum_j += hmm.getInitial(j)
        if sum_i == 0 and sum_j == 0:
            # the states are unreachable. Thus, we do not care about how to
            # process them but we should avoid null division
            sum_i = 1
            sum_j = 1
        prop_i = sum_i / (sum_i + sum_j)
        prop_j = sum_j / (sum_i + sum_j)

        hmm.setInitial(i, hmm.getInitial(i) + hmm.getInitial(j))
        hmm.setInitial(j, 0)
        for k in range(N):
            if k == i or k == j:
                continue
            p = hmm.getTransition(k, i) + hmm.getTransition(k, j)
            if p > 1:
                p = 1
            hmm.setTransition(k, i, p)
            hmm.setTransition(k, j, 0)
        for k in range(N):
            if k == i or k == j:
                continue
            hmm.setTransition(
                i,
                k,
                hmm.getTransition(
                    i,
                    k) *
                prop_i +
                hmm.getTransition(
                    j,
                    k) *
                prop_j)
            hmm.setTransition(j, k, 0)
        hmm.setTransition(i, i, (hmm.getTransition(i, i) + hmm.getTransition(i, j)) *
                          prop_i + (hmm.getTransition(j, j) + hmm.getTransition(j, i)) * prop_j)
        hmm.setTransition(i, j, 0)
        hmm.setTransition(j, i, 0)
        hmm.setTransition(j, j, 1)

        for s in range(M):
            hmm.setEmission(i, s, hmm.getEmission(i, s) *
                            prop_i + hmm.getEmission(j, s) * prop_j)

    def areProportional(v1, v2, tolerance=0.1, thresold=0.001):
        sum1 = 0
        sum2 = 0
        assert len(v1) == len(v2)
        for k in range(len(v1)):
            assert v1[k] >= 0 and v2[k] >= 0
            sum1 += v1[k]
            sum2 += v2[k]
        for k in range(len(v1)):
            p1 = v1[k] * sum2
            p2 = v2[k] * sum1
            if p1 * (1 + tolerance) + thresold < p2 or p2 * \
                    (1 + tolerance) + thresold < p1:
                return False
        return True

    def areReachedSimilarly(i, j):
        """indicate whether the probabilities of reaching a state from any sequence
        is proportional to the probability of reaching the other state from the
        same sequences
        """
        v_i = []
        v_j = []
        for k in range(N):
            v_i.append(hmm.getTransition(k, i))
            v_j.append(hmm.getTransition(k, j))
        v_i.append(hmm.getInitial(i))
        v_j.append(hmm.getInitial(j))
        return areProportional(v_i, v_j, reachedTolerance, reachedThresold)

    def haveSimilarEmission(i, j):
        """indicate whether two states have similar emissions"""
        for s in range(M):
            diff = hmm.getEmission(i, s) - hmm.getEmission(j, s)
            if diff > emissionThresold or diff < - emissionThresold:
                return False
        return True

    def haveSimilarOutputTransition(i, j):
        for k in range(N):
            p_i = hmm.getTransition(i, k)
            p_j = hmm.getTransition(j, k)
            diff = p_i - p_j
            if diff > outputTransThresold or diff < -outputTransThresold:
                return False
        return True

    merged = set()
    mustUpdate = True
    while mustUpdate:
        mustUpdate = False
        for j in range(N):
            if j in merged:
                continue
            for i in range(j):
                mustMerge = False
                if areReachedSimilarly(i, j):
                    if haveSimilarEmission(
                            i,
                            j) or haveSimilarOutputTransition(
                            i,
                            j):
                        mustMerge = True
                elif haveSimilarEmission(i, j) and haveSimilarOutputTransition(i, j):
                    mustMerge = True
                if mustMerge:
                    merge(i, j)
                    merged.add(j)
                    mustUpdate = True
    if remove:
        pos = sorted(set(range(N)).difference(merged))
        newStates = len(pos)
        newModel = HMM.createModel(newStates, hmm.getSymbols())
        for i in range(newStates):
            for j in range(newStates):
                newModel.setTransition(i, j, hmm.getTransition(pos[i], pos[j]))
            for sym in range(len(hmm.getSymbols())):
                newModel.setEmission(i, sym, hmm.getEmission(pos[i], sym))
            newModel.setInitial(i, hmm.getInitial(pos[i]))
        return newModel


class dotTransition:
    def __init__(
            self,
            fromS,
            toS,
            label=None,
            color: Optional[Color] = None,
            fontColor: Optional[Color] = None,
            fromInitial=False):
        self.fromS = fromS
        self.fromInitial = fromInitial
        self.toS = toS
        self.label = label
        self.color = color
        self.fontColor = fontColor


class Graph:
    def __init__(self, hmm):
        self.hmm = hmm
        self.transitionThresold = 0.05
        self.emissionThresold = 0.01
        self.initialThresold = 0.01
        self.pFormat = lambda p: "{:.0f}%".format(p * 100)
        self.extTransitions = []

    def showSequence(self, seq, color: Optional[Color] = None):
        """show in the graph the most likely transitions used by a sequence
        """
        self.showSequences([seq], color)

    def showSequences(self, seqs, color: Optional[Color] = None):
        """
        show a group of sequences on the graph
        """
        viterbi = self.hmm.getMostProbableStates(seqs)
        transitions = {}
        initials = {}
        max_t = 0
        invalid = 0
        for seq in viterbi:
            if seq[0] == -1:
                invalid += 1
                continue
            if seq[0] not in initials.keys():
                initials[seq[0]] = 0
            initials[seq[0]] += 1
            max_t = max(max_t, initials[seq[0]])
            for t in range(len(seq) - 1):
                from_ = seq[t]
                to_ = seq[t + 1]
                if from_ not in transitions:
                    transitions[from_] = {}
                if to_ not in transitions[from_]:
                    transitions[from_][to_] = 0
                transitions[from_][to_] += 1
                max_t = max(max_t, transitions[from_][to_])

        def makeColor(n: int, total: int) -> Color:
            if color is not None and color.alpha is None:
                return Color(color.r, color.g, color.b, int(50 + 205 * n / total))
            return color

        if invalid > 0:
            self.extTransitions.append(
                dotTransition(
                    None,
                    None,
                    color=color,
                    label="{} sequences cannot be build by this model".format(invalid)))
        for from_ in transitions.keys():
            for to in transitions[from_].keys():
                self.extTransitions.append(
                    dotTransition(
                        from_,
                        to,
                        color=makeColor(transitions[from_][to], max_t),
                        label=" {}".format(
                            transitions[from_][to]),
                        fontColor=color))
        for init in initials.keys():
            self.extTransitions.append(
                dotTransition(
                    init,
                    init,
                    color=makeColor(
                        initials[init],
                        len(seqs)),
                    label=" {}".format(
                        initials[init]),
                    fromInitial=True))

    def exportSingle(self, filename):
        dotFile = DotFile(filename)
        dotFile.addGraph(self)
        dotFile.export()

    def toDot(self, dotStream, graphID):
        """ a method which should be called by a DotFile object
        """
        def nodeName(i):
            return "node_{}_{}".format(i, graphID)
        initialsCreated = set()
        initialsTransitions = set()

        def initialNodeName(i, addTransition=False):
            name = "init_{}_{}".format(i, graphID)
            if i not in initialsCreated:
                dotStream.write(
                    "{} [label=\"initial\",fontcolor=blue,shape=none];\n".format(name))
                initialsCreated.add(i)
            if addTransition and i not in initialsTransitions:
                dotStream.write(
                    ("{} -> {} [label=\"p={}\",color=blue,fontcolor=blue];\n").format(
                        name, nodeName(i), self.pFormat(
                            hmm.getInitial(i))))
                initialsTransitions.add(i)
            return name
        hmm = self.hmm
        N = hmm.statesNumber()
        M = hmm.symbolsNumber()
        for i in range(N):
            label = "{}\n".format(i)

            # following loop works only for discrete values
            emissions = []
            for s in range(M):
                emissions.append(hmm.getEmission(i, s))

            for (s, p) in sorted(map(lambda x: (x, emissions[x]), range(
                    M)), key=lambda s_p: int(s_p[1] * -10000)):
                if p > self.emissionThresold:
                    label += ("'{}':{}\n").format(
                        hmm.getSymbol(s), self.pFormat(p))

            dotStream.write(nodeName(i) + " [label=\"" + label + "\"];\n")
            p = hmm.getInitial(i)
            if p > self.initialThresold:
                initialNodeName(i, addTransition=True)
        for i in range(N):
            for j in range(N):
                p = hmm.getTransition(i, j)
                if (p > self.transitionThresold):
                    dotStream.write(
                        ("{} -> {} [label=\"p={}\"];\n").format(
                            nodeName(i),
                            nodeName(j),
                            self.pFormat(p)))
        self.extNodesIds = 0

        def makeExternalNode():
            name = "external_{}_{}".format(self.extNodesIds, graphID)
            self.extNodesIds += 1
            dotStream.write("{} [label=\"\",shape=none];\n".format(name))
            return name
        for t in self.extTransitions:
            from_ = ""
            if t.fromS is None:
                from_ = makeExternalNode()
            elif t.fromInitial:
                from_ = initialNodeName(t.fromS, addTransition=True)
            else:
                from_ = nodeName(t.fromS)
            to_ = ""
            if t.toS is None:
                to_ = makeExternalNode()
            else:
                to_ = nodeName(t.toS)
            attributes = ""
            if t.label is not None:
                attributes += "label=\"" + t.label + "\","
            if t.color is not None:
                attributes += "color=\"" + t.color.toHex() + "\","
                if t.fontColor is None:
                    attributes += "fontcolor=\"" + t.color.toHex() + "\","
            if t.fontColor is not None:
                attributes += "fontcolor=\"" + t.fontColor.toHex() + "\","

            dotStream.write("{} -> {} [{}];\n".format(from_, to_, attributes))


class DotFile:
    def __init__(self, filename):
        self.fileName = filename
        self.graphs = []

    def addGraph(self, graph):
        """add a graph to be plotted in the dot file
        """
        self.graphs.append(graph)

    def export(self):
        """effectively create and fill the dot file, and convert it to svg if possible
        """
        with open(self.fileName, 'w') as f:
            f.write("digraph HMM{\nbgcolor=\"transparent\";\n ")
            ids = 0
            for graph in self.graphs:
                graph.toDot(f, ids)
                ids += 1
            f.write("}\n")
        imageFormat = "svg"
        assert imageFormat in ["svg", "png"]
        formatOption = "-T" + imageFormat
        imageFile = self.fileName[:-4] + "." + imageFormat
        if platform.system() == "Windows":
            os.environ["PATH"] += os.pathsep + \
                'C:/Program Files (x86)/Graphviz2.38/bin/'
            os.system('dot {} {} -o {}'.format(formatOption,
                                               self.fileName, imageFile))

        else:
            with open(imageFile, 'wb') as f:
                f.write(subprocess.check_output(
                    ["dot", formatOption, self.fileName]))


class ClusterAlgo:
    def fit(self, traces: 'TraceSet'):
        raise NotImplementedError()

    def generate(self) -> 'Trace':
        raise NotImplementedError()


class HMM_ClusterAlgo(ClusterAlgo):

    def __init__(self, K: int = 2, states: int = 5):
        """
        Create a new model.

        Args:
            K: the wanted number of clusters.
                The real number of cluster can be higher if one sub-model has several initial states,
                or it can be lower if one sub-model is not used.
            states: the number of states in each sub-model.
        """
        self.K = K
        self.states = states
        self._symbols = None
        self._models = None
        self._bigModel = None
        self.eventToSymbol = lambda event: event.action + str(event.outputs.get('Status',''))

    def tracesSetToSymbolSeq(self, traces: 'TraceSet') -> List[List[str]]:
        seqs = []
        for trace in traces:
            seq = []
            for event in trace:
                seq.append(self.eventToSymbol(event))
            seqs.append(seq)
        return seqs

    def getSymbols(self, traces: 'TraceSet') -> List[str]:
        symbols = set()
        for seq in traces:
            for event in seq:
                symbols.add(self.eventToSymbol(event))
        return list(symbols)

    def fit(self, traces):
        self._bigModel = None
        # statesToCluster is a mapping from states of bigModel to cluster. A trace
        # will be assigned to the cluster corresponding to the first state used in
        # the model by this trace.
        self._statesToCluster = {}
        self._fitSequences = traces
        self._symbols = self.getSymbols(self._fitSequences)
        fitSeqs = self.tracesSetToSymbolSeq(traces)
        symbols = set()
        for seq in fitSeqs:
            for symbol in seq:
                symbols.add(symbol)
        self._symbols = list(symbols)

        self._models = []
        for i in range(self.K):
            self._models.append(HMM.random(self.states, self._symbols))

        nb_randomize = 0
        prevLabels = np.zeros(shape=(len(fitSeqs),))
        self._fitLabels = np.zeros(shape=(len(fitSeqs),))
        iteration = 0
        while iteration < nb_randomize or not (prevLabels == self._fitLabels).all() or iteration == 0:
            prevLabels = self._fitLabels
            if iteration < nb_randomize:
                for m in self._models:
                    m.randomize(
                        noise=0.5 * (nb_randomize - iteration) / nb_randomize)

            seqsByCluster, self._fitLabels, farest = self._predictOnModels(
                fitSeqs)
            for i in range(len(self._models)):
                model = self._models[i]
                seqs = seqsByCluster[i]
                if seqs == []:
                    _, index = farest.pick()
                    seqs = [fitSeqs[index]]
                    prevModelIndex = self._fitLabels[index]
                    # by taking previous model, we ensure convergence
                    model = self._models[prevModelIndex].copy()
                    self._models[i] = model
                    # the randomize() call kill the proof of convergence, but we hope it will
                    # help to produce better models
                    model.randomize(noise=0.01)
                model.train(seqs)

            iteration += 1
            print("end of iteration {}".format(iteration))
        self._bigModel = HMM.assembleModels(self._models)
        self._bigModel.train(fitSeqs)

    def _predictOnModels(self, sequences: List[List[str]]) -> (List[List[List[str]]], List[int], MinList):
        """split sequences depending on which model has the highest probability of emitting it.

        Returns:
            + a list of size `len(self._models)` containing the lists of sequences assigned to each model.
            + a list of size `len(sequences)` indicating to which model each sequence is assigned. This is redundant with previous data, only the format change.
            + a MinList indicating the sequences with the lowest probabilities.
        """
        models = self._models
        farest = MinList(len(models) - 1)
        loglikelihoods = []
        for m in models:
            ll = m.logLikelihoods(sequences)
            loglikelihoods.append(ll)
        sortedSeqs = []
        modelsIndices = np.ndarray(shape=(len(sequences),), dtype=int)
        for m in range(len(models)):
            sortedSeqs.append([])
        for i in range(len(sequences)):
            max_log = loglikelihoods[0][i]
            max_for = 0
            for m in range(len(models)):
                if max_log < loglikelihoods[m][i]:
                    max_log = loglikelihoods[m][i]
                    max_for = m
            sortedSeqs[max_for].append(sequences[i])
            modelsIndices[i] = max_for
            farest.insert(max_log, i)
        return sortedSeqs, modelsIndices, farest

    def _predict(self, sequences: List[List[str]]) -> List[int]:
        labels = np.ndarray(shape=(len(sequences),), dtype=int)
        viterbi = self._bigModel.getMostProbableStates(sequences)
        for i in range(len(sequences)):
            state = viterbi[i][0]
            if state not in self._statesToCluster:
                self._statesToCluster[state] = len(self._statesToCluster)
            labels[i] = self._statesToCluster[state]
        return labels

    def checkFittedForTraceSet(self, traceSet: 'TraceSet') -> List[List[str]]:
        if self._bigModel is None:
            raise RuntimeError("The model was not trained before predicting the sequences.")
        seqs = self.tracesSetToSymbolSeq(traceSet)
        symbols = set()
        for seq in seqs:
            for symbol in seq:
                symbols.add(symbol)
        if not symbols.issubset(set(self._symbols)):
            raise RuntimeError("The model was not trained with the same symbols. Sequences with symbols {} cannot be evaluated".format(
                symbols.difference(set(self._symbols))))
        return seqs

    def predict(self, traces: 'TraceSet') -> List[int]:
        seqs = self.checkFittedForTraceSet(traces)
        return self._predict(seqs)

    def visualize(self, traceSet: 'TraceSet', colors: ColorList = ColorList(), simplifyModel=True) -> Graph:
        """
        Produces a graph with the model and draws the traces on it.

        Args:
            traceSet : the traces to draw on the graph.
            colors : the colors used to display each cluster.
            simplifyModel : draw a simplified model.
                However, the clustering is made on the real model and thus, the
                clusters (identified by colors) may be mixed on the simplified graph.
        """
        seqs = self.checkFittedForTraceSet(traceSet)
        labels = self._predict(seqs)
        byCluster = []
        for i in range(len(seqs)):
            cluster = labels[i]
            while len(byCluster) <= cluster:
                byCluster.append([])
            byCluster[cluster].append(seqs[i])
        model = self._bigModel
        if simplifyModel:
            models = []
            for model in self._models:
                model = model.copy()
                model = simplify(model, remove=True)
                models.append(model)
            model = HMM.assembleModels(models)
            model.train(self.tracesSetToSymbolSeq(self._fitSequences))
        graph = Graph(model)
        for cluster in byCluster:
            color = colors.pickColor()
            if len(cluster) != 0:
                graph.showSequences(cluster, color)
        return graph
