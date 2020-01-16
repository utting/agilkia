# -*- coding:utf8

import json
import csv
import pydot
from typing import List, Optional, Tuple, Dict, NoReturn, Iterable
import agilkia

from . utils import *


class ScanetteModel:
    """
    This class represents the internal behaviour of the scanette. It is based on a model written in dot format.
    """
    class Context:
        """
        This class is used to represent the internal variables of the model, as a complement to the states.
        """

        def __init__(self):
            self.dict = {}

        def isEnabled(self, preconds: List[str]) -> bool:
            """
            Indicate whether this context satisfy all the preconditions.
            """
            for precond in preconds:
                if not self._isEnabledOne(precond):
                    return False
            return True

        def _isEnabledOne(self, precond: str) -> bool:
            """
            Indicate whether the context satisfy the given preconditions.
            """
            for k, v in self.dict.items():
                if precond.startswith(k):
                    end = precond[len(k) + 1:]
                    op = end[0]
                    expected = end[2:]
                    if op == '=':
                        return v == int(expected)
                    if op == '>':
                        return v > int(expected)
            raise BaseException(
                "precondition '{}' is not handled".format(precond))

        def update(self, action: str):
            """
            update this context with the given action. The syntax is not clearly specified yet.
            """
            if action is None:
                pass
            elif action[-2:] == "++":
                var = action[:-2]
                if var not in self.dict:
                    self.dict[var] = 0
                self.dict[var] += 1
            elif action[-2:] == "--":
                var = action[:-2]
                self.dict[var] -= 1
            elif action == "produits = (panier > 12) ? 12 : panier":
                panier = self.dict["panier"]
                self.dict["produits"] = 12 if panier > 12 else panier
            else:
                raise BaseException(
                    "action '{}' is not handled".format(action))

        def __str__(self):
            entries = []
            for k, v in self.dict.items():
                entries.append("{}={}".format(k, v))
            return ",".join(entries)

    class State:
        """
        This class represent a state in the model. Basically, it contains the same informations than the dot file.
        """
        class Transition:
            def __init__(self, edge: pydot.Edge):
                self.edge = edge
                label = self.edge.get_label()
                if label[0] == '"' and label[-1] == '"':
                    label = label[1:-1]
                self.inputAction = label
                self.contextAction = None
                split = label.find(' / ')
                if split >= 0:
                    self.inputAction = label[:split]
                    self.contextAction = label[split + 3:]
                self.preconditions = []
                while self.inputAction.find('\\n') >= 0:
                    pos = self.inputAction.find('\\n')
                    precond = self.inputAction[:pos]
                    assert precond[0] == '[' and precond[-1] == ']'
                    precond = precond[1:-1]
                    self.preconditions.append(precond)
                    self.inputAction = self.inputAction[pos + 2:]

            def isEnabled(self, symbol: str, retCode: int, context: 'ScanetteModel.Context'):
                """
                Indicate whether this transition match the given action (method and return code) and the given context.
                """
                if self.inputAction == '*':
                    pass
                elif retCode is None:
                    if not symbol == self.inputAction:
                        return False
                elif symbol == 'C.payer' and retCode > 0:
                    if self.inputAction != 'C.payer > 0':
                        return False
                elif symbol == 'C.payer' and retCode < 0:
                    if self.inputAction != 'C.payer < 0':
                        return False
                else:
                    if not (symbol + " : " + str(int(retCode)) == self.inputAction):
                        return False
                return context.isEnabled(self.preconditions)

            def __str__(self):
                return self.edge.get_label()

        def __init__(self, node: pydot.Node, edges: List[pydot.Edge]):
            self.node = node
            self.transitions = []
            for edge in edges:
                if edge.get_source() == node.get_name():
                    self.transitions.append(__class__.Transition(edge))

        def __str__(self):
            return self.node.get_label()

        def getTransition(self, symbol: str, retCode: int, context) -> Optional['ScanetteModel.Transition']:
            """
            Get the transition which match the given symbol and context.
            Return `None` if no transition is found.
            """
            enabled = None
            for transition in self.transitions:
                if transition.isEnabled(symbol, retCode, context):
                    if enabled is not None:
                        print(
                            "Two transitions (at least) are available from {} with symbol {}, retCode {} and context {}".format(
                                self, symbol, retCode, context))
                        print("The available transitions are :")
                        print(" - {}".format(enabled))
                        print(" - {}".format(transition))
                        raise BaseException("two candidate transitions")
                    enabled = transition
            return enabled

        def getTransitions(self):
            return self.transitions

    def __init__(self, dotFile: str):
        """
        create a scanette model from the file containing the model in dot format
        """
        self._states = {}

        dots = pydot.graph_from_dot_file(dotFile)
        assert len(dots) == 1
        graph = dots.pop()
        self.graph = graph
        nodes = graph.get_node_list()
        edges = graph.get_edge_list()
        self.initial = None
        for node in nodes:
            state = __class__.State(node, edges)
            self._states[node.get_name()] = state
            if (node.get_shape() == 'diamond'):
                assert self.initial is None, "Only one initial state supported"
                self.initial = state
        assert self.initial is not None

    def getEndState(self, transition: 'ScanetteModel.State.Transition') -> 'ScanetteModel.State':
        """
        Get the state reached by the given transition.
        """
        return self._states[transition.edge.get_destination()]

    def getPath(self,
                trace: agilkia.Trace,
                verbose=False) -> Tuple[List['ScanetteModel.State'],
                                        List['ScanetteModel.State.Transition']]:
        """
        Compute which path in this model is used by the given trace.
        Returns the list of states crossed and the list of transitions used.
        """
        currentState = self.initial
        seenStates = [currentState]
        seenTransitions = []
        context = __class__.Context()

        for event in trace:
            if verbose:
                print(event)
            result = event.outputs.get('Status', None)
            if result == '?':
                result = None
            transition = currentState.getTransition(
                event.meta_data['object'][0].upper() + "." + event.action, result, context)
            if transition is None:
                print(
                    "error : no transition found from current state {} matching event {} and context {}.".format(
                        currentState, event, context))
                print("available transitions are :")
                for t in currentState.getTransitions():
                    print(" - {}".format(t))
                raise BaseException("No transition found")
            currentState = self.getEndState(transition)
            context.update(transition.contextAction)
            seenStates.append(currentState)
            seenTransitions.append(transition)

        return seenStates, seenTransitions

    class PlotableModel:
        """
        This class define a copy of the ScanettModel which aims to be exported in dot format after adding information in it.
        """

        def __init__(self, parent: 'ScanetteModel'):
            self.parent = parent
            self.graph = pydot.Dot()
            self.graph.obj_dict['attributes'] = parent.graph.obj_dict['attributes']
            for node in parent.graph.get_node_list():
                newNode = pydot.Node(
                    name=node.get_name(),
                    obj_dict=node.obj_dict.copy())
                newNode.obj_dict['attributes'] = node.obj_dict['attributes'].copy()
                self.graph.add_node(newNode)
            for edge in parent.graph.get_edge_list():
                newEdge = pydot.Edge(
                    src=edge.get_source(),
                    dst=edge.get_destination(),
                    obj_dict=edge.obj_dict.copy())
                newEdge.obj_dict['attributes'] = edge.obj_dict['attributes'].copy()
                self.graph.add_edge(newEdge)

        def plotSequence(self, sequence):
            """
            draw a sequence on the graph by adding new transitions.
            """
            path, _ = self.parent.getPath(sequence)
            for i in range(len(path) - 1):
                edge = pydot.Edge(
                    src=path[i].node.get_name(), dst=path[i + 1].node.get_name())
                self.graph.add_edge(edge)

        def countTransitions(self, sequences: Iterable[agilkia.Trace]
                             ) -> Tuple[Dict[str, Dict[str, Dict[str, int]]], int]:
            """
            count the transitions used by a set of traces.

            Returns a mapping of mapping of mapping sourceStateName -> destinationStateName -> transitionLabel -> count , and the maximum usage of a transition .
            If a transition is absent from the mapping, this means it is not used.
            """
            transitions = {}
            maxNb = 0
            for sequence in sequences:
                pathStates, pathTransitions = self.parent.getPath(sequence)
                for i in range(len(pathTransitions)):
                    src = pathStates[i].node.get_name()
                    dst = pathStates[i + 1].node.get_name()
                    fromSrc = transitions.get(src)
                    if fromSrc is None:
                        fromSrc = {}
                        transitions[src] = fromSrc
                    fromSrcToDst = fromSrc.get(dst)
                    if fromSrcToDst is None:
                        fromSrcToDst = {}
                        fromSrc[dst] = fromSrcToDst
                    label = pathTransitions[i].edge.get_label()
                    retainedLabel = None  # the actual label in the set, can be longer than original label
                    withLabel = None
                    for k, v in fromSrcToDst.items():
                        if k.startswith(label):
                            assert withLabel is None
                            withLabel = v
                            retainedLabel = k
                    if withLabel is None:
                        withLabel = 0
                        retainedLabel = label
                    withLabel += 1
                    fromSrcToDst[retainedLabel] = withLabel
                    maxNb = max(maxNb, withLabel)
            return transitions, maxNb

        def plotSequencesAfterLabel(self,
                                    sequences: Iterable[agilkia.Trace],
                                    openBalise: str = "",
                                    closeBalise: str = "") -> NoReturn:
            """
            Shows sequences on the model by writing after each transition label how many time this transition is used by the given sequences.

            Params:
                sequences: the sequences used to count the usage of each transition.
                openBalise: an html text to insert before the count of transition
                closeBalise: an html text to insert after the count of transition
            """
            transitions, maxNb = self.countTransitions(sequences)
            for src, fromSrc in transitions.items():
                for dst, toDst in fromSrc.items():
                    for label, occurences in toDst.items():
                        oldEdge = None
                        for e in self.graph.get_edge_list():
                            if e.get_source() == src and e.get_destination(
                            ) == dst and label.startswith(e.get_label()):
                                oldEdge = e
                        self.graph.del_edge(oldEdge)
                        newLabel = oldEdge.get_label()
                        if newLabel.startswith('"'):
                            newLabel = newLabel[1:-1]
                            newLabel = newLabel.replace('>', '&gt;')
                            newLabel = newLabel.replace('<', '&lt;')
                            newLabel = newLabel.replace('\\n', '<br/>')
                            newLabel = '<' + newLabel + '>'
                        newLabel = newLabel[:-1]
                        newLabel += '<br/>{}{}{}'.format(
                            openBalise, occurences, closeBalise)
                        newLabel += '>'
                        oldEdge.set_label(newLabel)

        def plotSequencesWithColors(self,
                                    sequences: Iterable[agilkia.Trace],
                                    colorMin: Color,
                                    colorMax: Optional[Color] = None) -> NoReturn:
            """
            plot sequence on this model by coloring the transitions used.

            Params:
                sequences: the sequences to show on this model.
                colorMin: the color for used transitions
                colorMax: if specified, the transitions will be colored with a gradient between colorMin and colorMax depending on how many time they are used.
            """
            if colorMax is None:
                colorMax = colorMin
            transitions, maxNb = self.countTransitions(sequences)
            for src, fromSrc in transitions.items():
                for dst, toDst in fromSrc.items():
                    for label, occurences in toDst.items():
                        oldEdge = None
                        for e in self.graph.get_edge_list():
                            if e.get_source() == src and e.get_destination(
                            ) == dst and label.startswith(e.get_label()):
                                oldEdge = e
                        color = colorMin.average(colorMax, float(occurences) / maxNb)
                        oldEdge.set_color(color.toHex())
                        oldEdge.set_fontcolor(color.toHex())

        def export(self, filename):
            self.graph.write_svg(filename)

    def plotableModel(self):
        return __class__.PlotableModel(self)
